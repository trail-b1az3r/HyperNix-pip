//  Speech.swift
//  Listening and talking, for the car and for hands-free on the phone.
//
//  Two singletons, because both wrap a system resource that there is
//  exactly one of: the microphone and the speech synthesiser. Making
//  these per-view would mean two recognisers fighting over the audio
//  session, which fails in a way that looks like the microphone not
//  working.
//
//  Permissions
//  -----------
//  Speech recognition needs two separate grants — `NSSpeechRecognition`
//  and `NSMicrophone` — and a refusal of either is an ordinary outcome
//  rather than an error. In a car it is very ordinary: the driver may
//  never have opened the phone app, and the first time they use CarPlay
//  is not the moment to put a permission dialogue in front of them. So
//  `listenOnce` returns nil and the caller says something useful out
//  loud instead.
//
//  On-device where possible
//  -------------------------
//  `requiresOnDeviceRecognition` is set when the locale supports it.
//  The whole premise of HyperLink is that the model is on your own PC;
//  routing the *speech* through Apple's servers on the way there would
//  undercut that for no benefit the driver can see.

#if canImport(Speech)
import AVFoundation
import Foundation
import Speech

/// Microphone in, text out. One shot per call.
@MainActor
final class SpeechRecogniser {
    static let shared = SpeechRecogniser()

    private let engine = AVAudioEngine()
    private var recogniser: SFSpeechRecognizer?
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var task: SFSpeechRecognitionTask?

    /// How long to keep listening after the last word.
    ///
    /// Short: a driver expects the microphone to close when they stop
    /// talking, and one that waits three seconds reads as broken.
    private let silenceTimeout: TimeInterval = 1.4
    /// A hard ceiling, so a noisy car does not leave this listening
    /// until the battery goes.
    private let maximum: TimeInterval = 30

    private init() {
        recogniser = SFSpeechRecognizer(locale: .current)
            ?? SFSpeechRecognizer(locale: Locale(identifier: "en_US"))
    }

    /// Ask once, if we have not already.
    ///
    /// Returns whether both grants are in hand. Never prompts twice —
    /// `requestAuthorization` is a no-op after the first answer, and a
    /// caller that treated a refusal as "ask again" would produce an app
    /// that appears to loop.
    func authorised() async -> Bool {
        let speech = await withCheckedContinuation { continuation in
            SFSpeechRecognizer.requestAuthorization { status in
                continuation.resume(returning: status == .authorized)
            }
        }
        guard speech else { return false }
        return await withCheckedContinuation { continuation in
            AVAudioApplication.requestRecordPermission { granted in
                continuation.resume(returning: granted)
            }
        }
    }

    /// Listen until the speaker stops, and return what they said.
    ///
    /// nil for a refused permission, a recogniser that is not available
    /// in this locale, or silence. All three are ordinary in a car.
    func listenOnce() async -> String? {
        guard await authorised(), let recogniser, recogniser.isAvailable else {
            return nil
        }
        stop()

        do {
            let session = AVAudioSession.sharedInstance()
            // `.duckOthers` rather than interrupting: the driver may be
            // listening to something, and taking the audio away entirely
            // to hear one sentence is worse than turning it down.
            try session.setCategory(
                .playAndRecord, mode: .measurement, options: [.duckOthers, .defaultToSpeaker]
            )
            try session.setActive(true, options: .notifyOthersOnDeactivation)
        } catch {
            return nil
        }

        let request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = true
        if recogniser.supportsOnDeviceRecognition {
            // See the header. The point of this app is that nothing
            // leaves the phone except to the user's own PC.
            request.requiresOnDeviceRecognition = true
        }
        self.request = request

        let input = engine.inputNode
        input.installTap(
            onBus: 0, bufferSize: 1024, format: input.outputFormat(forBus: 0)
        ) { buffer, _ in
            request.append(buffer)
        }
        engine.prepare()
        do {
            try engine.start()
        } catch {
            stop()
            return nil
        }

        // Everything mutable lives on the main actor, in `ListenState`.
        //
        // `recognitionTask`'s handler is called on an arbitrary queue,
        // so the transcript, the resume flag and the silence timer
        // cannot be captured `var`s in this closure. That is a data race
        // and, under strict concurrency, two compile errors: "reference
        // to property 'silenceTimeout' in closure requires explicit use
        // of 'self'" and "call to main actor-isolated instance method
        // 'stop()' in a synchronous nonisolated context". The handler
        // reads value types out of the result and hops.
        let silence = silenceTimeout
        let ceiling = maximum

        return await withCheckedContinuation { continuation in
            let listening = ListenState(continuation)

            // The hard ceiling, so a noisy car does not leave this
            // listening until the battery goes.
            Task { @MainActor [weak self] in
                try? await Task.sleep(for: .seconds(ceiling))
                guard !Task.isCancelled else { return }
                self?.stop()
                listening.finish()
            }

            task = recogniser.recognitionTask(with: request) { [weak self] result, error in
                // Read out of the result *here*. SFSpeechRecognitionResult
                // is not Sendable and must not cross to the main actor;
                // a String and two Bools may.
                let text = result?.bestTranscription.formattedString
                let isFinal = result?.isFinal ?? false
                let failed = error != nil

                Task { @MainActor in
                    if let text, !text.isEmpty { listening.heard = text }
                    if isFinal || failed {
                        // Whatever was heard before an error is still
                        // worth having — a recognition failure at the
                        // end of a sentence should not throw the
                        // sentence away.
                        self?.stop()
                        listening.finish()
                        return
                    }
                    // Restart the silence countdown on every word. When
                    // it fires, the speaker has stopped.
                    listening.silenceTimer?.cancel()
                    listening.silenceTimer = Task { @MainActor in
                        try? await Task.sleep(for: .seconds(silence))
                        guard !Task.isCancelled else { return }
                        self?.stop()
                        listening.finish()
                    }
                }
            }
        }
    }

    func stop() {
        task?.cancel()
        task = nil
        request?.endAudio()
        request = nil
        if engine.isRunning {
            engine.stop()
            engine.inputNode.removeTap(onBus: 0)
        }
        try? AVAudioSession.sharedInstance().setActive(
            false, options: .notifyOthersOnDeactivation
        )
    }
}

/// The mutable half of one `listenOnce`, pinned to the main actor.
///
/// Exists because the recognition handler runs on an arbitrary queue
/// and the continuation can be resumed from three places — the handler,
/// the silence timer and the hard ceiling — exactly one of which may
/// win. Keeping all of it on one actor makes "resumed twice" impossible
/// rather than unlikely, and resuming a continuation twice is a crash.
@MainActor
private final class ListenState {
    /// What has been heard so far.
    var heard = ""
    var silenceTimer: Task<Void, Never>?

    private var continuation: CheckedContinuation<String?, Never>?

    init(_ continuation: CheckedContinuation<String?, Never>) {
        self.continuation = continuation
    }

    /// Resume with whatever was heard. The second call does nothing.
    func finish() {
        guard let continuation else { return }
        self.continuation = nil
        silenceTimer?.cancel()
        silenceTimer = nil
        continuation.resume(returning: heard.isEmpty ? nil : heard)
    }
}

/// Text out.
@MainActor
final class SpeechSynthesis {
    static let shared = SpeechSynthesis()

    private let synthesiser = AVSpeechSynthesizer()

    private init() {}

    func say(_ text: String) {
        guard !text.isEmpty else { return }
        // Interrupt whatever was being said. In a car the newest answer
        // is the one wanted, and queueing means the driver hears the
        // answer to the question before last.
        if synthesiser.isSpeaking {
            synthesiser.stopSpeaking(at: .immediate)
        }
        do {
            try AVAudioSession.sharedInstance().setCategory(
                .playback, mode: .spokenAudio, options: [.duckOthers]
            )
            try AVAudioSession.sharedInstance().setActive(true)
        } catch {
            // Speaking without having got the session is still worth
            // attempting -- it usually works, and the alternative is
            // silence.
        }
        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = AVSpeechSynthesisVoice(language: Locale.current.identifier)
            ?? AVSpeechSynthesisVoice(language: "en-US")
        // A shade above default. The system default is pitched for
        // reading a long article; an answer to a question wants to be
        // over with.
        utterance.rate = AVSpeechUtteranceDefaultSpeechRate * 1.05
        synthesiser.speak(utterance)
    }

    func stop() {
        synthesiser.stopSpeaking(at: .immediate)
    }
}
#endif
