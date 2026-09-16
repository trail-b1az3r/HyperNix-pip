//  CarPlayController.swift
//  HyperLink in the car.
//
//  What CarPlay lets an app be
//  ---------------------------
//  Not a small copy of the phone app. CarPlay gives you a fixed set of
//  templates — a list, a grid, an alert, a "now playing" — and refuses
//  anything else, because the system is the one deciding what is safe to
//  look at while driving and an app cannot be trusted to decide that
//  about itself. So this is a list of conversations and a way to talk to
//  one, and nothing else: no settings, no model picker, no transcript
//  scrolling.
//
//  Three ways in, and which one is available depends on the handbrake
//  -------------------------------------------------------------------
//  1. **Siri dictation**, always. `CPVoiceControlTemplate` is the
//     system's own microphone UI — it looks the same in every CarPlay
//     app, which is the point, and the driver never takes their eyes off
//     the road.
//  2. **Canned replies**, always. A list of six things somebody actually
//     says to a model from a car, each one tap.
//  3. **A keyboard**, only when the car says it is parked. The car
//     reports this through `CPSessionConfiguration.limitedUserInterfaces`
//     and changes it *while the app is running*, which is why the row is
//     rebuilt from a delegate callback rather than decided once at
//     launch — a value read in the driveway would offer a keyboard at
//     70mph. CarPlay also refuses to present the template while moving,
//     so this is the manners rather than the safety mechanism.
//
//  Replies are spoken, not shown
//  ------------------------------
//  A model's answer goes to `CPVoiceControlTemplate`'s spoken feedback
//  and to a short alert, never to a scrolling transcript. There is a
//  limit on how much text CarPlay will show at once and it exists for a
//  reason; `Speech.trim` (shared with the Siri intents) is what turns
//  six paragraphs of markdown into something worth hearing.

#if canImport(CarPlay)
import CarPlay
import Foundation

/// Drives the CarPlay interface.
///
/// Holds its own client rather than the app's `AppState`: the CarPlay
/// scene can be connected while the phone app is not running at all, and
/// an object that assumed otherwise would work on the test bench and
/// fail in the car.
@MainActor
final class CarPlayController: NSObject {
    private let interfaceController: CPInterfaceController
    private var client: HyperLinkClient?
    private var sessions: [ChatSession] = []
    /// The conversation the car is looking at, if any.
    private var activeSession: ChatSession?
    /// The conversation template currently on screen, kept so its rows
    /// can be rebuilt in place when the car's limits change. Rebuilding
    /// by pushing again would stack a second copy on the navigation
    /// stack; `updateSections` replaces the rows of the one that is
    /// already there.
    private weak var activeTemplate: CPListTemplate?

    /// The car's own account of what it will currently allow. Held as a
    /// property because it must be: a `CPSessionConfiguration` that is
    /// not retained stops delivering its callbacks.
    private lazy var sessionConfiguration = CPSessionConfiguration(delegate: self)

    /// What a driver says to a model, in a car. Short, because each of
    /// these is a row the eye has to land on and leave.
    private static let quickReplies = [
        "What's the weather?",
        "Summarise my last chat.",
        "What's on my calendar?",
        "Give me a fact.",
        "Carry on.",
        "Never mind.",
    ]

    init(interfaceController: CPInterfaceController) {
        self.interfaceController = interfaceController
        super.init()
        // The client cannot be built here. Restoring a pairing reads the
        // keychain and then configures an actor, so it is async, and
        // `init` is not. The task below does it, and puts something on
        // the screen first — a CarPlay scene with no root template
        // shows a blank panel, and "blank" is what a driver reads as
        // broken.
        Task { await connect() }
    }

    // MARK: - Root

    private func connect() async {
        await setRoot(loadingTemplate())
        // Touch the session configuration so it is built, and so its
        // delegate is registered, before anything asks what the car
        // currently allows. Left lazy until first read, the keyboard row
        // would be correct when a conversation was opened and would then
        // never change.
        _ = sessionConfiguration.limitedUserInterfaces
        client = await PairingStore.client()
        await showRoot()
    }

    private func showRoot() async {
        guard client != nil else {
            await setRoot(notPairedTemplate())
            return
        }
        await refreshSessions()
    }

    private func notPairedTemplate() -> CPListTemplate {
        let item = CPListItem(
            text: "Not set up yet",
            detailText: "Open HyperLink on your phone to pair it with your PC."
        )
        item.isEnabled = false
        let template = CPListTemplate(
            title: "HyperLink",
            sections: [CPListSection(items: [item])]
        )
        return template
    }

    private func loadingTemplate() -> CPListTemplate {
        let item = CPListItem(text: "Loading…", detailText: nil)
        item.isEnabled = false
        return CPListTemplate(
            title: "HyperLink", sections: [CPListSection(items: [item])]
        )
    }

    private func refreshSessions() async {
        guard let client else { return }
        do {
            // Newest first and only a handful. A CarPlay list scrolls,
            // and a driver scrolling a list of forty conversations is
            // the thing this interface exists to avoid.
            sessions = Array(
                (try await client.sessions())
                    .sorted { $0.updatedAt > $1.updatedAt }
                    .prefix(8)
            )
        } catch {
            await setRoot(errorTemplate(IntentBridge.explain(error)))
            return
        }
        await setRoot(sessionListTemplate())
    }

    private func sessionListTemplate() -> CPListTemplate {
        var items: [CPListItem] = []

        // First row, always: start talking without picking anything.
        let ask = CPListItem(
            text: "Ask something",
            detailText: "Speak a question to your PC"
        )
        ask.handler = { [weak self] _, completion in
            Task { @MainActor in
                await self?.beginDictation(sessionID: nil)
                completion()
            }
        }
        items.append(ask)

        for session in sessions {
            let item = CPListItem(
                text: session.title,
                detailText: "\(session.messageCount) message"
                    + (session.messageCount == 1 ? "" : "s")
            )
            item.handler = { [weak self] _, completion in
                Task { @MainActor in
                    await self?.open(session)
                    completion()
                }
            }
            items.append(item)
        }

        let template = CPListTemplate(
            title: "HyperLink",
            sections: [CPListSection(items: items)]
        )
        template.emptyViewTitleVariants = ["No conversations"]
        template.emptyViewSubtitleVariants = [
            "Start one from your phone, or say something here."
        ]
        return template
    }

    private func errorTemplate(_ message: String) -> CPListTemplate {
        let item = CPListItem(text: "Can't reach your PC", detailText: message)
        item.isEnabled = false
        return CPListTemplate(
            title: "HyperLink", sections: [CPListSection(items: [item])]
        )
    }

    // MARK: - One conversation

    private func open(_ session: ChatSession) async {
        activeSession = session
        let template = CPListTemplate(
            title: session.title, sections: sections(for: session)
        )
        activeTemplate = template
        _ = try? await interfaceController.pushTemplate(template, animated: true)
    }

    /// The rows of one conversation.
    ///
    /// A function rather than inline in `open`, because the keyboard row
    /// comes and goes with the handbrake and the list has to be built
    /// again from the delegate callback below.
    private func sections(for session: ChatSession) -> [CPListSection] {
        var items: [CPListItem] = []

        let speak = CPListItem(text: "Speak a message", detailText: nil)
        speak.handler = { [weak self] _, completion in
            Task { @MainActor in
                await self?.beginDictation(sessionID: session.sessionID)
                completion()
            }
        }
        items.append(speak)

        let read = CPListItem(text: "Read the last message", detailText: nil)
        read.handler = { [weak self] _, completion in
            Task { @MainActor in
                await self?.readLatest(sessionID: session.sessionID)
                completion()
            }
        }
        items.append(read)

        // Only when the car is stopped. See `keyboardAvailable`.
        if keyboardAvailable {
            let type = CPListItem(
                text: "Type a message",
                detailText: "Available because the car is parked"
            )
            type.handler = { [weak self] _, completion in
                Task { @MainActor in
                    await self?.typeMessage(sessionID: session.sessionID)
                    completion()
                }
            }
            items.append(type)
        }

        let quick = Self.quickReplies.map { text -> CPListItem in
            let item = CPListItem(text: text, detailText: nil)
            item.handler = { [weak self] _, completion in
                Task { @MainActor in
                    await self?.send(text, sessionID: session.sessionID)
                    completion()
                }
            }
            return item
        }

        return [
            CPListSection(items: items),
            CPListSection(items: quick, header: "Quick", sectionIndexTitle: nil),
        ]
    }

    /// Whether the car will let us put a keyboard on screen.
    ///
    /// Read every time rather than cached: the answer changes while the
    /// app is running — that is the whole point of it — and a value read
    /// at launch would offer a keyboard at 70mph because the car was
    /// stationary in the driveway when the scene connected.
    ///
    /// The car decides, not this code and not a speed this code tries to
    /// work out for itself. `.keyboard` appearing in `limitedUserInterfaces`
    /// is the car saying "not now"; CarPlay additionally refuses to put a
    /// keyboard up while moving, so this is what stops the app offering a
    /// row that would then be rejected — which reads as a bug to the
    /// person tapping it — rather than the safety mechanism itself.
    private var keyboardAvailable: Bool {
        !sessionConfiguration.limitedUserInterfaces.contains(.keyboard)
    }

    // MARK: - Talking

    /// The system's own microphone UI.
    ///
    /// `CPVoiceControlTemplate` rather than a bespoke recorder: it looks
    /// identical in every CarPlay app, which is what lets a driver use
    /// it without learning it, and it is the only speech affordance
    /// CarPlay will show.
    private func beginDictation(sessionID: String?) async {
        let listening = CPVoiceControlState(
            identifier: "listening",
            titleVariants: ["Listening…", "Listening"],
            image: nil,
            repeats: true
        )
        let thinking = CPVoiceControlState(
            identifier: "thinking",
            titleVariants: ["Thinking…", "Thinking"],
            image: nil,
            repeats: true
        )
        let template = CPVoiceControlTemplate(
            voiceControlStates: [listening, thinking]
        )
        _ = try? await interfaceController.presentTemplate(template, animated: true)

        let heard = await Dictation.listen()
        guard let heard, !heard.isEmpty else {
            await dismiss()
            return
        }
        template.activateVoiceControlState(withIdentifier: "thinking")
        await send(heard, sessionID: sessionID, alreadyPresenting: true)
    }

    /// The parked-only path.
    ///
    /// `CPSearchTemplate` because it is the only public CarPlay template
    /// that carries a keyboard. The first draft of this used a
    /// `CPTextInputTemplate`, which does not exist — CarPlay has no
    /// general-purpose text-entry template, and inventing one is not
    /// something a reviewer or a Python test could catch; it failed at
    /// EmitSwiftModule on the first CI run that had an SDK.
    ///
    /// Search is a reasonable fit rather than a workaround: the keyboard
    /// is the system's, the car gates it exactly as it gates any other,
    /// and the search button is the send button. The single result row
    /// shows the whole line before it goes, which is worth having for
    /// somebody who glanced away mid-word.
    ///
    /// The row that leads here is hidden rather than disabled, because a
    /// disabled row somebody keeps tapping is worse than one that is not
    /// there.
    private func typeMessage(sessionID: String) async {
        pendingTypedSessionID = sessionID
        let search = CPSearchTemplate()
        search.delegate = self
        _ = try? await interfaceController.presentTemplate(search, animated: true)
    }

    /// Which conversation a typed message belongs to.
    ///
    /// Held rather than passed because the delegate callbacks below are
    /// the system's and carry no context of ours.
    private var pendingTypedSessionID: String?
    /// The most recent thing typed, so the search button has something to
    /// send when it is pressed without a row being tapped.
    private var pendingTypedText: String = ""

    func textEntered(_ text: String) {
        let sessionID = pendingTypedSessionID
        pendingTypedSessionID = nil
        pendingTypedText = ""
        Task { await send(text, sessionID: sessionID) }
    }

    private func send(
        _ text: String, sessionID: String?, alreadyPresenting: Bool = false
    ) async {
        guard let client else { return }
        do {
            let target: String
            if let sessionID {
                target = sessionID
            } else {
                target = try await client.createSession(title: "Car").sessionID
            }
            let reply = try await client.chat(sessionID: target, content: text)
            await speak(
                Speech.trim(reply.assistantMessage.content), wasPresenting: alreadyPresenting
            )
        } catch {
            await speak(IntentBridge.explain(error), wasPresenting: alreadyPresenting)
        }
    }

    private func readLatest(sessionID: String) async {
        guard let client else { return }
        do {
            let history = try await client.messages(in: sessionID)
            guard let last = history.last else {
                await speak("That chat has no messages yet.", wasPresenting: false)
                return
            }
            await speak(Speech.trim(last.content), wasPresenting: false)
        } catch {
            await speak(IntentBridge.explain(error), wasPresenting: false)
        }
    }

    /// Say it, and show a short version.
    ///
    /// Both, because a passenger reads and a driver listens, and the
    /// car may be in either state. The alert is capped hard — CarPlay
    /// will truncate it anyway and a truncated paragraph is worse than a
    /// sentence that fits.
    private func speak(_ text: String, wasPresenting: Bool) async {
        Dictation.say(text)
        if wasPresenting { await dismiss() }
        let shown = text.count > 120 ? String(text.prefix(117)) + "…" : text
        let alert = CPAlertTemplate(
            titleVariants: [shown, String(shown.prefix(60))],
            actions: [
                CPAlertAction(title: "Done", style: .default) { [weak self] _ in
                    Task { @MainActor in await self?.dismiss() }
                },
                CPAlertAction(title: "Ask again", style: .default) { [weak self] _ in
                    Task { @MainActor in
                        await self?.dismiss()
                        await self?.beginDictation(sessionID: self?.activeSession?.sessionID)
                    }
                },
            ]
        )
        _ = try? await interfaceController.presentTemplate(alert, animated: true)
    }

    // MARK: - Template plumbing

    private func setRoot(_ template: CPTemplate) async {
        _ = try? await interfaceController.setRootTemplate(template, animated: false)
    }

    private func dismiss() async {
        _ = try? await interfaceController.dismissTemplate(animated: true)
    }
}

// MARK: - Typing, when the car allows it

/// `CPSearchTemplate` is CarPlay's keyboard. The "result" is the one line
/// being composed: tapping it sends, and so does the search button,
/// because somebody should not have to work out which of the two is the
/// real one.
extension CarPlayController: CPSearchTemplateDelegate {
    /// `nonisolated` throughout, and the hop is the `Task`.
    ///
    /// Same reason as `CPSessionConfigurationDelegate` below: the
    /// protocol is `@objc` and carries no actor annotation, so a
    /// `@MainActor` method satisfying it is a concurrency warning today
    /// and an error under a stricter setting later. CarPlay does call
    /// these on the main thread; that is a fact about the runtime, not
    /// something the type system knows.
    nonisolated func searchTemplate(
        _ searchTemplate: CPSearchTemplate,
        updatedSearchText searchText: String,
        completionHandler: @escaping ([CPListItem]) -> Void
    ) {
        // The handler is answered from the parameter rather than from
        // stored state, so the keyboard stays responsive without waiting
        // on an actor hop for every keystroke.
        let trimmed = searchText.trimmingCharacters(in: .whitespaces)
        completionHandler(
            trimmed.isEmpty ? [] : [CPListItem(text: "Send", detailText: searchText)]
        )
        Task { @MainActor [weak self] in
            self?.pendingTypedText = searchText
        }
    }

    nonisolated func searchTemplate(
        _ searchTemplate: CPSearchTemplate,
        selectedResult item: CPListItem,
        completionHandler: @escaping () -> Void
    ) {
        let typed = item.detailText
        completionHandler()
        Task { @MainActor [weak self] in
            guard let self else { return }
            let text = typed ?? self.pendingTypedText
            await self.dismiss()
            self.textEntered(text)
        }
    }

    nonisolated func searchTemplateSearchButtonPressed(
        _ searchTemplate: CPSearchTemplate
    ) {
        Task { @MainActor [weak self] in
            guard let self else { return }
            let text = self.pendingTypedText
            guard !text.trimmingCharacters(in: .whitespaces).isEmpty else { return }
            await self.dismiss()
            self.textEntered(text)
        }
    }
}

// MARK: - The car changing its mind

/// Rebuild the open conversation when the car's limits change.
///
/// This is the half that makes `keyboardAvailable` worth reading per
/// use. The driver stops at lights, the car lifts the keyboard limit,
/// and the row has to appear on the screen that is already up; they pull
/// away and it has to go again. Without this the list would be whatever
/// was true when it was pushed.
extension CarPlayController: CPSessionConfigurationDelegate {
    /// `nonisolated` because the protocol is not annotated for an actor
    /// and this class is `@MainActor`; the hop back is the `Task`.
    nonisolated func sessionConfiguration(
        _ sessionConfiguration: CPSessionConfiguration,
        limitedUserInterfacesChanged limitedUserInterfaces: CPLimitableUserInterface
    ) {
        Task { @MainActor [weak self] in
            guard let self,
                  let session = self.activeSession,
                  let template = self.activeTemplate
            else { return }
            template.updateSections(self.sections(for: session))
        }
    }
}

/// Speech in and speech out.
///
/// Separated from the controller so the CarPlay code stays about
/// templates, and so the phone app can use the same speech output for
/// its own hands-free mode.
@MainActor
enum Dictation {
    /// Listen, and return what was said.
    ///
    /// Returns nil when permission was refused or nothing was heard.
    /// Both are ordinary outcomes in a car — a passenger talking over
    /// the prompt, a microphone the phone has not been given — and
    /// neither should be an error the driver has to deal with.
    static func listen() async -> String? {
        #if canImport(Speech)
        return await SpeechRecogniser.shared.listenOnce()
        #else
        return nil
        #endif
    }

    static func say(_ text: String) {
        #if canImport(Speech)
        SpeechSynthesis.shared.say(text)
        #endif
    }
}
#endif
