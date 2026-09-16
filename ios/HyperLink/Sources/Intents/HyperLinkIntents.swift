//  HyperLinkIntents.swift
//  HyperLink — what Siri can do.
//
//  Four intents, and an AppShortcutsProvider that gives each one the
//  phrases people will actually say:
//
//      "Siri, ask HyperLink what's the weather"
//      "Siri, load the model Gemma 4 E2B on blazeindustries in HyperLink"
//      "Siri, read me the HyperLink chat"
//      "Siri, read me the most recent message in HyperLink from Mason"
//
//  App Intents, not SiriKit
//  ------------------------
//  SiriKit is a fixed set of domains — messaging, payments, workouts —
//  and "ask a language model on my PC" is not one of them. App Intents
//  is the framework for an app's own verbs, it is what Shortcuts and the
//  Action button read, and it is what the newer Siri dispatches through.
//  An intent written here is reachable by voice, from Shortcuts, from a
//  Home Screen widget and from the Action button without any of those
//  being implemented separately.
//
//  Why every one of these returns a spoken string
//  ----------------------------------------------
//  `ProvidesDialog` is what makes an intent usable hands-free. An intent
//  that returns a value with no dialogue works in Shortcuts and does
//  nothing useful in a car, which is the place these are most wanted.
//
//  The one thing they do not do
//  -----------------------------
//  None of them opens the app. `openAppWhenRun` is false throughout,
//  because the whole point of asking Siri from a car dock or a watch is
//  that the phone stays where it is. Where an intent genuinely cannot
//  finish without the app — there is no paired server yet — it says so
//  rather than launching into a pairing screen the driver cannot use.

import AppIntents
import Foundation

// MARK: - Shared plumbing

/// The hands-free half of talking to the server.
///
/// Intents run in a separate process from the app, so they cannot reach
/// `AppState`; the client comes from `PairingStore`, which is the same
/// stored pairing the app restores at launch. What lives here is the
/// other half — turning a failure into a sentence worth hearing.
enum IntentBridge {
    /// The message a hands-free failure should say out loud.
    ///
    /// Deliberately specific about *which* thing is wrong. "HyperLink is
    /// not set up", "your PC is not answering" and "it is no longer
    /// paired" need different responses from whoever is listening, and
    /// only some of them can be acted on from the driver's seat. The
    /// wording is also shorter and plainer than `errorDescription`,
    /// which is written to be read on a screen: the endpoint list in
    /// `.noReachableEndpoint` is genuinely useful in Settings and is
    /// four Tailscale hostnames read aloud at a junction.
    static func explain(_ error: Error) -> String {
        guard let linkError = error as? HyperLinkError else {
            return "Something went wrong talking to your PC."
        }
        switch linkError {
        case .notConfigured:
            return notPaired
        case .noReachableEndpoint:
            return "I could not reach your PC. It may be asleep, or off the network."
        case .unauthorized:
            return "HyperLink is not paired with your PC any more. Open the app to pair it again."
        case let .serverError(_, message, _):
            // The server's own words. It is the only participant that
            // knows things like "no model is loaded", and that sentence
            // is exactly what the listener needs.
            return Speech.trim(message)
        case .transport:
            return "The connection to your PC dropped."
        case .decoding:
            return "Your PC answered with something this app could not read. It may need updating."
        }
    }

    static let notPaired =
        "HyperLink is not paired with a PC yet. Open the app on your phone to set it up."
}

// MARK: - Ask

/// "Siri, ask HyperLink what's the weather"
///
/// The one people will use most, and the reason the phrase list below
/// puts `${applicationName}` in the middle rather than at the front:
/// "ask HyperLink <anything>" is how an English sentence wants to go.
struct AskHyperLinkIntent: AppIntent {
    static let title: LocalizedStringResource = "Ask HyperLink"
    static let description = IntentDescription(
        "Send a question to a model running on your PC and hear the answer.",
        categoryName: "Chat"
    )
    /// False on purpose — see the header. Asking from a car dock should
    /// not put a chat transcript on the phone's screen.
    static let openAppWhenRun = false

    @Parameter(
        title: "Question",
        description: "What to ask.",
        requestValueDialog: "What would you like to ask?"
    )
    var question: String

    @MainActor
    func perform() async throws -> some IntentResult & ProvidesDialog {
        guard let client = await PairingStore.client() else {
            return .result(dialog: IntentDialog(stringLiteral: IntentBridge.notPaired))
        }
        do {
            // A fresh session per ask, not the last one the phone had
            // open. A spoken question is a question, not a continuation
            // of whatever was on screen yesterday — and threading it
            // into an existing chat would put voice turns in the middle
            // of a typed conversation.
            let session = try await client.createSession(title: "Siri: \(question.prefix(40))")
            let reply = try await client.chat(sessionID: session.sessionID, content: question)
            let spoken = Speech.trim(reply.assistantMessage.content)
            return .result(dialog: IntentDialog(stringLiteral: spoken))
        } catch {
            return .result(dialog: IntentDialog(stringLiteral: IntentBridge.explain(error)))
        }
    }
}

// MARK: - Load a model

/// "Siri, load the model Gemma 4 E2B on blazeindustries in HyperLink"
///
/// Two parameters because that is how the sentence is said: a model and,
/// optionally, whose it is. `blazeindustries/gemma-4-e2b` is the
/// canonical form and nobody says a slash out loud.
struct LoadModelIntent: AppIntent {
    static let title: LocalizedStringResource = "Load a model"
    static let description = IntentDescription(
        "Load a model on your PC, ready for the next question.",
        categoryName: "Models"
    )
    static let openAppWhenRun = false

    @Parameter(
        title: "Model",
        description: "The model's name, e.g. Gemma 4 E2B.",
        requestValueDialog: "Which model?"
    )
    var model: String

    @Parameter(
        title: "Owner",
        description: "Who publishes it, e.g. blazeindustries.",
        default: ""
    )
    var owner: String

    @MainActor
    func perform() async throws -> some IntentResult & ProvidesDialog {
        guard let client = await PairingStore.client() else {
            return .result(dialog: IntentDialog(stringLiteral: IntentBridge.notPaired))
        }
        do {
            // The catalogue, not the LM Studio bridge. Asking Siri to
            // switch to a model sitting in ~/.hypernix/models used to
            // get back "your PC has no models loaded", because the only
            // thing this looked at was what LM Studio had open.
            let available = try await client.modelCatalogue().models
            guard let match = ModelMatch.best(
                spoken: model, owner: owner, in: available.map(\.modelID)
            ) else {
                let names = available.prefix(3).map { shortModelName($0.modelID) }
                let suggestion = names.isEmpty
                    ? "Your PC has no models."
                    : "I could not find that one. You have \(Speech.list(names))."
                return .result(dialog: IntentDialog(stringLiteral: suggestion))
            }
            let session = try await client.createSession(title: "Siri", modelID: match)
            _ = session
            return .result(dialog: IntentDialog(
                stringLiteral: "\(shortModelName(match)) is ready."
            ))
        } catch {
            return .result(dialog: IntentDialog(stringLiteral: IntentBridge.explain(error)))
        }
    }
}

/// Matching a spoken model name against what the server has.
///
/// Speech gives you "gemma 4 e2b" for `gemma-4-e2b`, "blaze industries"
/// for `blazeindustries`, and occasionally "gemma for e two b". So the
/// comparison strips everything that is not a letter or a digit from
/// both sides before looking, which turns all three of those into the
/// same string.
enum ModelMatch {
    static func normalise(_ text: String) -> String {
        text.lowercased().filter { $0.isLetter || $0.isNumber }
    }

    static func best(spoken: String, owner: String, in available: [String]) -> String? {
        let wanted = normalise(spoken)
        guard !wanted.isEmpty else { return nil }
        let wantedOwner = normalise(owner)

        let candidates = available.filter { id in
            wantedOwner.isEmpty || normalise(id).contains(wantedOwner)
        }
        // Exact first, then contains. An exact match on a shorter name
        // must beat a substring match on a longer one -- "gemma 4"
        // should find `gemma-4` rather than `gemma-4-instruct-abliterated`.
        if let exact = candidates.first(where: {
            normalise(shortModelName($0)) == wanted
        }) {
            return exact
        }
        return candidates
            .filter { normalise($0).contains(wanted) }
            .min { $0.count < $1.count }
    }
}

// MARK: - Read a chat

/// "Siri, read me the HyperLink chat"
/// "Siri, read me the most recent message in HyperLink from Mason"
struct ReadChatIntent: AppIntent {
    static let title: LocalizedStringResource = "Read a chat"
    static let description = IntentDescription(
        "Hear the latest from a conversation on your PC.",
        categoryName: "Chat"
    )
    static let openAppWhenRun = false

    @Parameter(
        title: "Chat",
        description: "Which conversation. Leave empty for the most recent.",
        default: ""
    )
    var chat: String

    @Parameter(
        title: "Only the last message",
        description: "Read one message rather than the last few.",
        default: false
    )
    var latestOnly: Bool

    @MainActor
    func perform() async throws -> some IntentResult & ProvidesDialog {
        guard let client = await PairingStore.client() else {
            return .result(dialog: IntentDialog(stringLiteral: IntentBridge.notPaired))
        }
        do {
            let sessions = try await client.sessions()
            guard let session = ChatMatch.best(spoken: chat, in: sessions) else {
                return .result(dialog: IntentDialog(
                    stringLiteral: chat.isEmpty
                        ? "You have no conversations yet."
                        : "I could not find a chat called \(chat)."
                ))
            }
            let history = try await client.messages(in: session.sessionID)
            guard !history.isEmpty else {
                return .result(dialog: IntentDialog(
                    stringLiteral: "\(session.title) has no messages yet."
                ))
            }
            let spoken = latestOnly
                ? Speech.trim(history.last?.content ?? "")
                // Three, not the whole transcript. Siri reading forty
                // messages at a junction is worse than reading none.
                : Speech.transcript(history.suffix(3))
            return .result(dialog: IntentDialog(stringLiteral: spoken))
        } catch {
            return .result(dialog: IntentDialog(stringLiteral: IntentBridge.explain(error)))
        }
    }
}

enum ChatMatch {
    static func best(spoken: String, in sessions: [ChatSession]) -> ChatSession? {
        let wanted = ModelMatch.normalise(spoken)
        // No name said: the most recently updated one, which is what
        // "the HyperLink chat" means when somebody has one open.
        guard !wanted.isEmpty else {
            return sessions.max { $0.updatedAt < $1.updatedAt }
        }
        if let exact = sessions.first(where: {
            ModelMatch.normalise($0.title) == wanted
        }) {
            return exact
        }
        return sessions.first { ModelMatch.normalise($0.title).contains(wanted) }
    }
}

// MARK: - Send

/// "Siri, send a message in HyperLink" — dictation into the open chat.
struct SendMessageIntent: AppIntent {
    static let title: LocalizedStringResource = "Send a message"
    static let description = IntentDescription(
        "Dictate a message into a HyperLink conversation.",
        categoryName: "Chat"
    )
    static let openAppWhenRun = false

    @Parameter(
        title: "Message",
        requestValueDialog: "What should I send?"
    )
    var message: String

    @Parameter(title: "Chat", default: "")
    var chat: String

    @MainActor
    func perform() async throws -> some IntentResult & ProvidesDialog {
        guard let client = await PairingStore.client() else {
            return .result(dialog: IntentDialog(stringLiteral: IntentBridge.notPaired))
        }
        do {
            let sessions = try await client.sessions()
            let session = ChatMatch.best(spoken: chat, in: sessions)
                ?? (try await client.createSession(title: "Siri"))
            let reply = try await client.chat(sessionID: session.sessionID, content: message)
            return .result(dialog: IntentDialog(
                stringLiteral: Speech.trim(reply.assistantMessage.content)
            ))
        } catch {
            return .result(dialog: IntentDialog(stringLiteral: IntentBridge.explain(error)))
        }
    }
}

// MARK: - Making text speakable

/// Turning a model's reply into something worth hearing.
///
/// A reply is written to be read: it has code fences, markdown emphasis,
/// bullet lists and, often, six paragraphs. Read aloud verbatim that is
/// "asterisk asterisk important asterisk asterisk" and two minutes of
/// talking. These are the smallest set of fixes that make the difference.
enum Speech {
    /// How much of a reply to speak before stopping.
    ///
    /// About forty seconds at Siri's rate. Past that, whoever asked has
    /// stopped listening, and a voice assistant that cannot be
    /// interrupted is worse than one that stops early.
    static let limit = 700

    static func trim(_ text: String) -> String {
        var spoken = text

        // Code blocks. Read aloud a fenced block is unintelligible, and
        // saying that one is there is more use than reading it.
        spoken = spoken.replacingOccurrences(
            of: "```[\\s\\S]*?```",
            with: " (there's a code block here — it's on your phone) ",
            options: .regularExpression
        )
        // Markdown emphasis and headings, which have no spoken form.
        for pattern in ["\\*\\*", "\\*", "__", "`", "^#+\\s*"] {
            spoken = spoken.replacingOccurrences(
                of: pattern, with: "", options: [.regularExpression]
            )
        }
        // A bullet reads as a pause, which is what it means.
        spoken = spoken.replacingOccurrences(
            of: "^\\s*[-*+]\\s+", with: ", ", options: [.regularExpression]
        )
        spoken = spoken
            .replacingOccurrences(of: "\n\n", with: ". ")
            .replacingOccurrences(of: "\n", with: " ")
            .replacingOccurrences(of: "  ", with: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)

        if spoken.count > limit {
            // Cut at a sentence rather than mid-word, when there is one
            // near enough to the limit to be worth using.
            let head = String(spoken.prefix(limit))
            if let stop = head.lastIndex(where: { ".!?".contains($0) }),
               head.distance(from: head.startIndex, to: stop) > limit / 2 {
                spoken = String(head[...stop]) + " There's more on your phone."
            } else {
                spoken = head + "… There's more on your phone."
            }
        }
        return spoken.isEmpty ? "There was no reply." : spoken
    }

    /// The last few turns, said as a conversation rather than a dump.
    static func transcript(_ messages: some Sequence<ChatMessage>) -> String {
        let spoken = messages.map { message -> String in
            let who = message.role == "user" ? "You said" : "It said"
            return "\(who): \(trim(message.content))"
        }
        return spoken.joined(separator: " … ")
    }

    /// "a, b and c" — because "a, b, c" read aloud is a shopping list.
    static func list(_ items: [String]) -> String {
        switch items.count {
        case 0: return ""
        case 1: return items[0]
        default:
            return items.dropLast().joined(separator: ", ") + " and " + items[items.count - 1]
        }
    }
}

// MARK: - Phrases

/// What Siri listens for.
///
/// `${applicationName}` has to appear in every phrase — Apple requires
/// it, and it is also what stops "read me the chat" from being a phrase
/// every app on the phone competes for. The alternatives per intent are
/// the ways the same sentence is actually said, not synonyms for their
/// own sake: somebody says "ask HyperLink", "ask HyperLink about" and
/// "HyperLink, what is" and means the same thing all three times.
struct HyperLinkShortcuts: AppShortcutsProvider {
    static var appShortcuts: [AppShortcut] {
        AppShortcut(
            intent: AskHyperLinkIntent(),
            phrases: [
                "Ask \(.applicationName) \(\.$question)",
                "Ask \(.applicationName) about \(\.$question)",
                "\(.applicationName), \(\.$question)",
                "Ask my PC on \(.applicationName) \(\.$question)",
            ],
            shortTitle: "Ask",
            systemImageName: "bubble.left.and.text.bubble.right"
        )
        AppShortcut(
            intent: LoadModelIntent(),
            phrases: [
                "Load the model \(\.$model) on \(.applicationName)",
                "Load \(\.$model) in \(.applicationName)",
                "Switch \(.applicationName) to \(\.$model)",
            ],
            shortTitle: "Load a model",
            systemImageName: "shippingbox"
        )
        AppShortcut(
            intent: ReadChatIntent(),
            phrases: [
                "Read me the \(.applicationName) chat",
                "Read me my \(.applicationName) messages",
                "Read the latest \(.applicationName) message",
                "What did \(.applicationName) say",
            ],
            shortTitle: "Read a chat",
            systemImageName: "speaker.wave.2"
        )
        AppShortcut(
            intent: SendMessageIntent(),
            phrases: [
                "Send a message in \(.applicationName)",
                "Tell \(.applicationName) \(\.$message)",
            ],
            shortTitle: "Send a message",
            systemImageName: "paperplane"
        )
    }
}
