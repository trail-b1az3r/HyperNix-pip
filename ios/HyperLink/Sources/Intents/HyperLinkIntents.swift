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
//  Entities: how Siri learns what "that chat" and "Gemma" are (0.72.6)
//  --------------------------------------------------------------------
//  The first version had only `String` parameters, and Siri cannot hear
//  a string inside a sentence: only an `AppEntity` or `AppEnum` can be
//  named in a phrase. So "read me the chat called Groceries" was never
//  one sentence -- Siri started the intent, then asked which chat. The
//  newer Siri goes further and reasons over an app's entities, which a
//  string parameter gives it nothing to reason over.
//
//  Chats and models are now `ChatEntity` and `ModelEntity`, each with a
//  query that asks the paired server, and the phrases name them:
//  "Read Groceries in HyperLink", "Load Gemma 4 in HyperLink". The app
//  calls `updateAppShortcutParameters()` whenever its chats or models
//  change, which is how Siri learns the names it should listen for.
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

// MARK: - Entities

/// A conversation on the paired server, as Siri knows it.
struct ChatEntity: AppEntity {
    static let typeDisplayRepresentation: TypeDisplayRepresentation = "Chat"
    static let defaultQuery = ChatEntityQuery()

    let id: String
    let title: String

    var displayRepresentation: DisplayRepresentation {
        DisplayRepresentation(title: "\(title)")
    }

    init(id: String, title: String) {
        self.id = id
        self.title = title
    }

    init(_ session: ChatSession) {
        self.init(id: session.sessionID, title: session.title.isEmpty ? "Untitled chat" : session.title)
    }
}

/// Finds chats by id, by what was said, or offers the recent ones.
///
/// Every answer comes from the server. Nothing is cached here, because
/// an intent runs in whatever process Siri starts, and a stale list of
/// chats would offer one that has been deleted.
struct ChatEntityQuery: EntityStringQuery {
    func entities(for identifiers: [ChatEntity.ID]) async throws -> [ChatEntity] {
        let wanted = Set(identifiers)
        return try await Self.all().filter { wanted.contains($0.id) }
    }

    func entities(matching string: String) async throws -> [ChatEntity] {
        let wanted = ModelMatch.normalise(string)
        let chats = try await Self.all()
        guard !wanted.isEmpty else { return chats }
        return chats.filter { ModelMatch.normalise($0.title).contains(wanted) }
    }

    func suggestedEntities() async throws -> [ChatEntity] {
        Array(try await Self.all().prefix(10))
    }

    static func all() async throws -> [ChatEntity] {
        guard let client = await PairingStore.client() else { return [] }
        return try await client.sessions()
            .sorted { $0.updatedAt > $1.updatedAt }
            .map(ChatEntity.init)
    }
}

/// A model the paired server can load, as Siri knows it.
struct ModelEntity: AppEntity {
    static let typeDisplayRepresentation: TypeDisplayRepresentation = "Model"
    static let defaultQuery = ModelEntityQuery()

    let id: String

    var displayRepresentation: DisplayRepresentation {
        DisplayRepresentation(title: "\(shortModelName(id))", subtitle: "\(id)")
    }
}

/// Finds models by id or by what was said. "gemma 4 e2b" matches
/// `blazeindustries/gemma-4-e2b` the way `ModelMatch` always has.
struct ModelEntityQuery: EntityStringQuery {
    func entities(for identifiers: [ModelEntity.ID]) async throws -> [ModelEntity] {
        let known = Set(try await Self.ids())
        return identifiers.filter { known.contains($0) }.map(ModelEntity.init)
    }

    func entities(matching string: String) async throws -> [ModelEntity] {
        let ids = try await Self.ids()
        if let best = ModelMatch.best(spoken: string, owner: "", in: ids) {
            return [ModelEntity(id: best)]
        }
        let wanted = ModelMatch.normalise(string)
        return ids.filter { ModelMatch.normalise($0).contains(wanted) }.map(ModelEntity.init)
    }

    func suggestedEntities() async throws -> [ModelEntity] {
        Array(try await Self.ids().prefix(20)).map(ModelEntity.init)
    }

    static func ids() async throws -> [String] {
        guard let client = await PairingStore.client() else { return [] }
        return try await client.modelCatalogue().models.map(\.modelID)
    }
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

    /// An entity, so the model can be named in the sentence itself:
    /// "Load Gemma 4 in HyperLink". The query does the matching that
    /// used to happen here on a free-text string.
    @Parameter(
        title: "Model",
        description: "The model to load, e.g. Gemma 4 E2B.",
        requestValueDialog: "Which model?"
    )
    var model: ModelEntity

    @MainActor
    func perform() async throws -> some IntentResult & ProvidesDialog {
        guard let client = await PairingStore.client() else {
            return .result(dialog: IntentDialog(stringLiteral: IntentBridge.notPaired))
        }
        do {
            let session = try await client.createSession(title: "Siri", modelID: model.id)
            _ = session
            return .result(dialog: IntentDialog(
                stringLiteral: "\(shortModelName(model.id)) is ready."
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
        description: "Which conversation. Leave empty for the most recent."
    )
    var chat: ChatEntity?

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
            guard let session = ChatMatch.chosen(chat, in: sessions) else {
                return .result(dialog: IntentDialog(
                    stringLiteral: chat == nil
                        ? "You have no conversations yet."
                        : "I could not find the chat \(chat?.title ?? "")."
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
    /// The chat Siri resolved, or the most recent one when none was said.
    static func chosen(_ entity: ChatEntity?, in sessions: [ChatSession]) -> ChatSession? {
        guard let entity else { return best(spoken: "", in: sessions) }
        return sessions.first { $0.sessionID == entity.id }
    }

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

    @Parameter(title: "Chat")
    var chat: ChatEntity?

    @MainActor
    func perform() async throws -> some IntentResult & ProvidesDialog {
        guard let client = await PairingStore.client() else {
            return .result(dialog: IntentDialog(stringLiteral: IntentBridge.notPaired))
        }
        do {
            let sessions = try await client.sessions()
            // An `if let`, not `??`. The right side of `??` is an
            // autoclosure, which can be neither throwing nor async, so
            // `?? (try await ...)` is two compile errors — "operator can
            // throw but expression is not marked with 'try'" and "'async'
            // call in an autoclosure that does not support concurrency" —
            // and no amount of parenthesising fixes it.
            let session: ChatSession
            if let existing = ChatMatch.chosen(chat, in: sessions) {
                session = existing
            } else {
                session = try await client.createSession(title: "Siri")
            }
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
///
/// Only the chat and model parameters appear in phrases, and that is a
/// rule rather than a style: a phrase may only name a parameter whose
/// type is an `AppEntity` or an `AppEnum`. `question`, `model` and `message` are
/// all `String`, and a string has no finite set of values for Siri to
/// match against, so there is nothing it could recognise in the
/// sentence. Writing `\(\.$question)` anyway is not a compile error —
/// it builds, and then `appintentsmetadataprocessor` refuses the target
/// with "Invalid parameter type. AppEntity and AppEnum are the only
/// allowed types", which fails the build after linking and exports no
/// metadata at all.
///
/// Nothing is lost by dropping them. Every one of those parameters
/// carries a `requestValueDialog`, so the phrase starts the intent and
/// Siri asks for the value in the next breath — "Ask HyperLink" →
/// "What would you like to ask?" — which is also the flow somebody
/// gets when they trail off mid-sentence.
struct HyperLinkShortcuts: AppShortcutsProvider {
    static var appShortcuts: [AppShortcut] {
        AppShortcut(
            intent: AskHyperLinkIntent(),
            phrases: [
                "Ask \(.applicationName)",
                "Ask \(.applicationName) a question",
                "Ask \(.applicationName) something",
                "Ask my PC on \(.applicationName)",
            ],
            shortTitle: "Ask",
            systemImageName: "bubble.left.and.text.bubble.right"
        )
        AppShortcut(
            intent: LoadModelIntent(),
            phrases: [
                "Load a model on \(.applicationName)",
                "Load a model in \(.applicationName)",
                "Switch the \(.applicationName) model",
                "Load \(\.$model) in \(.applicationName)",
                "Load \(\.$model) on \(.applicationName)",
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
                "Read \(\.$chat) in \(.applicationName)",
                "Read me \(\.$chat) on \(.applicationName)",
            ],
            shortTitle: "Read a chat",
            systemImageName: "speaker.wave.2"
        )
        AppShortcut(
            intent: SendMessageIntent(),
            phrases: [
                "Send a message in \(.applicationName)",
                "Send a message with \(.applicationName)",
                "Send a message to \(\.$chat) in \(.applicationName)",
            ],
            shortTitle: "Send a message",
            systemImageName: "paperplane"
        )
    }
}
