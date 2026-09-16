//  BackgroundSession.swift
//  Staying connected after the phone is put down.
//
//  The complaint was that a conversation dies when the app goes to the
//  background — you ask a 70B something, lock the phone, come back two
//  minutes later, and the answer is gone. What had actually happened is
//  ordinary iOS behaviour: a suspended app's URLSession tasks are
//  cancelled, and the reply was being generated into a socket the
//  system had already torn down.
//
//  What iOS actually gives you
//  ---------------------------
//  This is the part worth being honest about, because the request asked
//  for **2.5 hours or more** and iOS does not sell that.
//
//  `beginBackgroundTask` buys about **30 seconds** on modern iOS — it
//  used to be three minutes, and is not any more. Nothing an ordinary
//  app can do keeps a socket open for two and a half hours; an app that
//  claims otherwise is an app that gets terminated and does not notice.
//
//  So the promise is kept a different way, and it is a better way: the
//  work does not live on the phone. The *server* is generating, and it
//  persists the reply as it goes — including when the client
//  disconnects half way, which
//  :func:`hypernix.t1api.routers.hyperlink.chat_turn_stream` already
//  does. So the phone does not need to stay connected at all. It needs
//  to:
//
//  1. use its ~30 background seconds to finish the frame it is on and
//     let the server know it is going (so a Stop is a *choice*, not a
//     side effect of locking the phone),
//  2. remember which conversation was in flight, and
//  3. reconcile on the way back — however long that is.
//
//  A conversation resumed after three hours is then indistinguishable
//  from one that never stopped, which is what was actually being asked
//  for. `MAXIMUM_ABSENCE` is the window over which that is *promised*;
//  past it the reply is still on the server, it is simply no longer
//  treated as "in flight".

import Foundation
import UIKit

/// Remembers what was happening when the app went away.
@MainActor
final class BackgroundSession {
    /// How long a backgrounded conversation is still treated as live.
    ///
    /// 2.5 hours, as asked for — and deliverable precisely *because*
    /// nothing is being held open for it. The server has the reply; this
    /// is how long the phone keeps caring about which conversation it
    /// belonged to.
    static let maximumAbsence: TimeInterval = 2.5 * 60 * 60

    /// What iOS actually grants for finishing work on the way out.
    /// Documented here so nobody later reads `maximumAbsence` and
    /// concludes the socket is kept open for it.
    static let systemGrant: TimeInterval = 30

    private var task: UIBackgroundTaskIdentifier = .invalid
    private(set) var leftAt: Date?
    private(set) var pendingSessionID: String?

    /// Called as the app goes to the background.
    ///
    /// *sessionID* is the conversation with an answer in flight, if
    /// there is one. `onExpiry` runs if iOS ends the grant before
    /// `finish()` — it must be fast and must not assume it will be
    /// followed by anything.
    func begin(sessionID: String?, onExpiry: @escaping () -> Void) {
        leftAt = Date()
        pendingSessionID = sessionID
        guard task == .invalid else { return }
        task = UIApplication.shared.beginBackgroundTask(
            withName: "hyperlink.finish-turn"
        ) { [weak self] in
            // iOS is about to suspend us whatever we do. Ending the task
            // here rather than being killed for holding it is the
            // difference between a clean stop and a crash report.
            onExpiry()
            self?.finish()
        }
    }

    /// Release the grant. Safe to call more than once.
    func finish() {
        guard task != .invalid else { return }
        UIApplication.shared.endBackgroundTask(task)
        task = .invalid
    }

    /// How long the app was away, or nil if it never left.
    var absence: TimeInterval? {
        guard let leftAt else { return nil }
        return Date().timeIntervalSince(leftAt)
    }

    /// Whether a conversation that was in flight should be reconciled
    /// rather than forgotten.
    ///
    /// True for any absence inside the window — including a very short
    /// one, which is the common case of glancing at a notification.
    var shouldResume: Bool {
        guard pendingSessionID != nil, let absence else { return false }
        return absence <= Self.maximumAbsence
    }

    /// Called on the way back in. Returns the conversation to reload,
    /// and forgets it either way — a session that is resumed twice would
    /// reload twice.
    func resumeTarget() -> String? {
        defer {
            pendingSessionID = nil
            leftAt = nil
        }
        return shouldResume ? pendingSessionID : nil
    }
}
