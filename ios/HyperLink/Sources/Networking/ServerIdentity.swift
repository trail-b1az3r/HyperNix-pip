//  ServerIdentity.swift
//  Is this the machine I paired with?
//
//  The app reaches its server at whichever of several addresses answers
//  first, and those addresses are not stable: a LAN IP is handed out by
//  DHCP and reassigned, a tailnet name can be transferred, and the app
//  will happily try an address that some *other* machine now answers on.
//
//  The obvious check is the server name, and the obvious check is wrong.
//  A name is chosen by whoever set the machine up and advertised in the
//  clear; on a home LAN or a tailnet anything can call itself `desktop`.
//  Authenticating on a name means the first machine to claim it wins.
//
//  So the server reports a fingerprint — a hash of a random seed it
//  generated once and keeps to itself (`hypernix.hyperlink.identity` on
//  the server side). The app pins it at pairing time and compares it on
//  every reconnection. That is not proof of identity on its own; anyone
//  who can read the fingerprint can repeat it, exactly as with a TLS
//  certificate fingerprint. What it gives is the ability to *notice*
//  that the address you reached is answering for a different machine
//  than last time, which a name comparison cannot do at all.
//
//  Ordering matters as much as the check. Verifying identity is what
//  gates handing over the *admin* credential — never the other way
//  around. The device token goes first because it is scoped to one
//  server and worthless anywhere else; the admin credential is only
//  presented to a connection whose fingerprint has already matched.

import Foundation

/// What a fingerprint comparison concluded.
enum IdentityVerdict: Equatable, Sendable {
    /// It matches what was pinned. The ordinary case.
    case matches

    /// Nothing was pinned. True on the first connection after pairing,
    /// and for a connection made before the server reported one at all —
    /// which is why it is a distinct case rather than a failure: an
    /// older server that does not send a fingerprint must keep working.
    case unknown

    /// A fingerprint was pinned and this is a different one. Something
    /// else is answering at this address.
    case mismatch(expected: String, actual: String)

    var isSafeForAdminCredentials: Bool {
        if case .matches = self { return true }
        return false
    }

    var message: String? {
        switch self {
        case .matches, .unknown:
            return nil
        case .mismatch:
            return "This address is answering for a different machine than "
                + "the one you paired with. That can happen when a home "
                + "router hands your PC's old address to something else — "
                + "or when something is pretending to be your PC. HyperLink "
                + "has not sent your credentials."
        }
    }
}

/// Comparing a reported fingerprint against the pinned one.
///
/// Deliberately holds nothing. The pinned value lives on
/// `ServerConnection`, which the app already persists, so there is one
/// record of what this phone is paired with rather than two that can
/// disagree — and this type stays pure, which is why it can be tested
/// without a device, a keychain, or a server.
enum ServerIdentity {
    static func check(reported: String, pinned: String) -> IdentityVerdict {
        // An empty *reported* fingerprint is a server too old to send
        // one, not a mismatch. Treating it as a mismatch would break
        // every existing pairing on upgrade, which is how a security
        // check comes to be switched off in the release after it lands.
        guard !reported.isEmpty else { return .unknown }
        // An empty *pinned* one is a pairing made before this existed.
        // The caller pins what it sees — trust on first use — rather
        // than refusing to connect to a server it has always used.
        guard !pinned.isEmpty else { return .unknown }
        return reported == pinned
            ? .matches
            : .mismatch(expected: pinned, actual: reported)
    }

    /// The fingerprint to store, given what is pinned and what the
    /// server just reported.
    ///
    /// Never overwrites a pinned value with a different one: that would
    /// turn every mismatch into a silent re-pin, which is precisely the
    /// event this is here to catch. It only fills in a blank.
    static func pinning(current pinned: String, reported: String) -> String {
        if !pinned.isEmpty { return pinned }
        return reported
    }

    /// A fingerprint as a human can compare it against what the server
    /// printed: four groups of eight, short enough to read aloud and
    /// long enough that skimming still catches a change.
    static func display(_ fingerprint: String) -> String {
        stride(from: 0, to: fingerprint.count, by: 8)
            .map { offset -> String in
                let start = fingerprint.index(fingerprint.startIndex, offsetBy: offset)
                let end = fingerprint.index(
                    start, offsetBy: 8, limitedBy: fingerprint.endIndex
                ) ?? fingerprint.endIndex
                return String(fingerprint[start..<end])
            }
            .joined(separator: " ")
    }
}
