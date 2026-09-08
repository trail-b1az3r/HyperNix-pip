import Foundation

/// Why a server address cannot work, decided before a request is sent.
///
/// Three of the addresses a person will naturally try are unreachable by
/// construction, and iOS reports all three in words that describe
/// Apple's policy rather than the situation:
///
/// - `127.0.0.1:8000` — on a phone that is *the phone*. The request goes
///   out, finds nothing, and comes back "Could not connect to the
///   server", which reads like the PC is down.
/// - `100.109.195.71:8000` — a Tailscale address, and a bare IP literal.
///   App Transport Security exceptions are matched against **domains**;
///   they do not apply to IP addresses at all. So no entry in
///   Info.plist can permit this, and the only thing that would is
///   `NSAllowsArbitraryLoads`, which would permit plain HTTP to
///   everywhere on the internet as well.
/// - any other public host over `http://` — refused by ATS for the same
///   policy reason, correctly.
///
/// The failure text for the last two is *"The resource could not be
/// loaded because the App Transport Security policy requires the use of
/// a secure connection."* It is accurate, it is about the phone rather
/// than the server, and it does not mention the one thing that fixes it:
/// the MagicDNS name the server prints right next to the address that
/// does not work.
enum AddressAdvice: Equatable {
    /// Send it. Nothing here will stop the request.
    case ok
    /// Do not send it — it cannot succeed, for the stated reason.
    case refuse(reason: String, fix: String)

    var isRefusal: Bool {
        if case .refuse = self { return true }
        return false
    }

    /// The message to show, or nil when there is nothing to say.
    var message: String? {
        if case let .refuse(reason, fix) = self { return reason + "\n\n" + fix }
        return nil
    }
}

enum AddressCheck {
    /// Hosts that name the device the app is running on.
    private static let loopbackNames: Set<String> = ["localhost", "::1", "[::1]"]

    /// Judge *address* the way it will be spelled on the wire.
    ///
    /// Takes the already-normalized form so that this and the request
    /// agree about the scheme and port — checking the raw text and
    /// sending the normalized one is how a validator ends up passing an
    /// address the request then fails on.
    static func advice(for address: String) -> AddressAdvice {
        let normalized = HyperLinkClient.normalize(address)
        guard let url = URL(string: normalized), let host = url.host else { return .ok }
        let scheme = (url.scheme ?? "http").lowercased()
        let bare = host.lowercased().trimmingCharacters(in: CharacterSet(charactersIn: "[]"))

        if isLoopback(bare) {
            return .refuse(
                reason: "\(host) is this iPhone, not your computer. Nothing on the "
                    + "phone is listening there, so the connection fails no matter "
                    + "what the PC is doing.",
                fix: "On the computer running HyperNix, run `waiter hyperlink pair`. "
                    + "It prints an address this phone can actually reach — a "
                    + "192.168.x.y one on your home network, or a Tailscale name "
                    + "ending in .ts.net that also works away from home."
            )
        }

        // https is never blocked, whatever the host is.
        guard scheme == "http" else { return .ok }

        if isIPv4Literal(bare) || bare.contains(":") {
            if isPrivateIPv4(bare) { return .ok }  // NSAllowsLocalNetworking
            if isTailscaleIPv4(bare) {
                return .refuse(
                    reason: "\(host) is a Tailscale address, and iOS will not make a "
                        + "plain-HTTP request to a bare IP address. App Transport "
                        + "Security exceptions are matched against domain names, so "
                        + "no setting in this app can allow an IP — only allowing "
                        + "insecure HTTP to the entire internet would, which it does "
                        + "not do.",
                    fix: "Use the MagicDNS name for the same machine instead — it ends "
                        + "in .ts.net and reaches exactly this address. "
                        + "`waiter hyperlink pair` prints it above the numeric one, "
                        + "and `tailscale status` on the PC shows it too."
                )
            }
            return .refuse(
                reason: "iOS will not make a plain-HTTP request to the public address "
                    + "\(host). App Transport Security allows insecure HTTP only on "
                    + "your local network and to .ts.net names.",
                fix: "Use https://, or reach the machine over Tailscale by its "
                    + ".ts.net name."
            )
        }

        // A hostname. Local and tailnet names are exempted in Info.plist.
        if bare.hasSuffix(".local") || bare == "local" { return .ok }
        if bare == "ts.net" || bare.hasSuffix(".ts.net") { return .ok }

        return .refuse(
            reason: "iOS will not make a plain-HTTP request to \(host). App Transport "
                + "Security allows insecure HTTP only on your local network, to "
                + ".local names, and to .ts.net names.",
            fix: "Use https://, or reach the machine over Tailscale by its .ts.net "
                + "name — `waiter hyperlink pair` prints it."
        )
    }

    // MARK: - Address shapes

    static func isLoopback(_ host: String) -> Bool {
        if loopbackNames.contains(host) { return true }
        guard let octets = ipv4Octets(host) else { return false }
        return octets[0] == 127
    }

    /// RFC 1918 plus link-local — what `NSAllowsLocalNetworking` exempts.
    static func isPrivateIPv4(_ host: String) -> Bool {
        guard let o = ipv4Octets(host) else { return false }
        if o[0] == 10 { return true }
        if o[0] == 172 && (16...31).contains(o[1]) { return true }
        if o[0] == 192 && o[1] == 168 { return true }
        if o[0] == 169 && o[1] == 254 { return true }
        return false
    }

    /// 100.64.0.0/10 — the shared address space Tailscale assigns.
    /// Deliberately *not* folded into `isPrivateIPv4`: it is not RFC 1918
    /// and `NSAllowsLocalNetworking` does not cover it, which is the
    /// whole reason a tailnet address behaves differently from a LAN one.
    static func isTailscaleIPv4(_ host: String) -> Bool {
        guard let o = ipv4Octets(host) else { return false }
        return o[0] == 100 && (64...127).contains(o[1])
    }

    static func isIPv4Literal(_ host: String) -> Bool { ipv4Octets(host) != nil }

    private static func ipv4Octets(_ host: String) -> [Int]? {
        let parts = host.split(separator: ".", omittingEmptySubsequences: false)
        guard parts.count == 4 else { return nil }
        var octets: [Int] = []
        for part in parts {
            guard let value = Int(part), (0...255).contains(value), !part.isEmpty else {
                return nil
            }
            octets.append(value)
        }
        return octets
    }
}

/// Turn a failure that already happened into something actionable.
///
/// The pre-flight check above catches what it can see. This catches what
/// only the network can tell us, and exists for the same reason: the
/// system's own wording describes a policy or an errno, not a situation
/// anyone can act on.
enum FailureAdvice {
    static func explain(_ error: Error, address: String) -> String {
        let text = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
        let host = URL(string: HyperLinkClient.normalize(address))?.host ?? address

        if text.contains("App Transport Security") {
            // Reached here rather than pre-flight, so the address looked
            // permissible and was not. Say what ATS actually allows.
            return "iOS blocked this request before it left the phone: plain HTTP is "
                + "allowed only on your local network, to .local names, and to "
                + ".ts.net names.\n\nUse the machine's .ts.net name, or put the "
                + "server behind https://.\n\n(\(text))"
        }

        let nsError = error as NSError
        let refused = nsError.code == NSURLErrorCannotConnectToHost
            || text.localizedCaseInsensitiveContains("could not connect")
            || text.localizedCaseInsensitiveContains("connection refused")
        if refused {
            return "Nothing answered at \(host).\n\nCheck the server is running — "
                + "`hypernix-t1 status` on the PC — and that the address and port "
                + "match what `waiter hyperlink pair` printed."
        }

        if nsError.code == NSURLErrorTimedOut {
            return "\(host) did not answer in time.\n\nIf that is a Tailscale name, "
                + "check both machines are on the tailnet (`tailscale status`). On a "
                + "home network, check the phone and the PC are on the same Wi-Fi."
        }

        if nsError.code == NSURLErrorCannotFindHost {
            return "\(host) could not be resolved.\n\nFor a Tailscale name, MagicDNS "
                + "has to be on. For a .local name, both machines need to be on the "
                + "same network."
        }

        return text
    }
}
