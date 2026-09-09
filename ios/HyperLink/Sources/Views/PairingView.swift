//  PairingView.swift
//  Where the PC is, and how to prove you are allowed to talk to it —
//  either the six characters it showed you, or a T2S key.

import SwiftUI
import UIKit

/// How this phone proves who it is.
///
/// Pairing is the normal route, but it needs someone at the PC: minting a
/// code is an admin operation. A T2S key is the route for when the PC is
/// not to hand — 26 typeable characters, deliberately limited to reading
/// and non-admin writing, which is why it is safe to type into a phone.
private enum ConnectMethod: String, CaseIterable, Identifiable {
    case code = "Pairing code"
    case key = "T2S key"
    /// No credential at all, to a server whose operator turned on
    /// trusted-network mode. The *server* decides whether this works —
    /// it refuses any origin it classifies as public, however it is
    /// configured — so the app can offer it and let the answer come
    /// back rather than trying to guess which network it is on.
    case trusted = "This network"

    var id: String { rawValue }
}

struct PairingView: View {
    @Environment(AppState.self) private var state

    @State private var address = ""
    @State private var code = ""
    @State private var key = ""
    @State private var method: ConnectMethod = .code
    @State private var deviceName = UIDevice.current.name
    @State private var isPairing = false

    private var codeIsPlausible: Bool {
        code.filter { $0.isLetter || $0.isNumber }.count == 6
    }

    private var keyIsPlausible: Bool {
        HyperLinkClient.looksLikeKey(key)
    }

    private var credentialIsPlausible: Bool {
        switch method {
        case .code: return codeIsPlausible
        case .key: return keyIsPlausible
        case .trusted: return true
        }
    }

    private var canConnect: Bool {
        !address.trimmingCharacters(in: .whitespaces).isEmpty
            && credentialIsPlausible
            && !deviceName.trimmingCharacters(in: .whitespaces).isEmpty
            && !isPairing
    }

    private var actionTitle: String {
        switch method {
        case .code: return isPairing ? "Pairing…" : "Pair"
        case .key, .trusted: return isPairing ? "Connecting…" : "Connect"
        }
    }

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Connect to your PC")
                            .font(.title2.weight(.semibold))
                        Text(explanation)
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                    }
                    .padding(.vertical, 4)
                }

                Section("Server address") {
                    TextField("desktop.tailnet.ts.net:8000", text: $address)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .keyboardType(.URL)
                        .submitLabel(.next)
                    Text("A Tailscale name keeps working when you leave the house. A 192.168.x.y address only works at home.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Section("How to sign in") {
                    Picker("Method", selection: $method) {
                        ForEach(ConnectMethod.allCases) { option in
                            Text(option.rawValue).tag(option)
                        }
                    }
                    .pickerStyle(.segmented)
                }

                switch method {
                case .code:
                    Section("Pairing code") {
                        TextField("ABC 123", text: $code)
                            .textInputAutocapitalization(.characters)
                            .autocorrectionDisabled()
                            // The alphabet excludes 0/O/1/I/L, so the code is
                            // unambiguous when read off a screen.
                            .font(.system(.title3, design: .monospaced))
                            .submitLabel(.done)
                        if !code.isEmpty && !codeIsPlausible {
                            Text("A pairing code is six characters.")
                                .font(.caption)
                                .foregroundStyle(.orange)
                        }
                    }
                case .key:
                    Section("T2S key") {
                        // Never autocapitalised and never autocorrected: a
                        // T2S key is case-sensitive and full of
                        // punctuation, and "helpfully" changing either
                        // turns a correct key into a wrong one.
                        TextField("T2S_…", text: $key)
                            .textInputAutocapitalization(.never)
                            .autocorrectionDisabled()
                            .font(.system(.body, design: .monospaced))
                            .submitLabel(.done)
                        if !key.isEmpty && !keyIsPlausible {
                            Text("A key starts with T2S_ (or T2_ / T1_). Check the paste kept every character.")
                                .font(.caption)
                                .foregroundStyle(.orange)
                        }
                        Text("On the PC: `gkey create -v v2short --scopes read,write`")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                case .trusted:
                    Section("No key") {
                        Text("Works only if someone turned this on at the PC, and only from your home network or your tailnet. HyperNix refuses it from anywhere else however it is set up.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Text("On the PC: `install-t1.sh --trusted-network`")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Label("A connection with no key never gets administrator access.", systemImage: "info.circle")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }

                Section("This device") {
                    TextField("iPhone", text: $deviceName)
                    Text(deviceNameHelp)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                if let error = state.connectionError {
                    Section {
                        Label(error, systemImage: "exclamationmark.triangle")
                            .foregroundStyle(.red)
                            .font(.callout)
                    }
                }

                Section {
                    Button {
                        Task {
                            isPairing = true
                            switch method {
                            case .code:
                                _ = await state.pair(
                                    address: address, code: code, deviceName: deviceName
                                )
                            case .key:
                                _ = await state.connect(
                                    address: address, key: key, deviceName: deviceName
                                )
                            case .trusted:
                                _ = await state.connectKeyless(
                                    address: address, deviceName: deviceName
                                )
                            }
                            isPairing = false
                        }
                    } label: {
                        HStack {
                            if isPairing { ProgressView().padding(.trailing, 6) }
                            Text(actionTitle)
                        }
                        .frame(maxWidth: .infinity)
                    }
                    .disabled(!canConnect)
                }
            }
            .navigationTitle("HyperLink")
        }
    }

    private var explanation: String {
        switch method {
        case .code:
            return "On the computer running HyperNix, run `waiter hyperlink pair`. "
                + "It prints an address and a six-character code."
        case .key:
            return "Paste a T2S key from the PC. It is 26 characters plus a prefix, "
                + "limited to reading and non-admin writing, and needs nobody at the "
                + "computer when you use it."
        case .trusted:
            return "Connect with no key at all. Only works if the PC was set up to "
                + "trust your home network or your tailnet, and never from outside "
                + "either of them."
        }
    }

    private var deviceNameHelp: String {
        switch method {
        case .code:
            return "Shown on the PC in `waiter hyperlink devices`, so you can tell your devices apart later."
        case .key, .trusted:
            return "Used to label this phone in the app. Neither a key nor a keyless connection is a paired device, so it will not appear in `waiter hyperlink devices`."
        }
    }
}
