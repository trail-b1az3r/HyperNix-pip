//  HyperLinkApp.swift
//  HyperLink — HyperNix from a phone.
//
//  Chat with models running on your own PC, on the home network or
//  anywhere via Tailscale. See ios/README.md for the build, and
//  wiki/HyperLink.md for how the pairing and the bridge work.

import SwiftUI

@main
struct HyperLinkApp: App {
    @State private var state = AppState()
    @State private var themes = ThemeStore()
    @State private var background = BackgroundSession()
    @Environment(\.scenePhase) private var scenePhase

    init() {
        // Before anything can read one. An admin credential is not meant
        // to survive a restart unless someone asked for it to, and
        // "unless someone asked" has to be decided at launch rather than
        // at first use — by first use the app is already holding it.
        AdminCredentialStore.beginSession()
    }

    var body: some Scene {
        WindowGroup {
            RootView()
                .environment(state)
                .environment(themes)
                // Replaces `.tint(.accentColor)`, which was the whole of
                // HyperLink's colour: one accent, applied everywhere,
                // saying nothing about which bubble is whose.
                .hyperLinkTheme(themes.theme)
        }
        .onChange(of: scenePhase) { _, phase in
            switch phase {
            case .background:
                // ~30 seconds, which is what iOS grants — see
                // BackgroundSession. Long enough to finish the frame we
                // are on and to note which conversation was in flight;
                // nowhere near long enough to hold a socket, which is
                // why the server persists the reply instead.
                guard state.isPaired else { return }
                background.begin(sessionID: state.openSessionID) {
                    // The grant ran out. Nothing to do but let go
                    // cleanly — the answer is being written on the
                    // server and will be there on the way back.
                }
            case .active:
                background.finish()
                guard state.isPaired else { return }
                // Coming back is the moment the phone is most likely to
                // be on a different network than when it went away —
                // refreshing here re-runs endpoint failover before the
                // user taps anything.
                let resuming = background.resumeTarget()
                Task {
                    await state.refreshAll()
                    if let resuming {
                        // The reply carried on without us. Reloading the
                        // thread is what makes a conversation picked up
                        // three hours later indistinguishable from one
                        // that never stopped.
                        await state.reload(sessionID: resuming)
                    }
                }
            default:
                break
            }
        }
    }
}
