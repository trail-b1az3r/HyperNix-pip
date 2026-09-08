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
                .tint(.accentColor)
        }
        .onChange(of: scenePhase) { _, phase in
            // Coming back from the background is the moment the phone is
            // most likely to be on a different network than when it went
            // away — refreshing here is what re-runs endpoint failover
            // before the user taps anything.
            guard phase == .active, state.isPaired else { return }
            Task { await state.refreshAll() }
        }
    }
}
