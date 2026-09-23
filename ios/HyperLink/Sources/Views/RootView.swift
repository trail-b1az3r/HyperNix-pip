//  RootView.swift
//  Pairing screen, or the app. Nothing in between.

import SwiftUI

struct RootView: View {
    @Environment(AppState.self) private var state

    var body: some View {
        Group {
            if state.isPaired {
                MainTabView()
            } else {
                PairingView()
            }
        }
        // Animating the swap makes signing out and pairing read as one
        // app changing state rather than two screens fighting.
        .animation(Motion.deliberate, value: state.isPaired)
    }
}

struct MainTabView: View {
    @Environment(AppState.self) private var state

    var body: some View {
        TabView {
            // ChatListView brings its own NavigationStack — see the
            // comment on its `path`.
            ChatListView()
                .tabItem { Label("Chats", systemImage: "bubble.left.and.bubble.right") }

            NavigationStack {
                ModelsView()
            }
            .tabItem { Label("Models", systemImage: "cpu") }

            NavigationStack {
                SettingsView()
            }
            .tabItem { Label("Server", systemImage: "desktopcomputer") }

            // Models on the phone itself. A tab of its own rather than a
            // row under "Models", which lists what the *PC* can serve —
            // mixing the two would make it unclear which device a model
            // is about to be downloaded to.
            NavigationStack {
                OnDeviceModelsView()
            }
            .tabItem { Label("On iPhone", systemImage: "iphone.gen3") }

            // Separate from "Server" on purpose: that tab is about the
            // machine, this one is about the person. Burying a bio and
            // a system prompt under a screen titled "Server" is how
            // nobody finds them.
            NavigationStack {
                MySettingsView()
            }
            .tabItem { Label("You", systemImage: "person.crop.circle") }
        }
        .task { await state.refreshAll() }
    }
}
