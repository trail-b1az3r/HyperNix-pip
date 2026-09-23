//  OnDeviceHub.swift
//  The on-device pieces, held once, for the screens that use them.
//
//  Search, fit, download and the runner were all written and none of
//  them was reachable: no screen created a ModelStore or a
//  LocalInference, so a model could not be downloaded, installed or run
//  on the phone at all. This is the one place they are created, and
//  what `OnDeviceModelsView` and `LocalChatView` read.

import Combine
import SwiftUI
// `UIApplicationDelegate` is UIKit, and SwiftUI does not re-export it.
import UIKit

@MainActor
final class OnDeviceHub: ObservableObject {
    let settings: OnDeviceSettings
    let store: ModelStore
    let inference: LocalInference
    let search = HuggingFaceSearch()
    private var forwarding: Set<AnyCancellable> = []

    init(store: ModelStore = .shared) {
        let settings = OnDeviceSettings()
        self.settings = settings
        self.store = store
        self.inference = LocalInference(settings: settings)
        // A view observing the hub is not observing what the hub holds:
        // `inference.loaded` changing would not redraw a screen that
        // reads it through here, and "Load" would stay "Load" after the
        // model had loaded. Each child's change is the hub's change.
        for publisher in [inference.objectWillChange.eraseToAnyPublisher(),
                          store.objectWillChange.eraseToAnyPublisher(),
                          settings.objectWillChange.eraseToAnyPublisher()] {
            publisher
                .receive(on: RunLoop.main)
                .sink { [weak self] _ in self?.objectWillChange.send() }
                .store(in: &forwarding)
        }
        // At launch, not at first download — see ModelStore.reconnect.
        store.reconnect()
        applyToken()
    }

    /// The same Hugging Face token for searching and for downloading.
    /// Separate copies drifted: a token added in settings reached the
    /// search and not the download, and a gated model listed fine and
    /// then failed with a 401.
    func applyToken() {
        let token = settings.huggingFaceToken
        store.setToken(token)
        Task { await search.setToken(token) }
    }
}

/// The app delegate, for the one thing SwiftUI's App cannot do: take
/// iOS's completion handler when it wakes the app for a finished
/// background download.
final class HyperLinkAppDelegate: NSObject, UIApplicationDelegate {
    func application(
        _ application: UIApplication,
        handleEventsForBackgroundURLSession identifier: String,
        completionHandler: @escaping () -> Void
    ) {
        guard identifier == ModelStore.sessionIdentifier else {
            completionHandler()
            return
        }
        MainActor.assumeIsolated {
            ModelStore.shared.backgroundCompletion = completionHandler
            // Recreating the session is what gets the events delivered.
            ModelStore.shared.reconnect()
        }
    }
}
