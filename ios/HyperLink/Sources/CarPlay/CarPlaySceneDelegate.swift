//  CarPlaySceneDelegate.swift
//  The connection between the car and the app.
//
//  CarPlay is a second UIScene, not a second app. The system connects it
//  when the phone is plugged into (or wirelessly paired with) a car that
//  supports the app's entitlement, and disconnects it when the phone
//  leaves. Both can happen while the phone app is in the background or
//  not running at all, which is why `CarPlayController` builds its own
//  client from the stored pairing rather than reaching for `AppState`.
//
//  The scene manifest in Info.plist is what tells iOS this exists — see
//  `UIApplicationSceneManifest` in `ios/project.yml`. Without that entry
//  the delegate is never instantiated and CarPlay shows nothing, with no
//  error anywhere.
//
//  The entitlement
//  ---------------
//  `com.apple.developer.carplay-communication` is granted by Apple per
//  app, not per developer account, and a build without it will not show
//  in a car however correct this code is. That is Apple's gate rather
//  than this project's, and the README says so rather than leaving
//  somebody to discover it by plugging a cable in.

#if canImport(CarPlay)
import CarPlay
import UIKit

final class CarPlaySceneDelegate: UIResponder, CPTemplateApplicationSceneDelegate {
    private var controller: CarPlayController?

    func templateApplicationScene(
        _ scene: CPTemplateApplicationScene,
        didConnect interfaceController: CPInterfaceController
    ) {
        // Same as the phone app does at launch. An admin credential
        // should not survive into a car session unless somebody asked
        // for it to, and "unless somebody asked" has to be decided here
        // as well as in HyperLinkApp — the car can connect first.
        AdminCredentialStore.beginSession()

        interfaceController.delegate = self
        controller = CarPlayController(interfaceController: interfaceController)
    }

    func templateApplicationScene(
        _ scene: CPTemplateApplicationScene,
        didDisconnectInterfaceController interfaceController: CPInterfaceController
    ) {
        // Dropped rather than kept for next time. A controller holding a
        // stale interface controller is the shape of a crash the next
        // time the car connects, and rebuilding costs one API call.
        controller = nil
    }
}

// MARK: - Text entry

extension CarPlaySceneDelegate: CPInterfaceControllerDelegate {}

// Typing is `CPSearchTemplate`, and `CarPlayController` is its own
// delegate — see the extension there. This file used to carry a
// `CPTextInputTemplateDelegate` conformance for a template that does not
// exist in CarPlay; there is no general-purpose text-entry template, and
// the only public keyboard is the search one.
#endif
