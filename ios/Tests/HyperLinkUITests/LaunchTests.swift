//  LaunchTests.swift
//  Does the app actually start?
//
//  This target exists because of a shipped crash-on-launch that every
//  other test passed straight through.
//
//  `HyperLinkTests` is a `bundle.unit-test` with no host application.
//  That is a reasonable choice for what it tests — parsing, saved
//  servers, markdown, address advice, all of it pure — but it means the
//  bundle is loaded into a bare test runner and the *app* never starts.
//  `HyperLinkApp.init()` does not run. `AppState()` is never
//  constructed. No scene is ever created, so nothing reads the scene
//  manifest, the entitlements, or the Info.plist. A hundred and nine
//  green tests said nothing whatsoever about whether the app opens.
//
//  A UI test is the opposite: XCUITest installs the real bundle and
//  launches it the way the home screen does. If the app crashes on
//  launch, `launch()` fails here, in CI, on a simulator — instead of on
//  somebody's phone after a release.
//
//  This is deliberately not a test of what is on screen. Asserting
//  particular views turns every layout change into a failure here and
//  teaches people to ignore it. The question is only: did it start, and
//  is it still running a moment later.

import XCTest

final class LaunchTests: XCTestCase {

    /// The app starts and stays up.
    func testTheAppLaunches() {
        let app = XCUIApplication()
        app.launch()
        XCTAssertEqual(
            app.state, .runningForeground,
            "the app did not reach the foreground — it crashed during launch"
        )
    }

    /// Still alive a moment after launch.
    ///
    /// Separate from the test above because the two failures have
    /// different causes. Dying *in* launch is usually the Info.plist,
    /// the entitlements or a scene the system could not build. Dying
    /// just after is usually the first thing the app does on its own —
    /// restoring a pairing, reading the keychain, a task kicked off from
    /// an initialiser.
    func testTheAppIsStillRunningShortlyAfterLaunch() {
        let app = XCUIApplication()
        app.launch()
        // Long enough for the work `AppState.restore()` starts to have
        // got somewhere, short enough not to pad every CI run.
        Thread.sleep(forTimeInterval: 3)
        XCTAssertEqual(
            app.state, .runningForeground,
            "the app launched and then died — something it started for "
            + "itself brought it down"
        )
    }

    /// Backgrounding and coming back, which is its own launch path.
    ///
    /// `HyperLinkApp` does real work on `scenePhase` changes — it takes
    /// a background grant on the way out and calls `refreshAll()` on the
    /// way in. Neither had ever run anywhere before this target existed.
    func testItSurvivesBeingBackgroundedAndResumed() {
        let app = XCUIApplication()
        app.launch()
        XCUIDevice.shared.press(.home)
        Thread.sleep(forTimeInterval: 2)
        app.activate()
        Thread.sleep(forTimeInterval: 2)
        XCTAssertEqual(
            app.state, .runningForeground,
            "the app did not come back from the background"
        )
    }
}
