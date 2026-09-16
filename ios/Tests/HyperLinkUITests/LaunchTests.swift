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

    /// How long a simulator on a loaded CI runner may take to do any of
    /// this. Generous on purpose: these waits return the moment the
    /// state is reached, so a large timeout costs nothing when things
    /// are working and only buys patience when they are not.
    private let settle: TimeInterval = 30

    /// The app starts and stays up.
    func testTheAppLaunches() {
        let app = XCUIApplication()
        app.launch()
        XCTAssertTrue(
            app.wait(for: .runningForeground, timeout: settle),
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
    ///
    /// The sleep here is the test rather than a race: the point is to
    /// let the work `AppState.restore()` starts get somewhere and then
    /// look. The wait before it is what stops a slow launch being
    /// mistaken for a death.
    func testTheAppIsStillRunningShortlyAfterLaunch() {
        let app = XCUIApplication()
        app.launch()
        XCTAssertTrue(
            app.wait(for: .runningForeground, timeout: settle),
            "the app never reached the foreground"
        )
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
    ///
    /// Every step waits for a state instead of sleeping a fixed time.
    /// The first version of this test slept two seconds after
    /// `activate()` and asserted, which is a race, and CI won it: the
    /// app was found in `.runningBackground` (3) rather than
    /// `.runningForeground` (4) — alive, and simply not finished coming
    /// back. A longer sleep would have hidden that rather than fixed it,
    /// and would have cost every later run the same wait whether it
    /// needed it or not.
    func testItSurvivesBeingBackgroundedAndResumed() {
        let app = XCUIApplication()
        app.launch()
        XCTAssertTrue(
            app.wait(for: .runningForeground, timeout: settle),
            "the app never reached the foreground"
        )

        XCUIDevice.shared.press(.home)
        // Not asserted: iOS may park a backgrounded app in either
        // `.runningBackground` or `.runningBackgroundSuspended`, and
        // which one is not this test's business. Waiting for it to stop
        // being foreground is enough to know the transition happened.
        _ = app.wait(for: .runningBackground, timeout: settle)
        XCTAssertNotEqual(
            app.state, .runningForeground,
            "pressing Home did not background the app"
        )

        app.activate()
        XCTAssertTrue(
            app.wait(for: .runningForeground, timeout: settle),
            "the app did not come back from the background"
        )
    }
}
