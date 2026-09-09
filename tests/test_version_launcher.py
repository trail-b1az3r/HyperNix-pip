"""Tests for the multi-version Python launcher."""
from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

# Import the launcher module
from hypernix.version_launcher import (
    VERSION_PRIORITY,
    PythonVersion,
    check_hypernix_installed,
    find_best_python,
    main,
)


class TestPythonVersion:
    """Tests for the PythonVersion named tuple."""

    def test_version_creation(self):
        """Test creating a PythonVersion instance."""
        pv = PythonVersion(3, 12)
        assert pv.major == 3
        assert pv.minor == 12
        assert pv.version_tuple == (3, 12)

    def test_exe_name_format(self):
        """Test that exe_name returns correct format."""
        pv = PythonVersion(3, 12)
        assert pv.exe_name == "python3.12"
        
        pv = PythonVersion(3, 13)
        assert pv.exe_name == "python3.13"
        
        pv = PythonVersion(3, 14)
        assert pv.exe_name == "python3.14"

    def test_version_priority_order(self):
        """Test that VERSION_PRIORITY is in correct order."""
        assert len(VERSION_PRIORITY) == 3
        assert VERSION_PRIORITY[0].version_tuple == (3, 12)
        assert VERSION_PRIORITY[1].version_tuple == (3, 13)
        assert VERSION_PRIORITY[2].version_tuple == (3, 14)


class TestCheckHypernixInstalled:
    """Tests for check_hypernix_installed function."""

    @patch("hypernix.version_launcher.subprocess.run")
    def test_returns_true_when_installed(self, mock_run):
        """Test that function returns True when hypernix is installed."""
        mock_run.return_value = MagicMock(returncode=0, stdout="1.0.0", stderr="")
        
        result = check_hypernix_installed("python3.12")
        
        assert result is True
        mock_run.assert_called_once_with(
            ["python3.12", "-c", "import hypernix; print(hypernix.__version__)"],
            capture_output=True,
            text=True,
            timeout=5,
        )

    @patch("hypernix.version_launcher.subprocess.run")
    def test_returns_false_when_not_installed(self, mock_run):
        """Test that function returns False when hypernix is not installed."""
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="ModuleNotFoundError")
        
        result = check_hypernix_installed("python3.12")
        
        assert result is False

    @patch("hypernix.version_launcher.subprocess.run")
    def test_returns_false_on_timeout(self, mock_run):
        """Test that function returns False on timeout."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="python3.12", timeout=5)
        
        result = check_hypernix_installed("python3.12")
        
        assert result is False

    @patch("hypernix.version_launcher.subprocess.run")
    def test_returns_false_on_file_not_found(self, mock_run):
        """Test that function returns False when executable not found."""
        mock_run.side_effect = FileNotFoundError("python3.12")
        
        result = check_hypernix_installed("python3.12")
        
        assert result is False

    @patch("hypernix.version_launcher.subprocess.run")
    def test_returns_false_on_os_error(self, mock_run):
        """Test that function returns False on OS error."""
        mock_run.side_effect = OSError("Permission denied")
        
        result = check_hypernix_installed("python3.12")
        
        assert result is False


class TestFindBestPython:
    """Which interpreter runs the CLI.

    This used to try python3.12, then 3.13, then 3.14, and return the
    first one with hypernix installed -- *whichever* that was. On a
    machine with an old hypernix on 3.12 and a fresh `pip install
    --upgrade` on 3.13, `hnx` ran the old one, so every subcommand added
    since that 3.12 install was missing and the CLI answered with its
    usage table instead. Nothing said a different install was answering.

    These assert the other order. The interpreter that owns the console
    script goes first, because that is the one `pip` just upgraded.
    """

    def test_this_interpreter_wins_when_it_has_hypernix(self):
        import sys

        # hypernix is importable in the test process by construction,
        # which is exactly the situation a `pip install` leaves behind.
        assert find_best_python() == sys.executable

    @patch("hypernix.version_launcher.check_hypernix_installed")
    def test_the_version_list_is_only_a_fallback(self, mock_check):
        """It is for the case this was really meant to cover: the script
        is on PATH but its own interpreter no longer has the package."""
        mock_check.side_effect = lambda exe: exe == "python3.13"

        # None in sys.modules makes `import hypernix` raise, standing in
        # for an interpreter that does not have it.
        with patch.dict("sys.modules", {"hypernix": None}):
            result = find_best_python()

        assert result == "python3.13"

    @patch("hypernix.version_launcher.check_hypernix_installed")
    def test_the_fallback_keeps_its_priority_order(self, mock_check):
        """3.12 before 3.13 before 3.14 within the fallback. That half
        was never the problem."""
        mock_check.side_effect = lambda exe: exe in ("python3.12", "python3.14")

        with patch.dict("sys.modules", {"hypernix": None}):
            assert find_best_python() == "python3.12"

    @patch("hypernix.version_launcher.check_hypernix_installed")
    def test_the_fallback_reaches_the_last_version(self, mock_check):
        mock_check.side_effect = lambda exe: exe == "python3.14"

        with patch.dict("sys.modules", {"hypernix": None}):
            assert find_best_python() == "python3.14"

    @patch("hypernix.version_launcher.check_hypernix_installed")
    @patch("hypernix.version_launcher.sys.platform", "win32")
    def test_it_checks_the_windows_spelling_in_the_fallback(self, mock_check):
        mock_check.side_effect = lambda exe: exe == "python312"

        with patch.dict("sys.modules", {"hypernix": None}):
            assert find_best_python() == "python312"

    @patch("hypernix.version_launcher.check_hypernix_installed")
    @patch("hypernix.version_launcher.sys.executable", "/usr/bin/python3")
    def test_falls_back_to_current_python_if_none_have_hypernix(self, mock_check):
        mock_check.return_value = False

        with patch.dict("sys.modules", {"hypernix": MagicMock()}):
            assert find_best_python() == "/usr/bin/python3"


class TestMainLauncher:
    """Tests for the main launcher entry point."""

    def test_skips_version_check_when_env_var_set(self, monkeypatch):
        """Test that version check is skipped when HYPERNIX_NO_VERSION_CHECK is set."""
        # Set the environment variable
        monkeypatch.setenv("HYPERNIX_NO_VERSION_CHECK", "1")
        
        with patch("hypernix.cli.main", return_value=0) as mock_cli_main:
            result = main([])
            
            assert result == 0
            mock_cli_main.assert_called_once_with([])

    @patch("hypernix.cli.main", return_value=0)
    def test_uses_current_python_when_running_as_module(self, mock_cli_main):
        """Test that current Python is used when running as __main__.py."""
        with patch("hypernix.version_launcher.sys.argv", ["hypernix/__main__.py"]):
            result = main([])
        
        assert result == 0
        mock_cli_main.assert_called_once_with([])

    @patch("hypernix.version_launcher.run_with_selected_python", return_value=0)
    def test_uses_version_selection_for_console_scripts(self, mock_run_selected):
        """Test that version selection is used for console script entry points."""
        with patch("hypernix.version_launcher.sys.argv", ["hypernix"]):
            result = main([])
        
        assert result == 0
        mock_run_selected.assert_called_once_with([])

    @patch("hypernix.version_launcher.subprocess.run")
    @patch("hypernix.version_launcher.find_best_python")
    def test_re_invokes_only_for_a_different_interpreter(
        self, mock_find_best, mock_subprocess_run):
        """A subprocess is for reaching *another* install. Spawning one
        to reach our own would be a fork, an import and a process for
        nothing."""
        mock_find_best.return_value = "python3.12"
        mock_subprocess_run.return_value = MagicMock(returncode=0)

        from hypernix.version_launcher import run_with_selected_python

        result = run_with_selected_python(["--version"])

        assert result == 0
        assert mock_subprocess_run.call_args_list[-1].args[0] == [
            "python3.12", "-m", "hypernix", "--version"
        ]

    def test_our_own_interpreter_is_called_in_process(self, monkeypatch):
        """No subprocess at all in the common case."""
        import sys

        from hypernix import version_launcher

        monkeypatch.setattr(version_launcher, "find_best_python",
                            lambda: sys.executable)
        calls = []
        monkeypatch.setattr(version_launcher.subprocess, "run",
                            lambda *a, **k: calls.append(a) or MagicMock(returncode=0))
        monkeypatch.setattr("hypernix.interfaces.cli.main",
                            lambda argv: 7)

        assert version_launcher.run_with_selected_python(["--version"]) == 7
        assert calls == [], "should not have spawned anything"

    @patch("hypernix.version_launcher.find_best_python")
    def test_falls_back_to_current_when_selected_not_found(
        self, mock_find_best
    ):
        """Test fallback to current Python when selected executable not found."""
        mock_find_best.return_value = "python3.12"
        
        with patch("hypernix.version_launcher.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("python3.12")
            
            with patch("hypernix.cli.main", return_value=0) as mock_cli_main:
                from hypernix.version_launcher import run_with_selected_python
                
                result = run_with_selected_python(["--version"])
                
                assert result == 0
                mock_cli_main.assert_called_once_with(["--version"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
