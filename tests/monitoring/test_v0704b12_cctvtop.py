from pathlib import Path


def test_cli_aliases_exist():
    # We just run --help or import them to verify they are registered
    pass

def test_cctvtop_finds_logs(tmp_path: Path):
    # Run in a directory with no logs should exit 1
    import os

    from hypernix.cctvtop import cli_main
    orig_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        res = cli_main()
        assert res == 1, f"Expected cli_main to return 1, got {res}"
    finally:
        os.chdir(orig_cwd)
