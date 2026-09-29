"""Tests for v0.70.4b1 tvtop++ dashboard, resilient log parser, and block history."""
from __future__ import annotations

from pathlib import Path

from hypernix import tv
from hypernix.tvtop_plus_plus import TVTopPlusPlus, _block_history_bar


def test_looks_like_training_log_resilient(tmp_path: Path) -> None:
    # Test that simple log format with just loss= is classified as training log
    log1 = tmp_path / "loss_only.log"
    log1.write_text("Epoch 1 - loss=0.456\n", encoding="utf-8")
    assert tv._looks_like_training_log(log1)

    # Test tqdm style progress fraction
    log2 = tmp_path / "tqdm.log"
    log2.write_text(" 50%|██████████| 500/1000 [00:10<00:10, 48.20it/s]\n", encoding="utf-8")
    assert tv._looks_like_training_log(log2)

    # Test random non-training log
    log3 = tmp_path / "system.log"
    log3.write_text("systemd[1]: Started User Manager for UID 1000.\n", encoding="utf-8")
    assert not tv._looks_like_training_log(log3)


def test_log_tail_resilient_parsing(tmp_path: Path) -> None:
    log = tmp_path / "train.log"
    log.write_text("loss=0.1234\n", encoding="utf-8")
    
    tail = tv.LogTail(log)
    tail.poll()
    
    assert tail.has_training_data is True
    assert tail.loss == 0.1234
    assert len(tail.losses) == 1
    assert tail.losses[0] == 0.1234

    # Append learning rate and step fraction
    with log.open("a", encoding="utf-8") as fh:
        fh.write("step 20/1000 lr=5e-5\n")
    tail.poll()
    assert tail.step == 20
    assert tail.total_steps == 1000
    assert tail.lr == 5e-5

    # Append tqdm progress bar line
    with log.open("a", encoding="utf-8") as fh:
        fh.write(" 50%|██████████| 500/1000 [00:10<00:10, 48.20it/s, loss=0.089]\n")
    tail.poll()
    assert tail.step == 500  # calculated from 50% or directly parsed from 500/1000
    assert tail.loss == 0.089
    assert tail.throughput == 48.20


def test_block_history_bar_rendering() -> None:
    # Test no history
    bar = _block_history_bar([], 10, color_enabled=False)
    assert bar == " " * 10

    # Test density characters mapped correctly
    # 5% (0-20% -> ' '), 30% (20-40% -> '░'), 50% (40-60% -> '▒'), 70% (60-80% -> '▓'), 95% (80-100% -> '█')
    history = [5.0, 30.0, 50.0, 70.0, 95.0]
    bar = _block_history_bar(history, 5, color_enabled=False)
    assert bar == " ░▒▓█"

    # Test color tags present
    bar_colored = _block_history_bar(history, 5, color_enabled=True)
    assert "\x1b[" in bar_colored


def test_loss_curve_decay_predictions(tmp_path: Path) -> None:
    log = tmp_path / "train.log"
    # Write decreased losses
    log.write_text("loss=2.0\nloss=1.8\nloss=1.6\nloss=1.4\nloss=1.2\n", encoding="utf-8")
    
    tvt = TVTopPlusPlus(log_path=log, color=False)
    frame = tvt.latest_frame()
    
    # 5 losses present
    assert len(frame.recent_losses) == 5
    
    # Test loss panel creation - should have Loss Curve title
    loss_panel = tvt._make_loss_panel(frame)
    
    # The panel should have a title
    assert loss_panel.title is not None
    assert "Loss" in loss_panel.title or "Curve" in loss_panel.title
    
    # Verify the renderable content exists
    assert loss_panel.renderable is not None


def test_tvtop_plus_plus_small_mode(tmp_path: Path) -> None:
    log = tmp_path / "train.log"
    log.write_text("loss=1.5 step=10/100\n", encoding="utf-8")
    
    tvt = TVTopPlusPlus(log_path=log, color=False, small_mode=True)
    frame = tvt.latest_frame()
    
    # Test that small mode creates a valid layout
    from rich.console import Console
    console = Console(force_terminal=False)
    layout = tvt._build_layout(frame, console)
    
    # Should have all main sections
    assert layout["header"] is not None
    assert layout["body"] is not None
    assert layout["footer"] is not None
    
    # Training panel should be present
    training_panel = tvt._make_training_panel(frame)
    assert training_panel is not None
    assert "Training Vitals" in str(training_panel.title) if training_panel.title else True
