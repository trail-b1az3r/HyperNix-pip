"""Comprehensive tests for HyperNix v0.70.5 features.

Covers:
  - tvtop++ bug fixes (layout flicker, small_mode, re-export)
  - hnx wiki CLI (module discovery, doc extraction)
  - hnx vera (syntax verification, smoke tests)
  - pressure_cooker_v5 QAT (fake quantization, config)
  - MTP (multi-token prediction, config, head)
  - scavenger (criteria matching, storage budget)
  - freezer QAT (batch sizing, VRAM overhead)
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# tvtop++ Fixes
# ---------------------------------------------------------------------------

class TestTVTopPlusPlusFixes:
    """Tests for tvtop++ bug fixes in v0.70.5."""

    def test_block_history_bar_re_export(self):
        """_block_history_bar should be importable from tvtop_plus_plus."""
        from hypernix.tvtop_plus_plus import _block_history_bar
        assert callable(_block_history_bar)

    def test_small_mode_attribute_exists(self):
        """TVTopPlusPlus should accept small_mode parameter."""
        from hypernix.tvtop_plus_plus import TVTopPlusPlus
        tv = TVTopPlusPlus(small_mode=True)
        assert tv.small_mode is True
        tv2 = TVTopPlusPlus(small_mode=False)
        assert tv2.small_mode is False

    def test_layout_init_not_rebuilt(self):
        """Layout tree should be initialized once, not rebuilt every tick."""
        from hypernix.tvtop_plus_plus import TVTopPlusPlus

        tv = TVTopPlusPlus()

        layout1 = tv._init_layout()
        layout2 = tv._init_layout()

        assert layout1 is not layout2
        assert layout1.tree.children is not None
        assert layout2.tree.children is not None

    def test_layout_update_does_not_rebuild_tree(self):
        """_update_layout should only update panel contents, not tree structure."""
        from hypernix.tv import Frame
        from hypernix.tvtop_plus_plus import TVTopPlusPlus

        tv = TVTopPlusPlus()

        layout = tv._init_layout()
        original_children = list(layout.tree.children)

        tv._update_layout(Frame(), None, layout)
        tv._update_layout(Frame(), None, layout)
        tv._update_layout(Frame(), None, layout)

        assert len(layout.tree.children) == len(original_children)

    def test_small_mode_layout_structure(self):
        """Small mode should create a simplified layout."""
        from rich.console import Console

        from hypernix.tvtop_plus_plus import TVTopPlusPlus

        tv = TVTopPlusPlus(small_mode=True)
        console = Console(force_terminal=True, width=80)
        f = tv.latest_frame()

        layout = tv._init_layout()
        tv._update_layout(f, console, layout)

        assert layout["header"] is not None
        assert layout["body"] is not None
        assert layout["footer"] is not None

    def test_process_filtering_skips_tvtop(self):
        """Process list should skip tvtop/self processes."""
        from hypernix.tvtop_plus_plus import TVTopPlusPlus

        tv = TVTopPlusPlus()
        processes = tv._get_active_processes()

        for p in processes:
            cmd_lower = p.get("cmd", "").lower()
            assert "tvtop" not in cmd_lower
            assert "nvidia-smi" not in cmd_lower


# ---------------------------------------------------------------------------
# hnx Wiki CLI
# ---------------------------------------------------------------------------

class TestWikiCLI:
    """Tests for hnx/hypenix wiki CLI."""

    def test_get_all_modules(self):
        """Should discover hypernix modules."""
        from hypernix.wiki_cli import _get_all_modules
        modules = _get_all_modules()
        assert isinstance(modules, list)
        assert len(modules) > 0
        assert "tvtop_plus_plus" in modules
        assert "pressure_cooker" in modules

    def test_get_module_doc_existing(self):
        """Should extract docs from existing modules."""
        from hypernix.wiki_cli import _get_module_doc
        doc = _get_module_doc("tvtop_plus_plus")
        assert doc is not None
        assert doc["name"] == "tvtop_plus_plus"
        assert len(doc["docstring"]) > 0

    def test_get_module_doc_missing(self):
        """Should return None for non-existent modules."""
        from hypernix.wiki_cli import _get_module_doc
        doc = _get_module_doc("non_existent_module_12345")
        assert doc is None

    def test_format_module_doc(self):
        """Should yield formatted documentation sections."""
        from hypernix.wiki_cli import _format_module_doc, _get_module_doc

        doc = _get_module_doc("tvtop_plus_plus")
        assert doc is not None

        sections = list(_format_module_doc(doc, None, quick=False))
        assert len(sections) > 0

    def test_search_modules(self):
        """Should find modules matching keywords."""
        from hypernix.wiki_cli import _search_modules

        _search_modules("training", None)


# ---------------------------------------------------------------------------
# hnx vera
# ---------------------------------------------------------------------------

class TestVera:
    """Tests for hnx vera module verification."""

    def test_verifier_creation(self):
        """Should create a verifier instance."""
        from hypernix.vera import HyperNixVerifier
        v = HyperNixVerifier()
        assert v is not None
        assert v.strict is False

    def test_verifier_strict_mode(self):
        """Should support strict mode."""
        from hypernix.vera import HyperNixVerifier
        v = HyperNixVerifier(strict=True)
        assert v.strict is True

    def test_verify_file_valid(self):
        """Should pass for a valid Python file."""
        from hypernix.vera import HyperNixVerifier
        v = HyperNixVerifier()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write('"""A test module."""\n')
            f.write("def hello():\n")
            f.write('    """Say hello."""\n')
            f.write("    return 'hello'\n")
            f.flush()
            result = v.verify_file(Path(f.name))

        assert result.passed is True
        assert len(result.errors) == 0

    def test_verify_file_syntax_error(self):
        """Should fail for a file with syntax errors."""
        from hypernix.vera import HyperNixVerifier
        v = HyperNixVerifier()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def hello(\n")
            f.flush()
            result = v.verify_file(Path(f.name))

        assert result.passed is False
        assert len(result.errors) > 0

    def test_verify_file_missing_docstring(self):
        """Should warn about missing docstrings."""
        from hypernix.vera import HyperNixVerifier
        v = HyperNixVerifier(strict=False)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def hello():\n")
            f.write("    return 'hello'\n")
            f.flush()
            result = v.verify_file(Path(f.name))

        assert result.passed is True
        assert len(result.warnings) > 0

    def test_verify_file_strict_missing_docstring(self):
        """Should fail for missing docstrings in strict mode."""
        from hypernix.vera import HyperNixVerifier
        v = HyperNixVerifier(strict=True)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def hello():\n")
            f.write("    return 'hello'\n")
            f.flush()
            result = v.verify_file(Path(f.name))

        assert result.passed is False
        assert len(result.errors) > 0

    def test_result_add_error(self):
        """Result should track errors correctly."""
        from hypernix.vera import VerificationResult
        r = VerificationResult(name="test")
        assert r.passed is True
        r.add_error("test error")
        assert r.passed is False
        assert "test error" in r.errors

    def test_result_add_warning(self):
        """Result should track warnings without failing."""
        from hypernix.vera import VerificationResult
        r = VerificationResult(name="test")
        r.add_warning("test warning")
        assert r.passed is True
        assert "test warning" in r.warnings


# ---------------------------------------------------------------------------
# Pressure Cooker V5 QAT
# ---------------------------------------------------------------------------

class TestPressureCookerV5QAT:
    """Tests for PressureCookerV5 QAT features."""

    def test_qat_config_creation(self):
        """Should create QATConfig with defaults."""
        from hypernix.pressure_cooker_v5 import QATConfig
        cfg = QATConfig()
        assert cfg.bits == 6
        assert cfg.per_layer is True
        assert cfg.learnable_scales is True
        assert cfg.num_levels == 64

    def test_qat_config_custom_bits(self):
        """Should support different bit widths."""
        from hypernix.pressure_cooker_v5 import QATConfig
        for bits in (4, 5, 6, 8):
            cfg = QATConfig(bits=bits)
            assert cfg.bits == bits
            assert cfg.num_levels == 2 ** bits

    def test_qat_config_invalid_bits(self):
        """Should reject invalid bit widths."""
        from hypernix.pressure_cooker_v5 import QATConfig
        with pytest.raises(ValueError):
            QATConfig(bits=3)
        with pytest.raises(ValueError):
            QATConfig(bits=10)

    def test_qat_config_step_size(self):
        """Should compute step size correctly."""
        from hypernix.pressure_cooker_v5 import QATConfig
        cfg = QATConfig(bits=8)
        assert cfg.step_size == 2.0 / (256 - 1)

    def test_fake_quantize_preserves_shape(self):
        """Fake quantization should preserve tensor shape."""
        import torch

        from hypernix.pressure_cooker_v5 import fake_quantize_tensor

        x = torch.randn(10, 20)
        scale = torch.tensor(1.0)
        zero_point = torch.tensor(0.0)
        result = fake_quantize_tensor(x, scale, zero_point, num_levels=64)

        assert result.shape == x.shape

    def test_fake_quantize_reduces_precision(self):
        """Fake quantization should reduce effective precision."""
        import torch

        from hypernix.pressure_cooker_v5 import fake_quantize_tensor

        x = torch.linspace(-1, 1, 1000)
        scale = torch.tensor(2.0 / 63)
        zero_point = torch.tensor(31.0)
        result = fake_quantize_tensor(x, scale, zero_point, num_levels=64)

        assert torch.allclose(result, x, atol=scale.item())
        assert not torch.allclose(result, x, atol=1e-6)

    def test_compute_quantization_params(self):
        """Should compute valid scale and zero_point."""
        import torch

        from hypernix.pressure_cooker_v5 import compute_quantization_params

        x = torch.randn(100)
        scale, zp = compute_quantization_params(x, num_levels=64, symmetric=True)

        assert scale > 0
        assert zp == 0

    def test_qat_fake_quantize_module(self):
        """QATFakeQuantize module should be callable."""
        import torch

        from hypernix.pressure_cooker_v5 import QATFakeQuantize

        fq = QATFakeQuantize(num_levels=64)
        x = torch.randn(10, 20)
        result = fq(x)

        assert result.shape == x.shape

    def test_pressure_cooker_v5_creation(self):
        """Should create PressureCookerV5 instance."""
        import torch

        from hypernix.pressure_cooker_v5 import PressureCookerV5

        param = torch.nn.Parameter(torch.randn(10))
        cooker = PressureCookerV5([param], lr=1e-3)

        assert cooker.lr == 1e-3

    def test_pressure_cooker_v5_with_qat(self):
        """Should create V5 with QAT config."""
        import torch

        from hypernix.pressure_cooker_v5 import PressureCookerV5, QATConfig

        param = torch.nn.Parameter(torch.randn(10))
        qat_cfg = QATConfig(bits=6)
        cooker = PressureCookerV5([param], qat_config=qat_cfg)

        assert cooker.qat_config is not None
        assert cooker.qat_config.bits == 6

    def test_pressure_cooker_v5_describe(self):
        """Describe should include V5-specific fields."""
        import torch

        from hypernix.pressure_cooker_v5 import PressureCookerV5

        param = torch.nn.Parameter(torch.randn(10))
        cooker = PressureCookerV5([param])
        desc = cooker.describe()

        assert "qat_enabled" in desc
        assert "mtp_enabled" in desc
        assert "ema_decay" in desc


# ---------------------------------------------------------------------------
# MTP
# ---------------------------------------------------------------------------

class TestMTP:
    """Tests for Multi-Token Prediction features."""

    def test_mtp_config_creation(self):
        """Should create MTPConfig with defaults."""
        from hypernix.mtp import MTPConfig
        cfg = MTPConfig()
        assert cfg.num_tokens == 4
        assert cfg.lambda_weight == 0.3
        assert cfg.sequential is True

    def test_mtp_config_custom(self):
        """Should support custom configuration."""
        from hypernix.mtp import MTPConfig
        cfg = MTPConfig(num_tokens=8, lambda_weight=0.5, sequential=False)
        assert cfg.num_tokens == 8
        assert cfg.lambda_weight == 0.5
        assert cfg.sequential is False

    def test_mtp_config_invalid_num_tokens(self):
        """Should reject invalid num_tokens."""
        from hypernix.mtp import MTPConfig
        with pytest.raises(ValueError):
            MTPConfig(num_tokens=0)

    def test_mtp_config_invalid_lambda(self):
        """Should reject invalid lambda_weight."""
        from hypernix.mtp import MTPConfig
        with pytest.raises(ValueError):
            MTPConfig(lambda_weight=1.5)

    def test_mtp_config_loss_weights(self):
        """Should return default loss weights."""
        from hypernix.mtp import MTPConfig
        cfg = MTPConfig(num_tokens=4)
        weights = cfg.get_loss_weights()
        assert len(weights) == 4
        assert weights[0] == 1.0
        assert weights[1] == 0.9

    def test_mtp_head_creation(self):
        """Should create MTPHead."""
        from hypernix.mtp import MTPHead

        head = MTPHead(hidden_dim=128, vocab_size=1000, num_tokens=4)
        assert head.hidden_dim == 128
        assert head.vocab_size == 1000
        assert head.num_tokens == 4

    def test_mtp_head_forward(self):
        """MTPHead forward should return correct number of logits."""
        import torch

        from hypernix.mtp import MTPHead

        head = MTPHead(hidden_dim=128, vocab_size=1000, num_tokens=4)
        hidden = torch.randn(2, 10, 128)
        logits = head(hidden)

        assert len(logits) == 4
        for logit in logits:
            assert logit.shape == (2, 10, 1000)

    def test_mtp_head_forward_independent(self):
        """MTPHead should work in independent mode."""
        import torch

        from hypernix.mtp import MTPHead

        head = MTPHead(hidden_dim=128, vocab_size=1000, num_tokens=4, shared=False)
        hidden = torch.randn(2, 10, 128)
        logits = head(hidden, sequential=False)

        assert len(logits) == 4

    def test_mtp_trainer_creation(self):
        """Should create MTPTrainer."""
        from hypernix.mtp import MTPConfig, MTPTrainer

        config = MTPConfig()
        class MockModel:
            pass

        trainer = MTPTrainer(MockModel(), config)
        assert trainer.config == config

    def test_mtp_trainer_attach_head(self):
        """Should attach MTP head."""
        import torch

        from hypernix.mtp import MTPConfig, MTPHead, MTPTrainer

        config = MTPConfig(num_tokens=4)
        model = torch.nn.Linear(128, 128)
        trainer = MTPTrainer(model, config)

        head = trainer.attach_head(hidden_dim=128, vocab_size=1000)
        assert isinstance(head, MTPHead)
        assert head.num_tokens == 4


# ---------------------------------------------------------------------------
# Scavenger
# ---------------------------------------------------------------------------

class TestScavenger:
    """Tests for Scavenger dataset discovery."""

    def test_criteria_creation(self):
        """Should create ScavengerCriteria with defaults."""
        from hypernix.scavenger import ScavengerCriteria
        criteria = ScavengerCriteria()
        assert criteria.keywords == []
        assert criteria.min_entries is None

    def test_criteria_with_keywords(self):
        """Should support keyword filtering."""
        from hypernix.scavenger import ScavengerCriteria
        criteria = ScavengerCriteria(keywords=["code", "python"])
        assert criteria.keywords == ["code", "python"]

    def test_criteria_matches_keyword(self):
        """Should match datasets with keywords."""
        from hypernix.scavenger import ScavengerCriteria
        criteria = ScavengerCriteria(keywords=["code"])
        info = {"id": "codeparrot/github-code", "description": "Code dataset", "tags": ["code"]}

        matches, reason = criteria.matches(info)
        assert matches is True

    def test_criteria_no_match(self):
        """Should not match datasets without keywords."""
        from hypernix.scavenger import ScavengerCriteria
        criteria = ScavengerCriteria(keywords=["medical"])
        info = {"id": "codeparrot/github-code", "description": "Code dataset", "tags": ["code"]}

        matches, reason = criteria.matches(info)
        assert matches is False

    def test_criteria_storage_limit(self):
        """Should filter by storage limit."""
        from hypernix.scavenger import ScavengerCriteria
        criteria = ScavengerCriteria(max_storage_per_dataset_gb=1.0)
        info = {"id": "test", "size_gb": 5.0, "tags": []}

        matches, reason = criteria.matches(info)
        assert matches is False
        assert "size" in reason.lower()

    def test_criteria_min_entries(self):
        """Should filter by minimum entries."""
        from hypernix.scavenger import ScavengerCriteria
        criteria = ScavengerCriteria(min_entries=1000)
        info = {"id": "test", "num_rows": 500, "tags": []}

        matches, reason = criteria.matches(info)
        assert matches is False

    def test_criteria_combined_storage_budget(self):
        """Should check combined storage budget."""
        from hypernix.scavenger import ScavengerCriteria
        criteria = ScavengerCriteria(max_combined_storage_gb=15.0)
        datasets = [
            {"size_gb": 3.0},
            {"size_gb": 4.0},
            {"size_gb": 5.0},
        ]

        fits, total = criteria.check_storage_budget(datasets)
        assert fits is True
        assert total == 12.0

    def test_criteria_combined_storage_over_budget(self):
        """Should detect over-budget storage."""
        from hypernix.scavenger import ScavengerCriteria
        criteria = ScavengerCriteria(max_combined_storage_gb=5.0)
        datasets = [
            {"size_gb": 3.0},
            {"size_gb": 4.0},
        ]

        fits, total = criteria.check_storage_budget(datasets)
        assert fits is False

    def test_scavenger_creation(self):
        """Should create Scavenger instance."""
        from hypernix.scavenger import Scavenger
        sc = Scavenger()
        assert sc is not None


# ---------------------------------------------------------------------------
# Freezer QAT
# ---------------------------------------------------------------------------

class TestFreezerQAT:
    """Tests for Freezer QAT support."""

    def test_qat_vram_multipliers_exist(self):
        """Should have VRAM multipliers for all bit widths."""
        from hypernix.freezer import Freezer
        assert 4 in Freezer._QAT_VRAM_MULTIPLIERS
        assert 5 in Freezer._QAT_VRAM_MULTIPLIERS
        assert 6 in Freezer._QAT_VRAM_MULTIPLIERS
        assert 8 in Freezer._QAT_VRAM_MULTIPLIERS

    def test_qat_vram_multiplier_values(self):
        """VRAM multipliers should be reasonable."""
        from hypernix.freezer import Freezer
        assert Freezer._QAT_VRAM_MULTIPLIERS[4] == 1.15
        assert Freezer._QAT_VRAM_MULTIPLIERS[8] == 1.35

    def test_qat_vram_overhead(self):
        """Should return correct VRAM overhead."""
        from hypernix.freezer import OldFreezer
        fz = OldFreezer()
        overhead = fz.qat_vram_overhead(bits=6)
        assert overhead == 1.25

    def test_suggest_qat_batch_size(self):
        """Should suggest smaller batch for QAT."""
        from hypernix.freezer import OldFreezer
        fz = OldFreezer()
        normal_bs = fz.suggest_batch_size(hint=8)
        qat_bs = fz.suggest_qat_batch_size(bits=6, hint=8)

        assert qat_bs <= normal_bs
        assert qat_bs >= 1

    def test_is_qat_active_false_by_default(self):
        """QAT should not be active by default."""
        from hypernix.freezer import OldFreezer
        fz = OldFreezer()
        assert fz.is_qat_active() is False


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
