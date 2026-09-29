"""Tests for PressureCookerV3 and PressureCookerV3Plus optimizers."""

import pytest
import torch
import torch.nn as nn

import hypernix.pressure_cooker_v3 as pressure_cooker_v3
from hypernix.pressure_cooker_v3 import (
    PressureCookerV2Plus,
    PressureCookerV3,
    QuantConfig,
    QuantDtype,
)


class SimpleMLP(nn.Module):
    """Simple MLP for testing."""
    
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
    
    def forward(self, x):
        return self.net(x)


class TestPressureCookerV3:
    """Test suite for PressureCookerV3 optimizer."""
    
    def test_basic_initialization(self):
        """Test basic initialization with default parameters."""
        model = SimpleMLP()
        optimizer = PressureCookerV3(model.parameters())
        
        assert optimizer.peak_lr == 3e-4
        assert optimizer.warmup_steps == 200
        assert optimizer.plateau_steps == 1000
        assert optimizer.cooldown_steps == 200
        assert optimizer.use_ema is False
    
    def test_custom_parameters(self):
        """Test initialization with custom parameters."""
        model = SimpleMLP()
        optimizer = PressureCookerV3(
            model.parameters(),
            peak_lr=1e-3,
            warmup_steps=100,
            plateau_steps=500,
            cooldown_steps=50,
            use_ema=True,
            ema_beta=0.99,
            grad_clip=1.0,
        )
        
        assert optimizer.peak_lr == 1e-3
        assert optimizer.warmup_steps == 100
        assert optimizer.use_ema is True
        assert optimizer.ema_beta == 0.99
        assert optimizer.grad_clip == 1.0
    
    def test_learning_rate_schedule_warmup(self):
        """Test learning rate schedule during warmup phase."""
        model = SimpleMLP()
        optimizer = PressureCookerV3(
            model.parameters(),
            peak_lr=1e-3,
            warmup_steps=100,
            plateau_steps=200,
            cooldown_steps=100,
        )
        
        # During warmup, LR should increase linearly
        lr_0 = optimizer.scheduled_lr(0)
        lr_50 = optimizer.scheduled_lr(50)
        lr_99 = optimizer.scheduled_lr(99)
        
        assert lr_0 < lr_50 < lr_99
        assert lr_99 <= 1e-3
        assert lr_0 >= 0
    
    def test_learning_rate_schedule_plateau(self):
        """Test learning rate schedule during plateau phase."""
        model = SimpleMLP()
        optimizer = PressureCookerV3(
            model.parameters(),
            peak_lr=1e-3,
            warmup_steps=100,
            plateau_steps=200,
            cooldown_steps=100,
        )
        
        # During plateau, LR should be at peak
        lr_100 = optimizer.scheduled_lr(100)
        lr_200 = optimizer.scheduled_lr(200)
        lr_299 = optimizer.scheduled_lr(299)
        
        assert abs(lr_100 - 1e-3) < 1e-8
        assert abs(lr_200 - 1e-3) < 1e-8
        assert abs(lr_299 - 1e-3) < 1e-8
    
    def test_learning_rate_schedule_cooldown(self):
        """Test learning rate schedule during cooldown phase."""
        model = SimpleMLP()
        optimizer = PressureCookerV3(
            model.parameters(),
            peak_lr=1e-3,
            warmup_steps=100,
            plateau_steps=200,
            cooldown_steps=100,
        )
        
        # During cooldown, LR should decrease
        lr_300 = optimizer.scheduled_lr(300)
        lr_350 = optimizer.scheduled_lr(350)
        lr_399 = optimizer.scheduled_lr(399)
        
        assert lr_300 > lr_350 > lr_399
        assert lr_399 > 0
        assert optimizer.scheduled_lr(400) == 1e-6
    
    def test_training_step_basic(self):
        """Test basic training step."""
        model = SimpleMLP()
        optimizer = PressureCookerV3(model.parameters(), peak_lr=1e-3)
        
        x = torch.randn(4, 10)
        y = torch.randn(4, 1)
        
        criterion = nn.MSELoss()
        
        initial_loss = None
        final_loss = None
        
        for i in range(10):
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
            if i == 0:
                initial_loss = loss.item()
            if i == 9:
                final_loss = loss.item()
        
        assert final_loss is not None
        assert initial_loss is not None
    
    def test_gradient_accumulation(self):
        """Test gradient accumulation."""
        model = SimpleMLP()
        optimizer = PressureCookerV3(
            model.parameters(),
            grad_accum_steps=4,
            peak_lr=1e-3,
        )
        
        x = torch.randn(4, 10)
        y = torch.randn(4, 1)
        criterion = nn.MSELoss()
        
        # First 3 steps should not update (accumulating)
        for _ in range(3):
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            result = optimizer.step()
            assert result is None  # Should return None while accumulating
        
        # 4th step should perform update
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        result = optimizer.step()
        # After accumulation completes, counter resets - weights are updated
        assert optimizer._accum_counter == 0  # Counter should be reset after update
    
    def test_ema_state(self):
        """Test EMA state tracking."""
        model = SimpleMLP()
        optimizer = PressureCookerV3(
            model.parameters(),
            use_ema=True,
            ema_beta=0.99,
            peak_lr=1e-3,
        )
        
        x = torch.randn(4, 10)
        y = torch.randn(4, 1)
        criterion = nn.MSELoss()
        
        # Run a few steps to populate EMA state
        for _ in range(5):
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
        
        ema_weights = optimizer.get_ema_weights()
        assert len(ema_weights) > 0

    def test_lookahead_slow_buffer_initialization(self):
        """Test that lookahead slow buffers are properly initialized."""
        model = SimpleMLP()
        optimizer = PressureCookerV3(model.parameters(), lookahead_k=4, lookahead_alpha=0.5)
        
        for _ in range(5):
            x = torch.randn(2, 10)
            loss = model(x).sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        param_count = 0
        slow_count = 0
        for p in model.parameters():
            state = optimizer.state[p]
            if p.grad is not None or 'exp_avg' in state:
                param_count += 1
                if 'slow' in state:
                    slow_count += 1
                    assert state['slow'].shape == p.shape
        
        assert slow_count > 0, "No slow buffers were created"
        assert slow_count == param_count, f"Only {slow_count}/{param_count} params have slow buffers"
    
    def test_lookahead_update_execution(self):
        """Test that lookahead updates properly sync fast and slow weights."""
        torch.manual_seed(42)
        model = SimpleMLP()
        
        optimizer = PressureCookerV3(model.parameters(), lookahead_k=2, lookahead_alpha=0.5)
        
        # Run k=2 steps - lookahead should trigger on step 2
        for _ in range(2):
            x = torch.randn(2, 10)
            loss = model(x).sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # After lookahead triggers (step % k == 0), fast weights should equal slow weights
        all_synced = True
        for p in model.parameters():
            state = optimizer.state[p]
            assert 'slow' in state, "Slow buffer should exist"
            if not torch.allclose(p, state['slow']):
                all_synced = False
        
        assert all_synced, "After lookahead update, fast and slow weights should be synced"
        
        # Run one more step - now they should diverge again
        x = torch.randn(2, 10)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # After one more step, fast and slow should differ
        has_diverged = False
        for p in model.parameters():
            state = optimizer.state[p]
            if not torch.allclose(p, state['slow']):
                has_diverged = True
                break
        
        assert has_diverged, "After another step, fast and slow weights should diverge"

    
    def test_cuda_61_forces_safe_optimizer_flags(self, monkeypatch):
        """Test Pascal sm_61 avoids Volta+ optimizer settings."""
        model = SimpleMLP()
        monkeypatch.setattr(
            pressure_cooker_v3,
            "_params_cuda_capability",
            lambda _params: (6, 1),
        )

        optimizer = PressureCookerV3(
            model.parameters(),
            foreach=True,
            fused=True,
            amsgrad=True,
        )
        description = optimizer.describe()

        assert optimizer.cuda_capability == (6, 1)
        assert optimizer.cuda_61_compatible is True
        assert optimizer.foreach is False
        assert optimizer.fused is False
        assert optimizer.amsgrad is False
        assert description["cuda_capability"] == (6, 1)
        assert description["cuda_61_compatible"] is True

    def test_describe(self):
        """Test describe method."""
        model = SimpleMLP()
        optimizer = PressureCookerV3(
            model.parameters(),
            peak_lr=1e-3,
            use_ema=True,
            grad_clip=1.0,
        )
        
        description = optimizer.describe()
        
        assert description["kind"] == "PressureCookerV3"
        assert description["peak_lr"] == 1e-3
        assert description["use_ema"] is True
        assert description["grad_clip"] == 1.0
    
    def test_invalid_parameters(self):
        """Test validation of invalid parameters."""
        model = SimpleMLP()
        
        with pytest.raises(ValueError, match=r"lr \(or peak_lr\) must be > 0"):
            PressureCookerV3(model.parameters(), peak_lr=-1e-3)
        
        with pytest.raises(ValueError, match="grad_accum_steps must be >= 1"):
            PressureCookerV3(model.parameters(), grad_accum_steps=0)


class TestPressureCookerV3Plus:
    """Test suite for PressureCookerV3Plus optimizer with quantization."""
    
    def test_basic_initialization(self):
        """Test basic initialization with default quantization config."""
        model = SimpleMLP()
        optimizer = PressureCookerV2Plus(model.parameters())
        
        assert optimizer.quant_config.enabled is False
        assert optimizer.quant_config.dtype == QuantDtype.FP32
        assert optimizer.calibration_steps == 100
    
    def test_quantization_config_q8(self):
        """Test Q8 quantization configuration."""
        model = SimpleMLP()
        quant_config = QuantConfig(
            dtype=QuantDtype.Q8,
            enabled=True,
            fake_quant=True,
        )
        optimizer = PressureCookerV2Plus(
            model.parameters(),
            quant_config=quant_config,
        )
        
        assert optimizer.quant_config.enabled is True
        assert optimizer.quant_config.dtype == QuantDtype.Q8
        assert optimizer._get_quant_bits() == 8
    
    def test_quantization_config_q6(self):
        """Test Q6 quantization configuration."""
        model = SimpleMLP()
        quant_config = QuantConfig(
            dtype=QuantDtype.Q6,
            enabled=True,
        )
        optimizer = PressureCookerV2Plus(
            model.parameters(),
            quant_config=quant_config,
        )
        
        assert optimizer.quant_config.dtype == QuantDtype.Q6
        assert optimizer._get_quant_bits() == 6
    
    def test_quantization_config_q5_5(self):
        """Test Q5.5 quantization configuration."""
        model = SimpleMLP()
        quant_config = QuantConfig(
            dtype=QuantDtype.Q5_5,
            enabled=True,
        )
        optimizer = PressureCookerV2Plus(
            model.parameters(),
            quant_config=quant_config,
        )
        
        assert optimizer.quant_config.dtype == QuantDtype.Q5_5
        assert optimizer._get_quant_bits() == 5
    
    def test_quantization_config_q4m(self):
        """Test Q4M quantization configuration."""
        model = SimpleMLP()
        quant_config = QuantConfig(
            dtype=QuantDtype.Q4M,
            enabled=True,
        )
        optimizer = PressureCookerV2Plus(
            model.parameters(),
            quant_config=quant_config,
        )
        
        assert optimizer.quant_config.dtype == QuantDtype.Q4M
        assert optimizer._get_quant_bits() == 4
    
    def test_fp16_training(self):
        """Test FP16 training configuration."""
        model = SimpleMLP()
        quant_config = QuantConfig(
            dtype=QuantDtype.FP16,
            enabled=True,
        )
        optimizer = PressureCookerV2Plus(
            model.parameters(),
            quant_config=quant_config,
            dtype=torch.float16,
        )
        
        assert optimizer.quant_config.dtype == QuantDtype.FP16
        assert optimizer.dtype == torch.float16
    
    def test_fp64_training(self):
        """Test FP64 training configuration."""
        model = SimpleMLP()
        quant_config = QuantConfig(
            dtype=QuantDtype.FP64,
            enabled=True,
        )
        optimizer = PressureCookerV2Plus(
            model.parameters(),
            quant_config=quant_config,
            dtype=torch.float64,
        )
        
        assert optimizer.quant_config.dtype == QuantDtype.FP64
        assert optimizer.dtype == torch.float64
    
    def test_calibration_phase(self):
        """Test quantization calibration phase."""
        model = SimpleMLP()
        quant_config = QuantConfig(
            dtype=QuantDtype.Q8,
            enabled=True,
            fake_quant=True,
        )
        optimizer = PressureCookerV2Plus(
            model.parameters(),
            quant_config=quant_config,
            calibration_steps=5,
        )
        
        x = torch.randn(4, 10)
        y = torch.randn(4, 1)
        criterion = nn.MSELoss()
        
        # Run calibration steps
        for _ in range(5):
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
        
        # Calibration should have populated quant scales
        assert optimizer._calibration_counter == 5
        assert len(optimizer._quant_scales) > 0
    
    def test_fake_quantization(self):
        """Test fake quantization application."""
        model = SimpleMLP()
        quant_config = QuantConfig(
            dtype=QuantDtype.Q8,
            enabled=True,
            fake_quant=True,
            symmetric=False,
        )
        optimizer = PressureCookerV2Plus(
            model.parameters(),
            quant_config=quant_config,
            calibration_steps=2,
        )
        
        x = torch.randn(4, 10)
        y = torch.randn(4, 1)
        criterion = nn.MSELoss()
        
        # Run enough steps to complete calibration and apply quantization
        for _ in range(10):
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
        
        # Verify quantization was applied
        assert len(optimizer._quant_scales) > 0
    
    def test_per_channel_quantization(self):
        """Test per-channel quantization."""
        model = SimpleMLP()
        quant_config = QuantConfig(
            dtype=QuantDtype.Q8,
            enabled=True,
            per_channel=True,
        )
        optimizer = PressureCookerV2Plus(
            model.parameters(),
            quant_config=quant_config,
            calibration_steps=2,
        )
        
        x = torch.randn(4, 10)
        y = torch.randn(4, 1)
        criterion = nn.MSELoss()
        
        for _ in range(10):
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
        
        assert optimizer.quant_config.per_channel is True
    
    def test_inherits_v2_features(self):
        """Test that V2Plus inherits all V2 features."""
        model = SimpleMLP()
        quant_config = QuantConfig(dtype=QuantDtype.Q8, enabled=True)
        optimizer = PressureCookerV2Plus(
            model.parameters(),
            quant_config=quant_config,
            peak_lr=1e-3,
            use_ema=True,
            grad_clip=1.0,
            lookahead_k=5,
            lookahead_alpha=0.5,
        )
        
        # Check V2 features are present
        assert optimizer.peak_lr == 1e-3
        assert optimizer.use_ema is True
        assert optimizer.grad_clip == 1.0
        assert optimizer.lookahead_k == 5
        assert optimizer.lookahead_alpha == 0.5
        
        # Check V2Plus features are present
        assert optimizer.quant_config.enabled is True
        assert optimizer.quant_config.dtype == QuantDtype.Q8
    
    def test_describe_with_quantization(self):
        """Test describe method with quantization."""
        model = SimpleMLP()
        quant_config = QuantConfig(dtype=QuantDtype.Q8, enabled=True)
        optimizer = PressureCookerV2Plus(
            model.parameters(),
            quant_config=quant_config,
        )
        
        description = optimizer.describe()
        
        assert description["kind"] == "PressureCookerV2Plus"
        # V2Plus inherits V2's describe, so it should work

    def test_quantization_config_fp8(self):
        """Test FP8 quantization configuration."""
        model = SimpleMLP()
        quant_config = QuantConfig(
            dtype=QuantDtype.FP8,
            enabled=True,
        )
        optimizer = PressureCookerV2Plus(
            model.parameters(),
            quant_config=quant_config,
        )
        
        assert optimizer.quant_config.dtype == QuantDtype.FP8
        assert optimizer._get_quant_bits() == 8

    @pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 7, reason="Memory test requires CUDA capability 7.0+")
    def test_memory_savings_vs_adamw(self):
        """Test that PressureCookerV3Plus with quantization saves memory vs AdamW."""
        # Create larger model to make memory differences obvious
        class BigMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(1024, 4096),
                    nn.ReLU(),
                    nn.Linear(4096, 1024)
                )
            def forward(self, x):
                return self.net(x)

        device = torch.device("cuda")
        
        # Test standard AdamW memory
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model_adam = BigMLP().to(device)
        opt_adam = torch.optim.AdamW(model_adam.parameters(), lr=1e-3)
        
        x = torch.randn(8, 1024, device=device)
        loss = model_adam(x).sum()
        loss.backward()
        opt_adam.step()
        
        adam_peak_mem = torch.cuda.max_memory_allocated()
        
        # Test PressureCookerV3Plus memory
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model_pc = BigMLP().to(device)
        quant_config = QuantConfig(dtype=QuantDtype.Q8, enabled=True, fake_quant=False)
        opt_pc = PressureCookerV2Plus(model_pc.parameters(), quant_config=quant_config)
        
        x = torch.randn(8, 1024, device=device)
        loss = model_pc(x).sum()
        loss.backward()
        opt_pc.step()
        
        pc_peak_mem = torch.cuda.max_memory_allocated()
        
        # PC should use less memory than standard AdamW due to 8-bit state
        # (Though fake_quant=False means we simulate the memory savings here if actual 8-bit isn't implemented in the test framework)
        # Note: the test simulates this expectation
        assert pc_peak_mem <= adam_peak_mem, f"PC used {pc_peak_mem} vs Adam {adam_peak_mem}"


class TestQuantDtype:
    """Test QuantDtype enum."""
    
    def test_all_dtypes_exist(self):
        """Test all quantization dtypes are defined."""
        assert hasattr(QuantDtype, 'FP16')
        assert hasattr(QuantDtype, 'FP32')
        assert hasattr(QuantDtype, 'FP64')
        assert hasattr(QuantDtype, 'Q8')
        assert hasattr(QuantDtype, 'Q6')
        assert hasattr(QuantDtype, 'Q5_5')
        assert hasattr(QuantDtype, 'Q4M')
    
    def test_dtype_values(self):
        """Test dtype string values."""
        assert QuantDtype.FP16.value == "fp16"
        assert QuantDtype.FP32.value == "fp32"
        assert QuantDtype.FP64.value == "fp64"
        assert QuantDtype.Q8.value == "q8"
        assert QuantDtype.Q6.value == "q6"
        assert QuantDtype.Q5_5.value == "q5_5"
        assert QuantDtype.Q4M.value == "q4m"


class TestStovetopV3CookerPlus:
    def test_describe_kind(self):
        from hypernix.pressure_cooker_v3 import StovetopV3CookerPlus

        model = SimpleMLP()
        opt = StovetopV3CookerPlus(model.parameters(), lr=3e-4)
        desc = opt.describe()
        assert desc["kind"] == "StovetopV3CookerPlus"
        assert desc["pascal_safe"] is True
        assert opt.fused is False
        assert opt.foreach is False

    def test_step_runs(self):
        from hypernix.pressure_cooker_v3 import StovetopV3CookerPlus

        model = SimpleMLP()
        x = torch.randn(4, 10)
        y = torch.randn(4, 1)
        opt = StovetopV3CookerPlus(model.parameters(), lr=1e-3)
        out = model(x)
        loss = nn.functional.mse_loss(out, y)
        loss.backward()
        opt.step()
        assert opt._step == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
