"""
Unit Tests — U-Net Model
"""

import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.unet import UNet, UNetResNet, build_model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_input():
    return torch.randn(2, 3, 256, 256).to(DEVICE)


@pytest.fixture
def unet_model():
    return UNet(in_channels=3, num_classes=6, features=[32, 64, 128, 256]).to(DEVICE)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestUNet:
    def test_output_shape(self, unet_model, dummy_input):
        with torch.no_grad():
            out = unet_model(dummy_input)
        assert out.shape == (2, 6, 256, 256), f"Expected (2,6,256,256), got {out.shape}"

    def test_no_nan_in_output(self, unet_model, dummy_input):
        with torch.no_grad():
            out = unet_model(dummy_input)
        assert not torch.isnan(out).any(), "Output contains NaN values"

    def test_no_inf_in_output(self, unet_model, dummy_input):
        with torch.no_grad():
            out = unet_model(dummy_input)
        assert not torch.isinf(out).any(), "Output contains Inf values"

    def test_parameter_count(self, unet_model):
        params = unet_model.get_num_parameters()
        assert params["trainable"] > 0, "No trainable parameters"
        print(f"\nTrainable params: {params['trainable']:,}")

    def test_gradient_flow(self, unet_model, dummy_input):
        targets = torch.randint(0, 6, (2, 256, 256)).to(DEVICE)
        criterion = torch.nn.CrossEntropyLoss()

        out  = unet_model(dummy_input)
        loss = criterion(out, targets)
        loss.backward()

        # Check that at least some gradients flowed
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in unet_model.parameters()
        )
        assert has_grad, "No gradients flowed through the model"

    def test_different_input_sizes(self, unet_model):
        """U-Net should handle various input sizes (multiples of 16)."""
        for size in [128, 256, 512]:
            x = torch.randn(1, 3, size, size).to(DEVICE)
            with torch.no_grad():
                out = unet_model(x)
            assert out.shape == (1, 6, size, size), \
                f"Failed for input size {size}: got {out.shape}"

    def test_batch_size_one(self, unet_model):
        x = torch.randn(1, 3, 256, 256).to(DEVICE)
        with torch.no_grad():
            out = unet_model(x)
        assert out.shape == (1, 6, 256, 256)

    def test_num_classes(self):
        for n_cls in [2, 4, 6, 7, 10]:
            model = UNet(num_classes=n_cls, features=[16, 32]).to(DEVICE)
            x = torch.randn(1, 3, 64, 64).to(DEVICE)
            with torch.no_grad():
                out = model(x)
            assert out.shape[1] == n_cls, f"Expected {n_cls} classes, got {out.shape[1]}"


class TestBuildModel:
    def test_build_unet(self):
        cfg = {"architecture": "unet", "in_channels": 3, "num_classes": 6,
               "features": [32, 64], "bilinear": True, "dropout": 0.0}
        model = build_model(cfg).to(DEVICE)
        x = torch.randn(1, 3, 64, 64).to(DEVICE)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 6, 64, 64)

    def test_unknown_architecture(self):
        with pytest.raises(ValueError):
            build_model({"architecture": "unknown_arch"})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
