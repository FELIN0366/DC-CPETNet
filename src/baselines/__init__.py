"""
Baseline Models for CPET Disease Classification

This module contains baseline models refactored to support:
- Variable-length sequences
- Static feature fusion (EHR + PFT)
- Unified forward signature: forward(x, lengths=None, prior_adj=None, static_x=None)

Models:
- ResNet1D: 1D ResNet for time series classification
- LSTMNet: Bidirectional LSTM baseline
- MedNet: Medical-informed network with multi-stream architecture
- STFinalNet: Spatio-temporal fusion network
- CNNGAF: CNN with GADF image encoding (Sharma et al., EMBC 2022)
- KESTNet: Knowledge-Enhanced Spatio-Temporal Network (Qu et al., BIBE 2024)

Usage:
    from baselines import ResNet1D, LSTMNet, MedNet, STFinalNet, CNNGAF, KESTNet

    model = ResNet1D(
        input_dim=162,
        output_dim=5,
        num_channel=30,
        use_static_features=True,
        static_dim=16
    )
"""

from .common import StaticFeatureEncoder
from .resnet1d import ResNet1D
from .lstmnet import LSTMNet
from .mednet import MedNet
from .stfinalnet import STFinalNet
from .cnn_gaf import CNNGAF, CNNGAFModel
from .kest_net import KESTNet, create_kest_net

# Model registry for easy access
MODEL_REGISTRY = {
    "ResNet1D": ResNet1D,
    "LSTMNet": LSTMNet,
    "MedNet": MedNet,
    "STFinalNet": STFinalNet,
    "CNNGAF": CNNGAF,
    "KESTNet": KESTNet,
}

__all__ = [
    "StaticFeatureEncoder",
    "ResNet1D",
    "LSTMNet",
    "MedNet",
    "STFinalNet",
    "CNNGAF",
    "CNNGAFModel",
    "KESTNet",
    "create_kest_net",
    "MODEL_REGISTRY",
]