"""
Gate Components for Protected Dual-Engine Hierarchical CGC/PLE Architecture.

This module contains the gate network components for the MTL framework:
- AlphaGateContextEncoder: Encodes alpha engine context for gate decisions
- BetaGateContextEncoder: Encodes beta engine context for gate decisions
- SharedGateStaticMLP: Shared MLP for static feature encoding
- TaskSpecificGate: Task-specific gate network with temperature annealing support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AlphaGateContextEncoder(nn.Module):
    """
    Alpha Gate Context Encoder.

    Encodes the alpha engine's base representation into a context vector
    for gate decision making.

    Args:
        input_channels (int): Number of input channels C. Default: 29
        input_dim (int): Input feature dimension D. Default: 16
        output_dim (int): Output context dimension. Default: 32

    Input:
        H_alpha_base: Tensor of shape [B, C, D] (alpha engine base representation)

    Output:
        Tensor of shape [B, output_dim] (encoded context for alpha gate)
    """

    def __init__(self, input_channels: int = 29, input_dim: int = 16, output_dim: int = 32):
        super().__init__()

        self.input_channels = input_channels
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Adaptive pooling to aggregate across channels and time
        self.channel_pool = nn.AdaptiveAvgPool2d((1, input_dim))

        # Project to output dimension with proper initialization
        self.proj = nn.Linear(input_dim, output_dim)

        # Initialize weights with Xavier for better gradient flow
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, H_alpha_base: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            H_alpha_base: [B, C, D] tensor from alpha engine

        Returns:
            [B, output_dim] context vector
        """
        # H_alpha_base: [B, C, D]
        # Pool across channels: [B, C, D] -> [B, 1, D]
        pooled = self.channel_pool(H_alpha_base.unsqueeze(1))  # [B, 1, 1, D]
        pooled = pooled.squeeze(1).squeeze(1)  # [B, D]

        # Project to output dimension
        context = self.proj(pooled)  # [B, output_dim]

        return context


class BetaGateContextEncoder(nn.Module):
    """
    Beta Gate Context Encoder.

    Encodes the beta engine's base representation into a context vector
    for gate decision making. Similar structure to AlphaGateContextEncoder
    but operates on beta engine features.

    Args:
        input_channels (int): Number of input channels C. Default: 29
        input_dim (int): Input feature dimension D. Default: 16
        output_dim (int): Output context dimension. Default: 32

    Input:
        H_beta_base: Tensor of shape [B, C, D] (beta engine base representation)

    Output:
        Tensor of shape [B, output_dim] (encoded context for beta gate)
    """

    def __init__(self, input_channels: int = 29, input_dim: int = 16, output_dim: int = 32):
        super().__init__()

        self.input_channels = input_channels
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Adaptive pooling to aggregate across channels and time
        self.channel_pool = nn.AdaptiveAvgPool2d((1, input_dim))

        # Project to output dimension with proper initialization
        self.proj = nn.Linear(input_dim, output_dim)

        # Initialize weights with Xavier for better gradient flow
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, H_beta_base: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            H_beta_base: [B, C, D] tensor from beta engine

        Returns:
            [B, output_dim] context vector
        """
        # H_beta_base: [B, C, D]
        # Pool across channels: [B, C, D] -> [B, 1, D]
        pooled = self.channel_pool(H_beta_base.unsqueeze(1))  # [B, 1, 1, D]
        pooled = pooled.squeeze(1).squeeze(1)  # [B, D]

        # Project to output dimension
        context = self.proj(pooled)  # [B, output_dim]

        return context


class SharedGateStaticMLP(nn.Module):
    """
    Shared Static Feature MLP.

    Encodes static features (age, gender, weight, height, BMI) into a
    compact representation for use in gate context computation.

    Args:
        input_dim (int): Number of static features. Default: 5
        hidden_dim (int): Hidden layer dimension. Default: 16
        output_dim (int): Output dimension. Default: 8

    Input:
        x_static: Tensor of shape [B, 5] (static features)

    Output:
        Tensor of shape [B, output_dim] (encoded static features)
    """

    def __init__(self, input_dim: int = 5, hidden_dim: int = 16, output_dim: int = 8):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Two-layer MLP
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

        # Initialize weights
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.zeros_(self.layer1.bias)
        nn.init.xavier_uniform_(self.layer2.weight)
        nn.init.zeros_(self.layer2.bias)

    def forward(self, x_static: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x_static: [B, input_dim] static features

        Returns:
            [B, output_dim] encoded static features
        """
        # First layer with ReLU
        h = self.layer1(x_static)  # [B, hidden_dim]
        h = self.relu(h)

        # Second layer (no activation)
        out = self.layer2(h)  # [B, output_dim]

        return out


class TaskSpecificGate(nn.Module):
    """
    Task-Specific Gate Network with Temperature Annealing Support.

    Computes expert selection weights based on context vector using
    softmax with configurable temperature. Temperature can be learned
    or externally controlled via scheduler.

    Args:
        num_experts (int): Number of experts to select from
        context_dim (int): Dimension of input context vector. Default: 40
            (32 from engine encoder + 8 from static MLP)
        tau_init (float): Initial temperature value. Default: 2.0
            Higher values produce softer distributions, lower values
            produce harder (more deterministic) selections.
        task_name (str): Name of the task for logging/debugging. Default: "unknown"

    Input:
        c_context: Tensor of shape [B, context_dim] (combined context)
        tau_override (float, optional): Override temperature for scheduler control

    Output:
        Tensor of shape [B, num_experts] (expert selection weights)

    Note:
        Temperature annealing strategy:
        - Start with tau=2.0 for exploration (soft selection)
        - Gradually decrease tau to encourage sharper selections
        - Use tau_override for external scheduler control
    """

    def __init__(
        self,
        num_experts: int,
        context_dim: int = 40,
        tau_init: float = 2.0,
        task_name: str = "unknown"
    ):
        super().__init__()

        self.num_experts = num_experts
        self.context_dim = context_dim
        self.task_name = task_name

        # Gate projection: context -> expert logits
        self.gate_proj = nn.Linear(context_dim, num_experts)

        # Learnable temperature parameter
        self.tau = nn.Parameter(torch.tensor(tau_init))

        # Initialize weights
        nn.init.xavier_uniform_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)

    def forward(
        self,
        c_context: torch.Tensor,
        tau_override: float = None
    ) -> torch.Tensor:
        """
        Forward pass with temperature-scaled softmax.

        Args:
            c_context: [B, context_dim] context vector
            tau_override: Optional temperature override for scheduler control

        Returns:
            [B, num_experts] expert selection weights (sum to 1)
        """
        # Compute gate logits
        gate_logits = self.gate_proj(c_context)  # [B, num_experts]

        # Use override temperature if provided, else use learned tau
        tau = tau_override if tau_override is not None else self.tau

        # Temperature-scaled softmax for expert selection
        # Higher tau -> softer distribution (more exploration)
        # Lower tau -> sharper distribution (more exploitation)
        gate_weights = F.softmax(gate_logits / tau, dim=-1)

        return gate_weights

    def get_tau(self) -> float:
        """Get current temperature value."""
        return self.tau.item()

    def set_tau(self, value: float) -> None:
        """Set temperature value (for manual control)."""
        self.tau.data = torch.tensor(value)

    def extra_repr(self) -> str:
        """Extra representation for module printing."""
        return f"num_experts={self.num_experts}, context_dim={self.context_dim}, " \
               f"tau_init={self.tau.item():.3f}, task_name='{self.task_name}'"