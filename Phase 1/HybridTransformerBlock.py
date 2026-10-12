import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import math
from typing import Optional
import random

import pennylane as qml


class QuantumProjection(nn.Module):
    """
    Hybrid quantum projection:
        classical pre-projection -> quantum layer -> classical post-projection

    Input:  [batch, seq_len, in_dim]
    Output: [batch, seq_len, out_dim]
    """

    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            n_qubits: int = 4,
            n_q_layers: int = 2,
            backend: str = "default.qubit",
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_qubits = n_qubits
        self.n_q_layers = n_q_layers

        hidden_dim = max(in_dim, n_qubits * 2)

        self.pre_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_qubits)
        )

        dev = qml.device(backend, wires=n_qubits)

        @qml.qnode(dev, interface="torch")
        def qnode(inputs, weights):
            # REPLACED manual RY loop with AngleEmbedding.
            # AngleEmbedding natively supports 2D batched inputs for broadcasting.
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")

            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        weight_shapes = {"weights": (n_q_layers, n_qubits, 3)}
        self.q_layer = qml.qnn.TorchLayer(qnode, weight_shapes)

        self.post_net = nn.Sequential(
            nn.Linear(n_qubits, out_dim),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq_len, in_dim]
        returns: [batch, seq_len, out_dim]
        """
        batch_size, seq_len, _ = x.shape

        x = self.pre_net(x)  # [B, S, n_qubits]
        x = torch.tanh(x) * (math.pi / 2.0)

        x = x.reshape(batch_size * seq_len, self.n_qubits)

        # ----- REMOVED FOR LOOP -----
        # Pass the batched tensor directly to the TorchLayer.
        # PennyLane will broadcast the circuit evaluation across the batch dimension.
        x = self.q_layer(x)

        # Ensure correct dtype and device mapping back to the classical network
        x = x.to(
            device=next(self.post_net.parameters()).device,
            dtype=next(self.post_net.parameters()).dtype
        )  # [B*S, n_qubits]

        x = self.post_net(x)  # [B*S, out_dim]
        x = x.reshape(batch_size, seq_len, self.out_dim)

        return x

class QuantumStateEncoder(nn.Module):
    """
    Encodes classical vectors into quantum states and returns the statevector.

    This is designed for simulator-based fidelity kernels:
        fidelity(x, y) = |<psi(x) | psi(y)>|^2

    Important:
      - Requires a statevector simulator backend (e.g. default.qubit)
      - Uses one circuit call per token, not per token pair
    """

    def __init__(
        self,
        in_dim: int,
        n_qubits: int = 4,
        n_q_layers: int = 2,
        backend: str = "default.qubit",
    ):
        super().__init__()
        self.in_dim = in_dim
        self.n_qubits = n_qubits
        self.n_q_layers = n_q_layers
        self.state_dim = 2 ** n_qubits

        hidden_dim = max(in_dim, n_qubits * 2)

        self.pre_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_qubits)
        )
        self.weights = nn.Parameter(
            0.01 * torch.randn(n_q_layers, n_qubits, 3)
        )

        dev = qml.device(backend, wires=n_qubits)

        @qml.qnode(dev, interface="torch")
        def qnode(inputs, weights):
            # Simple differentiable feature map
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return qml.state()

        self.qnode = qnode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [N, in_dim]
        returns: [N, 2**n_qubits] complex statevectors
        """
        x = self.pre_net(x)                              # [N, n_qubits]
        x = torch.tanh(x) * (math.pi / 2.0)

        # PennyLane QNodes generally do not fully vectorize over two-dimensional
        # token batches in a backend-agnostic way, so we evaluate once per token.
        # This is still much faster than once per token pair.
        # ----- REMOVED FOR LOOP -----
        # Pass the fully batched tensor directly into the QNode.
        # PennyLane broadcasts the AngleEmbedding over the N dimension.
        states = self.qnode(x, self.weights)  # [N, 2**n_qubits] complex

        states = states.to(
            device=x.device,
            dtype=states.dtype
        )
        return states

class QuantumFidelityKernel(nn.Module):
    """
    Fidelity-based quantum similarity.

    Given q and k of shape [B, H, S, D], compute:
        K[b, h, i, j] = |<psi(q[b,h,i]) | psi(k[b,h,j])>|^2

    Strategy:
      1) Flatten all Q tokens and encode once
      2) Flatten all K tokens and encode once
      3) Reshape back to [B, H, S, state_dim]
      4) Compute pairwise overlaps using vectorized complex matmul

    This removes the expensive S^2 circuit loop.
    """

    def __init__(
        self,
        in_dim: int,
        n_qubits: int = 4,
        n_q_layers: int = 2,
        backend: str = "default.qubit",
        normalize: bool = False,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.n_qubits = n_qubits
        self.n_q_layers = n_q_layers
        self.normalize = normalize
        self.eps = eps

        self.q_state_encoder = QuantumStateEncoder(
            in_dim=in_dim,
            n_qubits=n_qubits,
            n_q_layers=n_q_layers,
            backend=backend,
        )
        self.k_state_encoder = QuantumStateEncoder(
            in_dim=in_dim,
            n_qubits=n_qubits,
            n_q_layers=n_q_layers,
            backend=backend,
        )

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        q: [B, H, S, D]
        k: [B, H, S, D]

        returns logits: [B, H, S, S]
        """
        B, H, S, D = q.shape

        q_flat = q.reshape(B * H * S, D)
        k_flat = k.reshape(B * H * S, D)

        q_states = self.q_state_encoder(q_flat)         # [B*H*S, state_dim], complex
        k_states = self.k_state_encoder(k_flat)         # [B*H*S, state_dim], complex

        q_states = q_states.reshape(B, H, S, -1)        # [B, H, S, state_dim]
        k_states = k_states.reshape(B, H, S, -1)

        if self.normalize:
            q_norm = torch.linalg.norm(q_states, dim=-1, keepdim=True).clamp_min(self.eps)
            k_norm = torch.linalg.norm(k_states, dim=-1, keepdim=True).clamp_min(self.eps)
            q_states = q_states / q_norm
            k_states = k_states / k_norm

        # Complex overlap matrix:
        # overlaps[b,h,i,j] = <psi(q_i) | psi(k_j)>
        overlaps = torch.matmul(
            torch.conj(q_states),
            k_states.transpose(-2, -1)
        )                                               # [B, H, S, S]

        # Fidelity kernel = squared magnitude of overlap
        logits = torch.abs(overlaps) ** 2               # real-valued [B, H, S, S]

        return logits

class HybridQuantumSelfAttention(nn.Module):
    """
    Hybrid quantum-classical multi-head self-attention.

    Modes:
      - use_quantum_similarity=False:
          quantum Q/K/V projections + classical dot-product attention
      - use_quantum_similarity=True:
          quantum Q/K/V projections + fidelity-based quantum kernel similarity

    Input:
      inputs: [batch, seq_len, embed_dim]

    Output:
      output:  [batch, seq_len, embed_dim]
      weights: [batch, num_heads, seq_len, seq_len]
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        q_proj_qubits: int = 4,
        q_proj_layers: int = 2,
        use_quantum_similarity: bool = False,
        q_sim_qubits: int = 4,
        q_sim_layers: int = 2,
        backend: str = "default.qubit",
        dropout: float = 0.0,
    ):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.projection_dim = embed_dim // num_heads
        self.use_quantum_similarity = use_quantum_similarity

        self.query_proj = QuantumProjection(
            in_dim=embed_dim,
            out_dim=embed_dim,
            n_qubits=q_proj_qubits,
            n_q_layers=q_proj_layers,
            backend=backend,
        )
        self.key_proj = QuantumProjection(
            in_dim=embed_dim,
            out_dim=embed_dim,
            n_qubits=q_proj_qubits,
            n_q_layers=q_proj_layers,
            backend=backend,
        )
        self.value_proj = QuantumProjection(
            in_dim=embed_dim,
            out_dim=embed_dim,
            n_qubits=q_proj_qubits,
            n_q_layers=q_proj_layers,
            backend=backend,
        )

        if self.use_quantum_similarity:
            self.quantum_similarity = QuantumFidelityKernel(
                in_dim=self.projection_dim,
                n_qubits=q_sim_qubits,
                n_q_layers=q_sim_layers,
                backend=backend,
            )
            self.quantum_scale = nn.Parameter(torch.tensor(10.0))
        else:
            self.quantum_similarity = None

        self.attn_dropout = nn.Dropout(dropout)
        self.combine_heads = nn.Linear(embed_dim, embed_dim)

    def split_heads(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        x: [B, S, E] -> [B, H, S, D]
        """
        x = x.view(batch_size, -1, self.num_heads, self.projection_dim)
        return x.transpose(1, 2)

    def classical_attention_logits(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> torch.Tensor:
        """
        query, key: [B, H, S, D]
        returns logits: [B, H, S, S]
        """
        score = torch.matmul(query, key.transpose(-2, -1))
        scale = math.sqrt(key.size(-1))
        return score / scale

    def attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        if self.use_quantum_similarity:
            # Quantum-native attention replacement:
            #   scores[b,h,i,j] = |<psi(q_i)|psi(k_j)>|^2
            # These fidelity scores are already non-negative, so instead of
            # applying a classical softmax, use row normalization:
            #   weights_ij = scores_ij / sum_j scores_ij
            scores = self.quantum_similarity(query, key)  # [B, H, S, S]
            scores = scores.to(device=value.device, dtype=value.dtype)

            if attention_mask is not None:
                scores = scores.masked_fill(attention_mask == 0, 0.0)

            denom = scores.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            weights = scores / denom
        else:
            logits = self.classical_attention_logits(query, key)

            if attention_mask is not None:
                logits = logits.masked_fill(attention_mask == 0, float("-inf"))

            weights = F.softmax(logits, dim=-1)

        output = torch.matmul(weights, value)                     # [B, H, S, D]
        return output, weights

    def forward(
        self,
        inputs: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        batch_size = inputs.size(0)

        query = self.split_heads(self.query_proj(inputs), batch_size)
        key   = self.split_heads(self.key_proj(inputs), batch_size)
        value = self.split_heads(self.value_proj(inputs), batch_size)

        attention, weights = self.attention(query, key, value, attention_mask)

        attention = (
            attention.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.embed_dim)
        )

        output = self.combine_heads(attention)
        return output, weights


class QuantumTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads,
                 q_proj_qubits, q_proj_layers,
                 q_sim_qubits, q_sim_layers,
                 use_quantum_similarity,
                 backend="default.qubit"):
        super().__init__()
        # Pre-LayerNorms
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.ln_2 = nn.LayerNorm(embed_dim)

        # Your custom hybrid quantum attention
        self.attn = HybridQuantumSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            q_proj_qubits=q_proj_qubits,
            q_proj_layers=q_proj_layers,
            use_quantum_similarity=use_quantum_similarity,
            q_sim_qubits=q_sim_qubits,
            q_sim_layers=q_sim_layers,
            backend=backend,
            dropout=0.1
        )

        # Standard GPT-2 MLP
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(0.1)
        )

    def forward(self, x, attention_mask=None):
        # Attention path with residual
        attn_out, _ = self.attn(self.ln_1(x), attention_mask=attention_mask)
        x = x + attn_out

        # MLP path with residual
        x = x + self.mlp(self.ln_2(x))
        return x