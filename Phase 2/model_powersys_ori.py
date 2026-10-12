import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

from HybridAttention_QiskitComp_2_2.HybridTransformerBlock_Qiskit_4Stage_SessionParallel import (
    QuantumTransformerBlock,
    QuantumExecutionContext,
    build_quantum_context,
)
from qiskit_machine_learning.connectors import TorchConnector

logger = logging.getLogger(__name__)

import numpy as np

class GELU(nn.Module):
    def forward(self, input):
        return F.gelu(input)


class GPTQuantumConfig:
    """ base GPT config, params common to all GPT versions """
    """ additional parameters for hybrid self-attention """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    q_proj_qubits = 4
    q_proj_layers = 2
    q_sim_qubits = 4
    q_sim_layers = 2
    # Four-stage execution controls.
    # backend:  "aer" or "ibm"
    # strategy: "optimized" (fast differentiable statevector; Aer only)
    #           "hardware"  (direct Estimator/Sampler PUBs at inference; Aer or IBM)
    quantum_backend_type = "aer"
    quantum_execution_strategy = "optimized"
    ibm_backend_name = None          # used only when quantum_backend_type="ibm"
    quantum_shots = 4096
    quantum_optimization_level = 1
    estimator_precision = None
    ibm_channel = None
    ibm_instance = None
    aer_method = None                # e.g. "statevector"; None lets Aer choose
    aer_device = None                # e.g. "CPU" or "GPU" if supported
    report_circuit_stats = False     # logical + transpiled metrics where applicable
    enable_inference_kv_cache = True  # exact-prefix cache; auto-rebuilds on sliding-window rollover
    # Stages 2-4 bypass EstimatorQNN/SamplerQNN forward calls during model.eval().
    # The direct path reads the same TorchConnector.weight parameters.
    direct_primitive_inference = True
    direct_primitive_batch_size = 4096
    direct_primitive_fallback_to_qnn = True
    # Bind learned quantum weights once for inference and retranspile.  Q/K/V
    # vectorized PUBs are still submitted together in one Estimator.run call.
    direct_bind_inference_weights = True
    direct_fuse_qkv_estimator = True
    causal_pair_pruning = True

    # Stage-4 high-shot IBM controls. Batch mode partitions one large independent
    # parameter sweep into several child jobs while retaining the configured shot
    # count for every unique circuit/parameter row.
    ibm_execution_mode = "session_parallel"
    ibm_parallel_jobs = 5
    ibm_parallel_min_parameter_rows = 128
    ibm_parallel_min_shot_work = 500_000
    ibm_parallel_max_rows_per_job = 2048
    ibm_deduplicate_parameter_rows = False

    # Noisy-Aer execution optimizations. Shot branching/runtime binding operate
    # below the model layer and preserve the same logical circuits/noise model.
    aer_precision = "double"
    aer_shot_branching_enable = False
    aer_shot_branching_sampling_enable = False
    aer_runtime_parameter_bind_enable = False
    aer_batched_shots_gpu = False
    aer_max_parallel_threads = 0
    aer_max_parallel_experiments = 1
    aer_max_parallel_shots = 0
    aer_num_threads_per_device = 1

    # Experimental Aer-native shortcuts are OFF by default. Benchmarks on this
    # small-circuit workload showed shot branching/runtime binding/GPU paths can
    # be slower or unstable. High-level PUB vectorization/caching remain enabled.
    aer_dual_executor = False
    aer_estimator_shot_branching_enable = False
    aer_estimator_shot_branching_sampling_enable = False
    aer_estimator_runtime_parameter_bind_enable = False
    aer_sampler_shot_branching_enable = False
    aer_sampler_shot_branching_sampling_enable = False
    aer_sampler_runtime_parameter_bind_enable = False

    # IBM low-latency inference settings. Fixed-weight circuits are prepared
    # before the Session is opened; the shared primitives are then rebound to it.
    ibm_use_session = False
    ibm_resilience_level = 0
    ibm_disable_dynamical_decoupling = True
    ibm_disable_twirling = True
    ibm_rep_delay = None
    ibm_use_min_rep_delay = False

    # Fully local synthetic Aer noise controls. These are ignored unless
    # aer_noise_enabled=True. No IBM service/backend is queried for them.
    aer_noise_enabled = False
    noise_single_qubit_depolarizing = 0.001
    noise_two_qubit_depolarizing = 0.01
    noise_readout_error = 0.02
    noise_include_thermal_relaxation = False
    noise_t1_ns = 100_000.0
    noise_t2_ns = 80_000.0
    noise_one_qubit_gate_time_ns = 50.0
    noise_two_qubit_gate_time_ns = 300.0
    noise_seed_simulator = 123

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)



class QuantumBlockWrapper(nn.Module):
    """ Adapts the QuantumTransformerBlock to the GPT-2 interface. """

    def __init__(self, config, quantum_context: QuantumExecutionContext):
        super().__init__()

        # Reuse ONE shared Qiskit execution context for every transformer block.
        # Do not create a separate Runtime service/backend connection per layer.
        # Initialize your custom quantum block using the extended config
        self.block = QuantumTransformerBlock(
            embed_dim=config.n_embd,
            num_heads=config.n_head,
            q_proj_qubits=config.q_proj_qubits,
            q_proj_layers=config.q_proj_layers,
            q_sim_qubits=config.q_sim_qubits,
            q_sim_layers=config.q_sim_layers,
            use_quantum_similarity=config.use_quantum_similarity,
            quantum_context=quantum_context,
        )

        # Recreate the causal mask from the original powersys model
        # The +1 accounts for the specific sequence handling in your environment formulation
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size + 1, config.block_size + 1))
                             .view(1, 1, config.block_size + 1, config.block_size + 1))

    def forward(self, x):
        B, T, C = x.size()

        # Slice the mask to match the current sequence length
        causal_mask = self.mask[:, :, :T, :T]

        # Pass the mask explicitly into your hybrid block
        return self.block(x, attention_mask=causal_mask)

    def forward_cached(self, x_new, cache=None):
        return self.block.forward_cached(x_new, cache=cache)

class HybridQuantumGPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """
    """  The classical self-attention is replaced by the hybrid self attention """

    def __init__(self, config, quantum_context: QuantumExecutionContext = None):
        super().__init__()

        self.config = config

        # Build one simulator/hardware execution context and share it across all
        # QuantumBlockWrapper/QuantumTransformerBlock instances.  Passing an
        # existing context is useful when several models should reuse the same
        # Qiskit backend/primitives.
        if quantum_context is None:
            quantum_context = build_quantum_context(
                backend_type=getattr(config, "quantum_backend_type", "aer"),
                execution_strategy=getattr(config, "quantum_execution_strategy", "optimized"),
                backend_name=getattr(config, "ibm_backend_name", None),
                min_num_qubits=max(config.q_proj_qubits, config.q_sim_qubits),
                shots=getattr(config, "quantum_shots", 4096),
                optimization_level=getattr(config, "quantum_optimization_level", 1),
                estimator_precision=getattr(config, "estimator_precision", None),
                channel=getattr(config, "ibm_channel", None),
                instance=getattr(config, "ibm_instance", None),
                aer_method=getattr(config, "aer_method", None),
                aer_device=getattr(config, "aer_device", None),
                aer_noise_enabled=getattr(config, "aer_noise_enabled", False),
                noise_single_qubit_depolarizing=getattr(config, "noise_single_qubit_depolarizing", 0.001),
                noise_two_qubit_depolarizing=getattr(config, "noise_two_qubit_depolarizing", 0.01),
                noise_readout_error=getattr(config, "noise_readout_error", 0.02),
                noise_include_thermal_relaxation=getattr(config, "noise_include_thermal_relaxation", False),
                noise_t1_ns=getattr(config, "noise_t1_ns", 100_000.0),
                noise_t2_ns=getattr(config, "noise_t2_ns", 80_000.0),
                noise_one_qubit_gate_time_ns=getattr(config, "noise_one_qubit_gate_time_ns", 50.0),
                noise_two_qubit_gate_time_ns=getattr(config, "noise_two_qubit_gate_time_ns", 300.0),
                noise_seed_simulator=getattr(config, "noise_seed_simulator", None),
                report_circuit_stats=getattr(config, "report_circuit_stats", False),
                direct_primitive_inference=getattr(config, "direct_primitive_inference", True),
                direct_primitive_batch_size=getattr(config, "direct_primitive_batch_size", 4096),
                direct_primitive_fallback_to_qnn=getattr(config, "direct_primitive_fallback_to_qnn", True),
                direct_bind_inference_weights=getattr(config, "direct_bind_inference_weights", True),
                direct_fuse_qkv_estimator=getattr(config, "direct_fuse_qkv_estimator", True),
                causal_pair_pruning=getattr(config, "causal_pair_pruning", True),
                ibm_execution_mode=getattr(config, "ibm_execution_mode", "session_parallel"),
                ibm_parallel_jobs=getattr(config, "ibm_parallel_jobs", 5),
                ibm_parallel_min_parameter_rows=getattr(config, "ibm_parallel_min_parameter_rows", 128),
                ibm_parallel_min_shot_work=getattr(config, "ibm_parallel_min_shot_work", 500_000),
                ibm_parallel_max_rows_per_job=getattr(config, "ibm_parallel_max_rows_per_job", 2048),
                ibm_deduplicate_parameter_rows=getattr(config, "ibm_deduplicate_parameter_rows", False),
                aer_precision=getattr(config, "aer_precision", "double"),
                aer_shot_branching_enable=getattr(config, "aer_shot_branching_enable", False),
                aer_shot_branching_sampling_enable=getattr(config, "aer_shot_branching_sampling_enable", False),
                aer_runtime_parameter_bind_enable=getattr(config, "aer_runtime_parameter_bind_enable", False),
                aer_batched_shots_gpu=getattr(config, "aer_batched_shots_gpu", False),
                aer_max_parallel_threads=getattr(config, "aer_max_parallel_threads", 0),
                aer_max_parallel_experiments=getattr(config, "aer_max_parallel_experiments", 1),
                aer_max_parallel_shots=getattr(config, "aer_max_parallel_shots", 0),
                aer_num_threads_per_device=getattr(config, "aer_num_threads_per_device", 1),
                aer_dual_executor=getattr(config, "aer_dual_executor", False),
                aer_estimator_shot_branching_enable=getattr(config, "aer_estimator_shot_branching_enable", False),
                aer_estimator_shot_branching_sampling_enable=getattr(config, "aer_estimator_shot_branching_sampling_enable", False),
                aer_estimator_runtime_parameter_bind_enable=getattr(config, "aer_estimator_runtime_parameter_bind_enable", False),
                aer_sampler_shot_branching_enable=getattr(config, "aer_sampler_shot_branching_enable", False),
                aer_sampler_shot_branching_sampling_enable=getattr(config, "aer_sampler_shot_branching_sampling_enable", False),
                aer_sampler_runtime_parameter_bind_enable=getattr(config, "aer_sampler_runtime_parameter_bind_enable", False),
                ibm_use_session=getattr(config, "ibm_use_session", False),
                ibm_resilience_level=getattr(config, "ibm_resilience_level", 0),
                ibm_disable_dynamical_decoupling=getattr(config, "ibm_disable_dynamical_decoupling", True),
                ibm_disable_twirling=getattr(config, "ibm_disable_twirling", True),
                ibm_rep_delay=getattr(config, "ibm_rep_delay", None),
                ibm_use_min_rep_delay=getattr(config, "ibm_use_min_rep_delay", False),
            )
        self.quantum_context = quantum_context
        self.state_size = config.state_size

        self.model_type = config.model_type

        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        # self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size + 1, config.n_embd))
        self.global_pos_emb = nn.Parameter(torch.zeros(1, config.max_timestep + 1, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)

        # transformer
        self.blocks = nn.ModuleList([QuantumBlockWrapper(config, self.quantum_context) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.block_size = config.block_size
        self.apply(self._init_weights)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

        self.state_encoder = nn.Sequential(nn.Linear(config.state_size, 64), nn.Tanh(),
                                           nn.Linear(64, config.n_embd), nn.Tanh())

        self.ret_emb = nn.Sequential(nn.Linear(1, config.n_embd), nn.Tanh())

        self.action_embeddings = nn.Sequential(nn.Embedding(config.vocab_size, config.n_embd), nn.Tanh())
        nn.init.normal_(self.action_embeddings[0].weight, mean=0.0, std=0.02)

        # Inference-only autoregressive cache. It is intentionally excluded from
        # state_dict and is reset at every episode or when the context window rolls.
        self.enable_inference_kv_cache = bool(getattr(config, "enable_inference_kv_cache", True))
        self._inference_cache = None

    def reset_inference_cache(self):
        self._inference_cache = None

    def prepare_quantum_inference(self):
        """Eagerly build fixed-weight ISA circuits after loading a checkpoint.

        This moves the one-time binding/retranspilation cost out of the first
        restoration decision. It does not execute either primitive.
        """
        if not (
            self.quantum_context.execution_strategy == "hardware"
            and self.quantum_context.direct_primitive_inference
            and self.quantum_context.direct_bind_inference_weights
        ):
            return
        self.eval()
        n_projection = 0
        n_fidelity = 0
        for module in self.modules():
            # Both optimized quantum module types expose this private preparation
            # hook; distinguish them by the learned weight container they own.
            if not hasattr(module, "_ensure_bound_direct_circuit"):
                continue
            if hasattr(module, "q_layer"):
                module._ensure_bound_direct_circuit()
                n_projection += 1
            elif hasattr(module, "fidelity_qnn"):
                module._ensure_bound_direct_circuit()
                n_fidelity += 1
        print(
            f"Prepared fixed-weight inference circuits: "
            f"{n_projection} QuantumProjection, {n_fidelity} QuantumFidelityKernel"
        )

    def start_ibm_session(self):
        """Open one shared IBM Session after all inference circuits are prepared."""
        return self.quantum_context.start_ibm_session()

    def close_ibm_session(self):
        self.quantum_context.close_ibm_session()

    def set_quantum_inference_budget(self, *, shots=None, estimator_precision=None):
        self.quantum_context.set_inference_budget(
            shots=shots, estimator_precision=estimator_precision
        )

    def _build_token_embeddings(self, states, actions, targets, rtgs, timesteps):
        state_embeddings = self.state_encoder(
            states.reshape(-1, self.state_size).type(torch.float32).contiguous()
        ).reshape(states.shape[0], states.shape[1], self.config.n_embd)
        rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))

        if actions is not None:
            action_embeddings = self.action_embeddings(actions.type(torch.long).squeeze(-1))
            token_embeddings = torch.zeros(
                (states.shape[0], states.shape[1] * 3 - int(targets is None), self.config.n_embd),
                dtype=torch.float32, device=state_embeddings.device,
            )
            token_embeddings[:, ::3, :] = rtg_embeddings
            token_embeddings[:, 1::3, :] = state_embeddings
            token_embeddings[:, 2::3, :] = action_embeddings[:, -states.shape[1] + int(targets is None):, :]
        else:
            token_embeddings = torch.zeros(
                (states.shape[0], states.shape[1] * 2, self.config.n_embd),
                dtype=torch.float32, device=state_embeddings.device,
            )
            token_embeddings[:, ::2, :] = rtg_embeddings
            token_embeddings[:, 1::2, :] = state_embeddings

        # Each trajectory step must keep its own global timestep embedding.
        # Expand one timestep embedding to the RTG/state/action tokens belonging
        # to that step.  This is essential for exact autoregressive KV caching:
        # historical token embeddings no longer change when a new step arrives.
        if timesteps is None:
            raise ValueError("timesteps must be provided")
        if timesteps.dim() == 2:
            timesteps = timesteps.unsqueeze(-1)
        if timesteps.dim() != 3 or timesteps.size(-1) != 1:
            raise ValueError("timesteps must have shape [B,T,1]")
        if timesteps.size(1) != states.size(1):
            raise ValueError(
                f"timesteps length ({timesteps.size(1)}) must match states length "
                f"({states.size(1)}). Pass the cropped timestep history, not only the current timestep."
            )

        batch_size = states.shape[0]
        all_global_pos_emb = torch.repeat_interleave(self.global_pos_emb, batch_size, dim=0)
        step_pos = torch.gather(
            all_global_pos_emb, 1,
            torch.repeat_interleave(timesteps.long(), self.config.n_embd, dim=-1),
        )  # [B,T,E]

        # reward-conditioned layout is [R_t, s_t, a_t] during training and
        # [R_0,s_0,a_0,...,R_t,s_t] during autoregressive inference.
        if actions is not None:
            token_global_pos = step_pos.repeat_interleave(3, dim=1)
            token_global_pos = token_global_pos[:, :token_embeddings.shape[1], :]
        else:
            # First evaluation call contains only [R_0, s_0].
            token_global_pos = step_pos.repeat_interleave(2, dim=1)
            token_global_pos = token_global_pos[:, :token_embeddings.shape[1], :]

        local_pos = self.pos_emb[:, :token_embeddings.shape[1], :]
        return self.drop(token_embeddings + token_global_pos + local_pos)

    def _forward_full_tokens(self, x):
        for block in self.blocks:
            x = block(x)
        return x

    def _forward_cached_tokens(self, x_full, timestep_history):
        """Process only newly appended tokens while the trajectory window grows.

        ``timestep_history`` is stable per state.  A cache is reusable iff the
        current cropped history extends the cached history as an exact prefix.
        When the context window rolls over, the first timestep changes and the
        cache is rebuilt automatically, preserving exact positional semantics.
        """
        _, Tfull, _ = x_full.shape
        current_history = tuple(
            int(v) for v in timestep_history.detach().to("cpu").reshape(-1).tolist()
        )
        cache = self._inference_cache
        cached_tokens = 0 if cache is None else int(cache["num_tokens"])
        old_history = () if cache is None else tuple(cache.get("timestep_history", ()))

        extends_prefix = (
            cache is not None
            and len(current_history) >= len(old_history)
            and current_history[:len(old_history)] == old_history
            and Tfull > cached_tokens
        )

        if not extends_prefix:
            cached_tokens = 0
            layer_caches = [None] * len(self.blocks)
        else:
            layer_caches = cache["layers"]

        x = x_full[:, cached_tokens:, :]
        new_layer_caches = []
        for block, layer_cache in zip(self.blocks, layer_caches):
            x, updated = block.forward_cached(x, cache=layer_cache)
            new_layer_caches.append(updated)

        self._inference_cache = {
            "num_tokens": Tfull,
            "layers": new_layer_caches,
            "timestep_history": current_history,
        }
        return x, cached_tokens

    def checkpoint_metadata(self):
        """Architecture/execution metadata saved with a simulator-trained checkpoint."""
        return {
            "n_embd": self.config.n_embd,
            "n_head": self.config.n_head,
            "n_layer": self.config.n_layer,
            "q_proj_qubits": self.config.q_proj_qubits,
            "q_proj_layers": self.config.q_proj_layers,
            "q_sim_qubits": self.config.q_sim_qubits,
            "q_sim_layers": self.config.q_sim_layers,
            "use_quantum_similarity": self.config.use_quantum_similarity,
            "timestep_encoding": "per_step_stable_v2",
            "training_backend": getattr(self.config, "quantum_backend_type", None),
            "training_strategy": getattr(self.config, "quantum_execution_strategy", None),
            "training_aer_noise_enabled": getattr(self.config, "aer_noise_enabled", False),
        }

    def validate_checkpoint_metadata(self, metadata):
        """Reject a checkpoint if the logical quantum/classical architecture changed."""
        required = [
            "n_embd", "n_head", "n_layer",
            "q_proj_qubits", "q_proj_layers",
            "q_sim_qubits", "q_sim_layers",
            "use_quantum_similarity",
            "timestep_encoding",
        ]
        mismatches = []
        for key in required:
            if key == "timestep_encoding":
                current = "per_step_stable_v2"
                if key not in metadata:
                    mismatches.append((key, "missing/legacy", current))
                    continue
            else:
                current = getattr(self.config, key)
            if key in metadata and current != metadata[key]:
                mismatches.append((key, metadata[key], current))
        if mismatches:
            text = ", ".join(f"{k}: checkpoint={a}, current={b}" for k, a, b in mismatches)
            raise ValueError("Checkpoint architecture is incompatible: " + text)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        # whitelist_weight_modules = (torch.nn.Linear, )
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                # Qiskit TorchConnector exposes trainable circuit angles as
                # a Parameter named ``weight``.  These are rotation angles, not
                # ordinary dense-layer weights, so do not apply AdamW decay.
                elif isinstance(m, TorchConnector) and pn == 'weight':
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')
        no_decay.add('global_pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(
            param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params),)

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    # state, action, and return
    def forward(self, states, actions, targets=None, rtgs=None, timesteps=None):
        x_full = self._build_token_embeddings(states, actions, targets, rtgs, timesteps)

        use_cache = (
            self.enable_inference_kv_cache
            and not self.training
            and targets is None
        )

        if use_cache:
            x, token_offset = self._forward_cached_tokens(
                x_full, timestep_history=timesteps
            )
        else:
            x = self._forward_full_tokens(x_full)
            token_offset = 0

        x = self.ln_f(x)
        token_logits = self.head(x)

        # During cached inference x contains only newly appended tokens. Select
        # state-token positions using their absolute token positions so the output
        # contract remains suitable for autoregressive sample(...): last logit is
        # always the newest state/action prediction.
        if use_cache:
            absolute = torch.arange(
                token_offset, token_offset + token_logits.size(1),
                device=token_logits.device,
            )
            if self.model_type == 'reward_conditioned':
                keep = (absolute % 3) == 1
                logits = token_logits[:, keep, :]
            elif self.model_type == 'naive':
                logits = token_logits
            else:
                raise NotImplementedError()
        else:
            logits = token_logits
            if actions is not None and self.model_type == 'reward_conditioned':
                logits = logits[:, 1::3, :]
            elif actions is None and self.model_type == 'reward_conditioned':
                logits = logits[:, 1:, :]
            elif actions is not None and self.model_type == 'naive':
                logits = logits[:, ::2, :]
            elif actions is None and self.model_type == 'naive':
                logits = logits
            else:
                raise NotImplementedError()

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss
