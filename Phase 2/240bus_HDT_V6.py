import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import random

from data_creater_240_ori import create_dataset
from mingpt_240_5.model_powersys_ori import HybridQuantumGPT, GPTQuantumConfig
from mingpt_240_5.train_powersys_ori import Trainer, TrainerConfig


# =============================================================================
# Reproducibility
# =============================================================================
seed = 63
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# =============================================================================
# Four-stage workflow selector
# =============================================================================
# 1) "train_aer_optimized"
#       Fast differentiable PyTorch statevector training. Same logical circuit
#       and same q_layer/fidelity_qnn weights as the hardware path.
# 2) "validate_aer_hardware"
#       Load the trained checkpoint and evaluate with the hardware-style
#       EstimatorQNN/SamplerQNN path on Aer.
# 3) "validate_aer_noisy"
#       Load the same checkpoint and evaluate on a fully LOCAL synthetic noisy
#       Aer CPU backend. Aer-native shot branching, runtime parameter binding, and
#       GPU batching are OFF by default because benchmarked combinations were
#       slower and/or unstable for this small-circuit workload. High-level PUB
#       vectorization, caching, fixed-weight binding and retranspilation remain.
# 4) "infer_ibm"
#       Load the exact same checkpoint and perform inference on a real IBM QPU.
#       This is the only stage that accesses IBM Quantum services.
WORKFLOW_STAGE = "infer_ibm"

CHECKPOINT_PATH = "model_HDT_240Bus_Qiskit_4Stage_A_1.pt"


# =============================================================================
# Data / model settings
# =============================================================================
data_dir_prefix = './240Bus_Graph_Data_V3_2/'
context_length = 1  # preserved from the original 240-bus scenario
model_type = 'reward_conditioned'
epochs = 2
batch_size = 32
action_size = 23**5

q_proj_qubits = 4
q_proj_layers = 1
q_sim_qubits = 3
q_sim_layers = 1
use_quantum_similarity = True


# =============================================================================
# Quantum execution settings
# =============================================================================
# Stage-specific shot budgets. Optimized Aer ignores shots. Start lower for
# hardware-style validation and increase only if action decisions are unstable.
ideal_aer_shots = 2048
# Stage 3 remains a lower-cost validation profile. Stage 4 deliberately keeps a
# high shot budget; do not lower this to obtain the batching speedup. Increase
# ibm_hardware_shots to 10000+ if required by the hardware experiment.
noisy_aer_shots = 2048
ibm_hardware_shots = 2048
quantum_shots = ideal_aer_shots

# Estimator target precision. 0.05 is intentionally looser than IBM's common
# 1/sqrt(4096) default and should be validated against action stability.
ideal_aer_estimator_precision = None
noisy_aer_estimator_precision = 0.05
ibm_estimator_precision = 0.05
estimator_precision = ideal_aer_estimator_precision

# Evaluation workload budgets. Hardware-style paths are intentionally tested on
# fewer episodes because each autoregressive step requires quantum primitive work.
optimized_aer_eval_episodes = 50
ideal_aer_eval_episodes = 10
noisy_aer_eval_episodes = 10
ibm_eval_episodes = 10
eval_episodes = optimized_aer_eval_episodes

quantum_optimization_level = 2
report_circuit_stats = True
enable_inference_kv_cache = True  # implemented, but context_length=1 limits cross-step cache reuse
# Direct V2 primitive inference is used only in model.eval() hardware-style stages.
# Stage 1 optimized training keeps the differentiable PyTorch statevector path.
direct_primitive_inference = True
direct_primitive_batch_size = 4096
direct_primitive_fallback_to_qnn = True
# Inference-only: bind learned theta values, retranspile once, and vectorize only
# the data angles. Q/K/V bound PUBs are fused into one Estimator.run call.
direct_bind_inference_weights = True
direct_fuse_qkv_estimator = True
# Stages 2-4: do not submit fidelity pairs that the causal mask discards.
# Stage 1 keeps its dense statevector overlap matmul.
causal_pair_pruning = True

# Used by Qiskit's Aer backend in hardware-style validation. In optimized Aer
# training, the quantum math is executed by the differentiable PyTorch
# statevector path on the same device as the model, so aer_device is not the
# performance control for that stage.
# Ideal-Aer defaults. Stage 3 overrides these with the noisy-Aer profile below.
aer_method = "statevector"
aer_device = "CPU"
aer_precision = "double"
aer_shot_branching_enable = False
aer_shot_branching_sampling_enable = False
aer_runtime_parameter_bind_enable = False
aer_batched_shots_gpu = False
aer_max_parallel_threads = 0
aer_max_parallel_experiments = 1
aer_max_parallel_shots = 0
aer_num_threads_per_device = 1

# Dual-Aer settings are disabled outside Stage 3.
aer_dual_executor = False
aer_estimator_shot_branching_enable = False
aer_estimator_shot_branching_sampling_enable = False
aer_estimator_runtime_parameter_bind_enable = False
aer_sampler_shot_branching_enable = False
aer_sampler_shot_branching_sampling_enable = False
aer_sampler_runtime_parameter_bind_enable = False

# Noisy-Aer STABLE production profile.  The expensive but reliable optimizations
# remain ABOVE Aer (new-token/KV caching, direct primitive PUB vectorization,
# fixed-weight binding, cached retranspilation).  Native Aer shortcuts are OFF
# because measurements on this 3-4 qubit workload showed no speed benefit and
# GPU/combined native paths produced native crashes.
noisy_aer_method = "statevector"
noisy_aer_device = "CPU"
noisy_aer_precision = "double"
noisy_aer_batched_shots_gpu = False

# Benchmarked Stage-3 setting: automatic CPU shot parallelism reduced the
# evaluation period from about 1900 s to about 300 s on the user's workload.
# Keep experiment parallelism disabled so Aer can devote the thread pool to
# independent noisy-shot trajectories.
noisy_aer_max_parallel_threads = 0
noisy_aer_max_parallel_experiments = 1
noisy_aer_max_parallel_shots = 0
noisy_aer_num_threads_per_device = 1

# Native Aer micro-optimizations disabled by default. Primitive/PUB parameter
# vectorization remains enabled elsewhere in the code and is NOT disabled here.
noisy_aer_shot_branching_enable = False
noisy_aer_shot_branching_sampling_enable = False
noisy_aer_runtime_parameter_bind_enable = False

# Dual executors are retained as an experimental capability in the backend
# builder, but are not used by the production Stage-3 profile.
noisy_aer_dual_executor = False
noisy_aer_estimator_shot_branching_enable = False
noisy_aer_estimator_shot_branching_sampling_enable = False
noisy_aer_estimator_runtime_parameter_bind_enable = False
noisy_aer_sampler_shot_branching_enable = False
noisy_aer_sampler_shot_branching_sampling_enable = False
noisy_aer_sampler_runtime_parameter_bind_enable = False

# -----------------------------------------------------------------------------
# LOCAL synthetic noise model used only by WORKFLOW_STAGE="validate_aer_noisy".
# These numbers are user-controlled assumptions; they are NOT read from IBM.
# -----------------------------------------------------------------------------
noise_single_qubit_depolarizing = 0.001   # 0.1% per RY/RZ
noise_two_qubit_depolarizing = 0.01       # 1.0% per CX
noise_readout_error = 0.02                # symmetric 2% readout bit flip
noise_include_thermal_relaxation = False
noise_t1_ns = 100_000.0
noise_t2_ns = 80_000.0
noise_one_qubit_gate_time_ns = 50.0
noise_two_qubit_gate_time_ns = 300.0
noise_seed_simulator = seed

# IBM hardware selector. None -> least-busy suitable real QPU.
# These settings are ignored in every Aer stage.
ibm_backend_name = "ibm_phoenix"
ibm_channel = None
ibm_instance = None

# IBM high-shot Stage-4 profile. ``session_parallel`` opens one persistent
# Session for the complete iterative evaluation and submits independent child
# jobs (Q/K/V or fidelity chunks) before waiting for their results. This keeps
# dedicated QPU access while avoiding repeated Batch <-> Job mode switching.
ibm_execution_mode = "session_parallel"   # "session_parallel" recommended for Stage 4
ibm_use_session = (ibm_execution_mode in {"session", "session_parallel"})  # compatibility mirror
ibm_parallel_jobs = 5
# Row-based chunking trigger. Estimator sweeps use this criterion directly.
ibm_parallel_min_parameter_rows = 128
# Sampler-specific high-shot trigger. Fidelity sweeps are also eligible for parallel
# Session chunking when unique_rows * ibm_hardware_shots reaches this value, even if
# unique_rows is below ibm_parallel_min_parameter_rows. Example: 100 rows x 10,000
# shots = 1,000,000, so it will split with the default 500,000 threshold.
# Set <= 0 to disable this shot-work trigger and recover row-only behavior.
ibm_parallel_min_shot_work = 500_000
# Hard cap per child job. At 10k+ shots lower this if a child job approaches IBM's
# per-job maximum execution-time limit on the selected backend.
ibm_parallel_max_rows_per_job = 2048
# Exact equality deduplication across the N episode batch. Identical quantum input
# rows are executed once and scattered back to every matching episode. This
# preserves the expectation value but intentionally shares finite-shot noise
# across duplicates instead of drawing independent hardware samples.
ibm_deduplicate_parameter_rows = True

ibm_resilience_level = 0
ibm_disable_dynamical_decoupling = True
ibm_disable_twirling = True
# None keeps the backend default. A lower valid value can reduce QPU time but
# may increase state-preparation error; keep None until you benchmark accuracy.
ibm_rep_delay = None
# Optional speed/accuracy tradeoff. If True and ibm_rep_delay is None, use the
# selected backend's minimum supported repetition delay. Validate action/QPU
# fidelity before using this for final reported experiments.
ibm_use_min_rep_delay = False

# Stage-4 episode batching. The 240-bus environment uses a module-global OpenDSS
# feeder, so each of the N=10 episodes is isolated in its own Linux process. The
# parent process sends all currently active episode states through one batched
# model forward. context_length=1 makes this exact without padding fake tokens.
ibm_parallel_episode_evaluation = True
ibm_parallel_episode_workers = 10
ibm_parallel_env_start_method = "spawn"

# Stage-derived settings; individual branches below override these as needed.
direct_primitive_fallback_to_qnn_stage = direct_primitive_fallback_to_qnn

if WORKFLOW_STAGE == "train_aer_optimized":
    quantum_backend_type = "aer"
    quantum_execution_strategy = "optimized"
    do_train = True
    load_checkpoint = False
    force_cpu = False       # allows fast differentiable statevector path on CUDA
    aer_noise_enabled = False
    quantum_shots = ideal_aer_shots
    estimator_precision = ideal_aer_estimator_precision
    eval_episodes = optimized_aer_eval_episodes

elif WORKFLOW_STAGE == "validate_aer_hardware":
    quantum_backend_type = "aer"
    quantum_execution_strategy = "hardware"
    do_train = False
    load_checkpoint = True
    force_cpu = True        # safest interface for Qiskit TorchConnector
    aer_noise_enabled = False
    quantum_shots = ideal_aer_shots
    estimator_precision = ideal_aer_estimator_precision
    eval_episodes = ideal_aer_eval_episodes

elif WORKFLOW_STAGE == "validate_aer_noisy":
    quantum_backend_type = "aer"
    quantum_execution_strategy = "hardware"
    do_train = False
    load_checkpoint = True
    force_cpu = True
    aer_noise_enabled = True
    quantum_shots = noisy_aer_shots
    estimator_precision = noisy_aer_estimator_precision
    eval_episodes = noisy_aer_eval_episodes
    # Stage-3 backend optimization profile.
    aer_method = noisy_aer_method
    aer_device = noisy_aer_device
    aer_precision = noisy_aer_precision
    aer_shot_branching_enable = noisy_aer_shot_branching_enable
    aer_shot_branching_sampling_enable = noisy_aer_shot_branching_sampling_enable
    aer_runtime_parameter_bind_enable = noisy_aer_runtime_parameter_bind_enable
    aer_batched_shots_gpu = noisy_aer_batched_shots_gpu
    aer_max_parallel_threads = noisy_aer_max_parallel_threads
    aer_max_parallel_experiments = noisy_aer_max_parallel_experiments
    aer_max_parallel_shots = noisy_aer_max_parallel_shots
    aer_num_threads_per_device = noisy_aer_num_threads_per_device
    aer_dual_executor = noisy_aer_dual_executor
    aer_estimator_shot_branching_enable = noisy_aer_estimator_shot_branching_enable
    aer_estimator_shot_branching_sampling_enable = noisy_aer_estimator_shot_branching_sampling_enable
    aer_estimator_runtime_parameter_bind_enable = noisy_aer_estimator_runtime_parameter_bind_enable
    aer_sampler_shot_branching_enable = noisy_aer_sampler_shot_branching_enable
    aer_sampler_shot_branching_sampling_enable = noisy_aer_sampler_shot_branching_sampling_enable
    aer_sampler_runtime_parameter_bind_enable = noisy_aer_sampler_runtime_parameter_bind_enable
    # Do not hide a direct-primitive error behind a slower QNN/TorchConnector
    # fallback; this also avoids secondary native crashes masking the root cause.
    direct_primitive_fallback_to_qnn_stage = False

elif WORKFLOW_STAGE == "infer_ibm":
    quantum_backend_type = "ibm"
    quantum_execution_strategy = "hardware"
    do_train = False
    load_checkpoint = True
    force_cpu = True
    aer_noise_enabled = False
    quantum_shots = ibm_hardware_shots
    estimator_precision = ibm_estimator_precision
    eval_episodes = ibm_eval_episodes
    # Direct primitive failures should be visible on hardware; QNN fallback would
    # use the pre-Session job-mode primitive and defeat the Session guarantee.
    direct_primitive_fallback_to_qnn_stage = False

else:
    raise ValueError(f"Unknown WORKFLOW_STAGE={WORKFLOW_STAGE!r}")


# =============================================================================
# Dataset
# =============================================================================
class StateActionReturnDataset(Dataset):
    def __init__(self, data, block_size, actions, done_idxs, rtgs, timesteps):
        self.block_size = block_size
        self.vocab_size = action_size
        self.data = data
        self.actions = actions
        self.done_idxs = done_idxs
        self.rtgs = rtgs
        self.timesteps = timesteps

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        block_size = self.block_size // 3
        done_idx = idx + block_size
        for i in self.done_idxs:
            if i > idx:
                done_idx = min(int(i), done_idx)
                break
        idx = done_idx - block_size
        states = torch.tensor(np.array(self.data[idx:done_idx]), dtype=torch.float32)
        actions = torch.tensor(self.actions[idx:done_idx], dtype=torch.long).unsqueeze(1)
        rtgs = torch.tensor(self.rtgs[idx:done_idx], dtype=torch.float32).unsqueeze(1)
        timesteps = torch.tensor(self.timesteps[idx:done_idx], dtype=torch.int64).reshape(-1, 1)
        return states, actions, rtgs, timesteps


def create_graph():
    # Preserve the original 23-node graph used by the 240-bus scenario.
    edges = [[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4,
              5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 9,
              10, 10, 10, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 14, 14, 14,
              15, 15, 15, 15, 16, 16, 16, 16, 16, 17, 17, 18, 18, 18, 18,
              19, 19, 19, 20, 20, 21, 21, 21, 21, 22, 22],
             [0, 1, 2, 3, 0, 1, 4, 14, 0, 2, 6, 0, 3, 6, 11, 1, 4, 5, 10,
              4, 5, 9, 2, 3, 6, 7, 13, 16, 6, 7, 8, 7, 8, 9, 21, 5, 8, 9, 10, 15,
              4, 9, 10, 3, 11, 12, 11, 12, 13, 14, 15, 6, 12, 13, 1, 12, 14,
              9, 12, 15, 16, 6, 15, 16, 17, 18, 16, 17, 16, 18, 19, 21,
              18, 19, 20, 19, 20, 8, 18, 21, 22, 21, 22]]
    edge_index = torch.tensor(edges, dtype=torch.long)
    x = torch.eye(23)
    return Data(x=x, edge_index=edge_index)


def main():
    states, actions, rtg, traj_indx, timesteps = create_dataset(data_dir_prefix)
    train_dataset = StateActionReturnDataset(
        states, context_length * 3, actions, traj_indx, rtg, timesteps
    )


    # =============================================================================
    # Model
    # =============================================================================
    mconf = GPTQuantumConfig(
        train_dataset.vocab_size,
        train_dataset.block_size,
        state_size=len(states[0]),
        n_layer=6,
        n_head=8,
        n_embd=128,
        model_type=model_type,
        max_timestep=max(timesteps),
        q_proj_layers=q_proj_layers,
        q_proj_qubits=q_proj_qubits,
        q_sim_layers=q_sim_layers,
        q_sim_qubits=q_sim_qubits,
        use_quantum_similarity=use_quantum_similarity,
        quantum_backend_type=quantum_backend_type,
        quantum_execution_strategy=quantum_execution_strategy,
        quantum_shots=quantum_shots,
        quantum_optimization_level=quantum_optimization_level,
        estimator_precision=estimator_precision,
        aer_method=aer_method,
        aer_device=aer_device,
        aer_precision=aer_precision,
        aer_shot_branching_enable=aer_shot_branching_enable,
        aer_shot_branching_sampling_enable=aer_shot_branching_sampling_enable,
        aer_runtime_parameter_bind_enable=aer_runtime_parameter_bind_enable,
        aer_batched_shots_gpu=aer_batched_shots_gpu,
        aer_max_parallel_threads=aer_max_parallel_threads,
        aer_max_parallel_experiments=aer_max_parallel_experiments,
        aer_max_parallel_shots=aer_max_parallel_shots,
        aer_num_threads_per_device=aer_num_threads_per_device,
        aer_dual_executor=aer_dual_executor,
        aer_estimator_shot_branching_enable=aer_estimator_shot_branching_enable,
        aer_estimator_shot_branching_sampling_enable=aer_estimator_shot_branching_sampling_enable,
        aer_estimator_runtime_parameter_bind_enable=aer_estimator_runtime_parameter_bind_enable,
        aer_sampler_shot_branching_enable=aer_sampler_shot_branching_enable,
        aer_sampler_shot_branching_sampling_enable=aer_sampler_shot_branching_sampling_enable,
        aer_sampler_runtime_parameter_bind_enable=aer_sampler_runtime_parameter_bind_enable,
        aer_noise_enabled=aer_noise_enabled,
        noise_single_qubit_depolarizing=noise_single_qubit_depolarizing,
        noise_two_qubit_depolarizing=noise_two_qubit_depolarizing,
        noise_readout_error=noise_readout_error,
        noise_include_thermal_relaxation=noise_include_thermal_relaxation,
        noise_t1_ns=noise_t1_ns,
        noise_t2_ns=noise_t2_ns,
        noise_one_qubit_gate_time_ns=noise_one_qubit_gate_time_ns,
        noise_two_qubit_gate_time_ns=noise_two_qubit_gate_time_ns,
        noise_seed_simulator=noise_seed_simulator,
        ibm_backend_name=ibm_backend_name,
        ibm_channel=ibm_channel,
        ibm_instance=ibm_instance,
        ibm_use_session=ibm_use_session,
        ibm_resilience_level=ibm_resilience_level,
        ibm_disable_dynamical_decoupling=ibm_disable_dynamical_decoupling,
        ibm_disable_twirling=ibm_disable_twirling,
        ibm_rep_delay=ibm_rep_delay,
        ibm_use_min_rep_delay=ibm_use_min_rep_delay,
        report_circuit_stats=report_circuit_stats,
        enable_inference_kv_cache=enable_inference_kv_cache,
        direct_primitive_inference=direct_primitive_inference,
        direct_primitive_batch_size=direct_primitive_batch_size,
        direct_primitive_fallback_to_qnn=direct_primitive_fallback_to_qnn_stage,
        direct_bind_inference_weights=direct_bind_inference_weights,
        direct_fuse_qkv_estimator=direct_fuse_qkv_estimator,
        causal_pair_pruning=causal_pair_pruning,
        ibm_execution_mode=ibm_execution_mode,
        ibm_parallel_jobs=ibm_parallel_jobs,
        ibm_parallel_min_parameter_rows=ibm_parallel_min_parameter_rows,
        ibm_parallel_min_shot_work=ibm_parallel_min_shot_work,
        ibm_parallel_max_rows_per_job=ibm_parallel_max_rows_per_job,
        ibm_deduplicate_parameter_rows=(
            WORKFLOW_STAGE == "infer_ibm" and ibm_deduplicate_parameter_rows
        ),
    )

    print("Workflow stage:", WORKFLOW_STAGE)
    print("Quantum backend type:", quantum_backend_type)
    print("Execution strategy:", quantum_execution_strategy)
    print("Local Aer noise enabled:", aer_noise_enabled)
    print("Quantum shots:", quantum_shots)
    print("Estimator precision:", estimator_precision)
    print("Inference KV cache:", enable_inference_kv_cache)
    print("Direct primitive inference:", direct_primitive_inference)
    print("Direct primitive fallback to QNN:", direct_primitive_fallback_to_qnn_stage)
    print("Direct primitive PUB batch size:", direct_primitive_batch_size)
    print("Bind inference quantum weights:", direct_bind_inference_weights)
    print("Fuse Q/K/V Estimator scheduling:", direct_fuse_qkv_estimator)
    print("Hardware causal fidelity-pair pruning:", causal_pair_pruning)
    print("Evaluation episodes:", eval_episodes)
    if quantum_backend_type == "aer":
        print("IBM Quantum service access: DISABLED (local Aer only)")
        if aer_noise_enabled:
            print("Aer precision:", aer_precision)
            print("Experimental dual-Aer executor:", aer_dual_executor)
            if aer_dual_executor:
                print("  Estimator backend - shot branching:", aer_estimator_shot_branching_enable)
                print("  Estimator backend - sampling branching:", aer_estimator_shot_branching_sampling_enable)
                print("  Estimator backend - runtime binding:", aer_estimator_runtime_parameter_bind_enable)
                print("  Sampler backend   - shot branching:", aer_sampler_shot_branching_enable)
                print("  Sampler backend   - sampling branching:", aer_sampler_shot_branching_sampling_enable)
                print("  Sampler backend   - runtime binding:", aer_sampler_runtime_parameter_bind_enable)
            else:
                print("Aer shot branching:", aer_shot_branching_enable)
                print("Aer sampling-with-branching:", aer_shot_branching_sampling_enable)
                print("Aer runtime parameter binding:", aer_runtime_parameter_bind_enable)
            print("Aer device:", aer_device)
            print("Aer GPU batched shots:", aer_batched_shots_gpu)
            print("Aer parallel experiments/shots:", aer_max_parallel_experiments, aer_max_parallel_shots)
    else:
        print("IBM Quantum service access: ENABLED (real-hardware stage)")
        print("IBM execution mode:", ibm_execution_mode)
        print("IBM Session-parallel child-job limit:", ibm_parallel_jobs)
        print("IBM parallel minimum parameter rows:", ibm_parallel_min_parameter_rows)
        print("IBM parallel minimum Sampler shot-work:", ibm_parallel_min_shot_work)
        print("IBM parallel maximum rows/job:", ibm_parallel_max_rows_per_job)
        print("IBM parameter-row deduplication:", ibm_deduplicate_parameter_rows)
        print("IBM resilience level:", ibm_resilience_level)
        print("IBM minimum rep_delay mode:", ibm_use_min_rep_delay)

    model = HybridQuantumGPT(mconf)
    if quantum_backend_type == "aer" and aer_noise_enabled and aer_dual_executor:
        qctx = model.quantum_context
        print("Dual-Aer backend objects are distinct:", qctx.estimator_backend is not qctx.sampler_backend)
        print("QuantumProjection executor: Estimator ->", type(qctx.estimator_backend).__name__)
        print("QuantumFidelity executor: Sampler ->", type(qctx.sampler_backend).__name__)

    # Load simulator-trained parameters into either hardware-style Aer or IBM mode.
    if load_checkpoint:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.validate_checkpoint_metadata(checkpoint.get("metadata", {}))
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        else:
            # Legacy state_dict-only checkpoints can still be loaded, but there is no
            # metadata guard against changing the logical architecture.
            model.load_state_dict(checkpoint, strict=True)


    graph_data = create_graph()


    tconf = TrainerConfig(
        max_epochs=epochs,
        batch_size=batch_size,
        learning_rate=6e-4,
        lr_decay=True,
        warmup_tokens=512 * 20,
        final_tokens=2 * len(train_dataset) * context_length * 3,
        num_workers=4,
        model_type=model_type,
        max_timestep=max(timesteps),
        force_cpu=force_cpu,
        use_data_parallel=False,
        parallel_episode_evaluation=(
            WORKFLOW_STAGE == "infer_ibm" and ibm_parallel_episode_evaluation
        ),
        parallel_episode_workers=ibm_parallel_episode_workers,
        parallel_env_start_method=ibm_parallel_env_start_method,
    )
    trainer = Trainer(model, train_dataset, None, tconf)

    # Compile the checkpoint-specific fixed-weight quantum circuits once, before
    # timing/entering autoregressive evaluation. This performs no QPU/Aer execution.
    if load_checkpoint and quantum_execution_strategy == "hardware" and direct_primitive_inference:
        raw_model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
        raw_model.prepare_quantum_inference()


    # =============================================================================
    # Train or evaluate
    # =============================================================================
    percnum = 20 * 1500
    N = eval_episodes

    if do_train:
        trainer.train()

        raw_model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
        torch.save(
            {
                "model_state_dict": raw_model.state_dict(),
                "metadata": raw_model.checkpoint_metadata(),
            },
            CHECKPOINT_PATH,
        )
        print(f"Saved optimized-Aer checkpoint to {CHECKPOINT_PATH}")
        print("Now evaluate the trained model in the current optimized-Aer mode.")

    raw_model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
    _session_started = False
    try:
        if quantum_backend_type == "ibm" and ibm_execution_mode in {"session", "session_parallel"}:
            session = raw_model.start_ibm_session()
            _session_started = session is not None
            if _session_started:
                print("IBM Session opened for the complete evaluation trajectory.")
                if ibm_execution_mode == "session_parallel":
                    print(
                        f"Session-parallel mode: up to {ibm_parallel_jobs} independent "
                        "Estimator/Sampler child jobs are submitted before waiting."
                    )

        if model_type != 'reward_conditioned':
            raise NotImplementedError("The 240-bus evaluation path is configured for reward_conditioned mode.")
        trainer.evaluate(graph_data, percnum, N)
    finally:
        if _session_started:
            raw_model.close_ibm_session()
            print("IBM Session closed.")


if __name__ == "__main__":
    # Required for multiprocessing start_method="spawn". Child OpenDSS workers
    # import this module without recursively constructing the quantum model or
    # launching IBM jobs.
    main()
