"""
Four-stage hybrid quantum transformer workflow.

Workflow
--------
1) backend_type='aer', execution_strategy='optimized', aer_noise_enabled=False
   Fast differentiable PyTorch statevector evaluation for training.
2) backend_type='aer', execution_strategy='hardware', aer_noise_enabled=False
   Hardware-style direct Estimator/Sampler primitive execution on ideal local Aer.
3) backend_type='aer', execution_strategy='hardware', aer_noise_enabled=True
   Hardware-style direct primitive execution on a fully local synthetic noisy
   Aer CPU backend. Automatic CPU shot parallelism is retained; benchmarked
   shot-branching/runtime-binding/GPU shortcuts remain optional and disabled by
   the production configuration. No IBM service is contacted.
4) backend_type='ibm', execution_strategy='hardware'
   Qiskit execution on a real IBM QPU for inference. This is the only stage that
   creates QiskitRuntimeService or accesses an IBM backend.

The logical circuits and TorchConnector quantum weights are identical across all
four modes. Optimized Aer changes only how the same circuit is evaluated. During
inference in stages 2-4, direct V2 primitive PUBs bypass TorchConnector/QNN forward
overhead while reading the exact same learned TorchConnector.weight tensors. The
inference path additionally supports fixed-weight binding/retranspilation, fused
Q/K/V Estimator scheduling, all-Z observable broadcasting, new-query-only
Sampler fidelity sweeps, and causal-pair pruning so masked future pairs are never
submitted in Stages 2-4. Stage 4 additionally supports one persistent IBM Session with concurrent
child-job submission for independent Q/K/V and fidelity chunks, plus exact
parameter-row deduplication across a batched set of evaluation episodes.
"""
from __future__ import annotations

import math
import hashlib
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error, thermal_relaxation_error
from qiskit_ibm_runtime.executor_estimator import Estimator
from qiskit_ibm_runtime.executor_sampler import Sampler
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN, SamplerQNN


# -----------------------------------------------------------------------------
# Execution context
# -----------------------------------------------------------------------------

@dataclass
class QuantumExecutionContext:
    backend_type: str                 # 'aer' or 'ibm'
    execution_strategy: str           # 'optimized' or 'hardware'
    backend: object
    estimator: object
    sampler: object
    pass_manager: object
    shots: int
    # Stage-3 can use two AerSimulator instances with different native
    # execution optimizations while preserving one shared model context.
    estimator_backend: object = None
    sampler_backend: object = None
    dual_aer_executor: bool = False
    aer_noise_enabled: bool = False
    noise_model: object = None
    report_circuit_stats: bool = False
    direct_primitive_inference: bool = True
    direct_primitive_batch_size: int = 4096
    direct_primitive_fallback_to_qnn: bool = True
    # Inference-only execution optimizations.  When weights are bound, Q/K/V
    # no longer share one identical parameterized circuit, so their vectorized
    # PUBs are fused into one Estimator.run(...) rather than one single PUB.
    direct_bind_inference_weights: bool = True
    direct_fuse_qkv_estimator: bool = True
    # Hardware-style fidelity optimization (Stages 2-4): never submit query-key
    # pairs that the causal mask would discard. Stage 1 keeps dense statevector
    # overlap because the SxS matmul is cheap and highly optimized.
    causal_pair_pruning: bool = True

    # Stage-4 high-shot execution controls. ``session_parallel`` keeps one
    # persistent IBM Session open for the complete iterative evaluation while
    # submitting independent Q/K/V or fidelity-chunk jobs before waiting on
    # their results. This gives Session exclusivity plus multi-job parallelism
    # without repeatedly switching between Batch and ordinary Job modes.
    ibm_execution_mode: str = "session_parallel"
    ibm_parallel_jobs: int = 5
    ibm_parallel_min_parameter_rows: int = 128
    # Sampler-specific Stage-4 trigger. Fidelity rows are split into concurrent
    # Session jobs when either the unique-row threshold OR unique_rows * shots
    # reaches this shot-work threshold. Set <=0 to disable the shot-work trigger.
    ibm_parallel_min_shot_work: int = 500_000
    ibm_parallel_max_rows_per_job: int = 2048
    ibm_deduplicate_parameter_rows: bool = False

    # Aer noisy-simulation execution controls. These are backend execution
    # optimizations only; they do not change the logical circuits/noise model.
    aer_precision: str = "double"
    aer_shot_branching_enable: bool = False
    aer_shot_branching_sampling_enable: bool = False
    aer_runtime_parameter_bind_enable: bool = False
    aer_batched_shots_gpu: bool = False
    aer_max_parallel_threads: int = 0
    aer_max_parallel_experiments: int = 1
    aer_max_parallel_shots: int = 0
    aer_num_threads_per_device: int = 1

    # Dual-Aer Stage-3 controls. These override the three native execution
    # toggles independently for the Estimator and Sampler backends. Shared
    # settings (method/device/precision/noise/threading) remain identical.
    aer_estimator_shot_branching_enable: bool = False
    aer_estimator_shot_branching_sampling_enable: bool = False
    aer_estimator_runtime_parameter_bind_enable: bool = False
    aer_sampler_shot_branching_enable: bool = False
    aer_sampler_shot_branching_sampling_enable: bool = False
    aer_sampler_runtime_parameter_bind_enable: bool = False

    # IBM inference controls. Fixed-weight circuit preparation/transpilation is
    # intentionally completed before any Session/Batch work is submitted.
    estimator_precision: Optional[float] = None
    # Backward-compatible mirror used by older callers. New code should set
    # ibm_execution_mode instead.
    ibm_use_session: bool = False
    ibm_resilience_level: int = 0
    ibm_disable_dynamical_decoupling: bool = True
    ibm_disable_twirling: bool = True
    ibm_rep_delay: Optional[float] = None
    ibm_session: object = None

    transpile_cache: dict = field(default_factory=dict)

    def _configure_runtime_primitives(self, estimator, sampler):
        """Apply the common inference budget and low-overhead IBM options."""
        sampler.options.default_shots = int(self.shots)
        if self.estimator_precision is not None:
            estimator.options.default_precision = float(self.estimator_precision)

        if self.backend_type != "ibm":
            return

        # Resilience 0 avoids implicit measurement mitigation/noise-learning
        # overhead. All settings remain configurable from the main file.
        try:
            estimator.options.resilience_level = int(self.ibm_resilience_level)
        except Exception as exc:
            warnings.warn(f"Could not set Estimator resilience_level: {exc}")

        if self.ibm_disable_dynamical_decoupling:
            for primitive in (estimator, sampler):
                try:
                    primitive.options.dynamical_decoupling.enable = False
                except Exception:
                    pass

        if self.ibm_disable_twirling:
            for primitive in (estimator, sampler):
                try:
                    primitive.options.twirling.enable_gates = False
                    primitive.options.twirling.enable_measure = False
                except Exception:
                    pass

        if self.ibm_rep_delay is not None:
            for primitive in (estimator, sampler):
                try:
                    primitive.options.execution.rep_delay = float(self.ibm_rep_delay)
                except Exception as exc:
                    warnings.warn(f"Could not set IBM rep_delay: {exc}")

    def _ibm_mode(self):
        mode = str(self.ibm_execution_mode).lower()
        if mode not in {"job", "session", "session_parallel", "batch"}:
            raise ValueError(
                "ibm_execution_mode must be 'job', 'session', 'session_parallel', or 'batch'"
            )
        return mode

    def parameter_chunks(self, n_rows: int, *, shots: Optional[int] = None):
        """Return row ranges for one direct-primitive parameter sweep.

        Stage-4 ``session_parallel`` splitting is adaptive:

        * Estimator sweeps call this with ``shots=None`` and remain row-driven.
        * Sampler/fidelity sweeps pass their actual shot count and are split when
          either the unique-row threshold OR total shot work (rows * shots) is
          large enough.

        Batch mode remains supported for backward compatibility, but the new
        Stage-4 architecture uses one persistent Session instead.
        """
        n_rows = int(n_rows)
        if n_rows <= 0:
            return []

        base = max(1, int(self.direct_primitive_batch_size))
        mode = self._ibm_mode()
        parallel_mode = (
            self.backend_type == "ibm"
            and mode in {"session_parallel", "batch"}
        )
        row_trigger = n_rows >= max(1, int(self.ibm_parallel_min_parameter_rows))

        shot_work = None
        shot_trigger = False
        min_shot_work = int(self.ibm_parallel_min_shot_work)
        if shots is not None:
            use_shots = max(1, int(shots))
            shot_work = n_rows * use_shots
            shot_trigger = min_shot_work > 0 and shot_work >= min_shot_work

        should_split = parallel_mode and (row_trigger or shot_trigger)
        if not should_split:
            return list(_chunks(n_rows, base))

        target_jobs = max(1, int(self.ibm_parallel_jobs))
        chunk_size = int(math.ceil(n_rows / float(target_jobs)))
        max_rows = int(self.ibm_parallel_max_rows_per_job)
        if max_rows > 0:
            chunk_size = min(chunk_size, max_rows)
        chunk_size = min(chunk_size, base)
        ranges = list(_chunks(n_rows, max(1, chunk_size)))

        trigger_parts = []
        if row_trigger:
            trigger_parts.append(
                f"rows={n_rows} >= {int(self.ibm_parallel_min_parameter_rows)}"
            )
        if shot_trigger:
            trigger_parts.append(
                f"shot_work={shot_work:,} >= {min_shot_work:,}"
            )
        trigger_text = " or ".join(trigger_parts)
        label = "Session-parallel" if mode == "session_parallel" else "Batch"
        print(
            f"IBM {label} split: {n_rows} rows -> {len(ranges)} child job(s) "
            f"({trigger_text})."
        )
        return ranges

    def _run_session_parallel_jobs(self, primitive, pubs, *, shots=None, label="primitive"):
        """Submit independent jobs in waves inside the already-open Session.

        Every job in a wave is submitted before any ``result()`` call. The wave
        size is capped by ``ibm_parallel_jobs`` (five is a sensible default for
        current IBM backends). QPU execution may still serialize, but Runtime
        classical preparation can overlap while Session keeps dedicated access.
        """
        if self.ibm_session is None:
            raise RuntimeError(
                "session_parallel execution requires start_ibm_session() before inference"
            )
        pubs = list(pubs)
        if not pubs:
            return []

        width = max(1, int(self.ibm_parallel_jobs))
        results = []
        total_jobs = 0
        for wave_start in range(0, len(pubs), width):
            wave = pubs[wave_start:wave_start + width]
            if shots is None:
                jobs = [primitive.run([pub]) for pub in wave]
            else:
                jobs = [primitive.run([pub], shots=int(shots)) for pub in wave]
            total_jobs += len(jobs)
            # Only after all jobs in this wave are submitted do we wait.
            for job in jobs:
                results.extend(job.result())

        print(
            f"IBM Session-parallel {label}: submitted {total_jobs} child job(s) "
            f"in waves of at most {width}."
        )
        return results

    def run_estimator_pubs(self, pubs):
        """Execute Estimator PUBs with Session-parallel submission in Stage 4."""
        pubs = list(pubs)
        if not pubs:
            return []

        mode = self._ibm_mode()
        if self.backend_type == "ibm" and mode == "session_parallel":
            if len(pubs) == 1:
                return self.estimator.run(pubs).result()
            return self._run_session_parallel_jobs(
                self.estimator, pubs, label="Estimator"
            )

        # Backward-compatible Batch path for older experiments.
        if self.backend_type == "ibm" and mode == "batch" and len(pubs) > 1:
            from qiskit_ibm_runtime import Batch
            jobs = []
            with Batch(backend=self.backend) as batch:
                estimator = Estimator(mode=batch)
                sampler = Sampler(mode=batch)
                self._configure_runtime_primitives(estimator, sampler)
                for pub in pubs:
                    jobs.append(estimator.run([pub]))
            results = []
            for job in jobs:
                results.extend(job.result())
            return results

        return self.estimator.run(pubs).result()

    def run_sampler_pubs(self, pubs, *, shots=None):
        """Execute Sampler PUBs with Session-parallel submission in Stage 4."""
        pubs = list(pubs)
        if not pubs:
            return []
        use_shots = int(self.shots if shots is None else shots)

        mode = self._ibm_mode()
        if self.backend_type == "ibm" and mode == "session_parallel":
            if len(pubs) == 1:
                return self.sampler.run(pubs, shots=use_shots).result()
            return self._run_session_parallel_jobs(
                self.sampler, pubs, shots=use_shots, label="Sampler"
            )

        # Backward-compatible Batch path for older experiments.
        if self.backend_type == "ibm" and mode == "batch" and len(pubs) > 1:
            from qiskit_ibm_runtime import Batch
            jobs = []
            with Batch(backend=self.backend) as batch:
                estimator = Estimator(mode=batch)
                sampler = Sampler(mode=batch)
                self._configure_runtime_primitives(estimator, sampler)
                for pub in pubs:
                    jobs.append(sampler.run([pub], shots=use_shots))
            results = []
            for job in jobs:
                results.extend(job.result())
            return results

        return self.sampler.run(pubs, shots=use_shots).result()

    def start_ibm_session(self):
        """Open one persistent Session after fixed-weight circuits are prepared."""
        if self.backend_type != "ibm" or self._ibm_mode() not in {"session", "session_parallel"}:
            return None
        if self.ibm_session is not None:
            return self.ibm_session
        from qiskit_ibm_runtime import Session

        self.ibm_session = Session(backend=self.backend)
        self.estimator = Estimator(mode=self.ibm_session)
        self.sampler = Sampler(mode=self.ibm_session)
        self.estimator_backend = self.backend
        self.sampler_backend = self.backend
        self._configure_runtime_primitives(self.estimator, self.sampler)
        return self.ibm_session

    def close_ibm_session(self):
        if self.ibm_session is not None:
            try:
                self.ibm_session.close()
            finally:
                self.ibm_session = None

    def set_inference_budget(self, *, shots=None, estimator_precision=None):
        """Change the sampling budget without rebuilding/transpiling circuits."""
        if shots is not None:
            self.shots = int(shots)
            self.sampler.options.default_shots = int(shots)
        if estimator_precision is not None:
            self.estimator_precision = float(estimator_precision)
            self.estimator.options.default_precision = float(estimator_precision)


IBMQuantumContext = QuantumExecutionContext


def build_local_aer_noise_model(
    single_qubit_depolarizing: float = 0.001,
    two_qubit_depolarizing: float = 0.01,
    readout_error: float = 0.02,
    include_thermal_relaxation: bool = False,
    t1_ns: float = 100_000.0,
    t2_ns: float = 80_000.0,
    one_qubit_gate_time_ns: float = 50.0,
    two_qubit_gate_time_ns: float = 300.0,
) -> NoiseModel:
    """Build a fully local, synthetic Aer noise model.

    No IBM service/backend is queried. Error parameters are supplied explicitly
    by the user, which guarantees that simulator-only stages remain offline with
    respect to IBM Quantum services.

    Noise channels:
      * 1-qubit depolarizing error on RY/RZ
      * 2-qubit depolarizing error on CX
      * symmetric readout bit-flip error
      * optional thermal relaxation on 1q/2q gates
    """
    p1 = float(single_qubit_depolarizing)
    p2 = float(two_qubit_depolarizing)
    pr = float(readout_error)
    for name, p in [("single_qubit_depolarizing", p1),
                    ("two_qubit_depolarizing", p2),
                    ("readout_error", pr)]:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"{name} must be in [0,1], got {p}")

    noise_model = NoiseModel()

    one_q_error = depolarizing_error(p1, 1) if p1 > 0.0 else None
    two_q_error = depolarizing_error(p2, 2) if p2 > 0.0 else None

    if include_thermal_relaxation:
        t1 = float(t1_ns)
        t2 = float(t2_ns)
        if t1 <= 0.0 or t2 <= 0.0:
            raise ValueError("T1 and T2 must be positive.")
        # Physical single-qubit relaxation channels require T2 <= 2*T1.
        if t2 > 2.0 * t1:
            raise ValueError("thermal_relaxation_error requires T2 <= 2*T1.")

        thermal_1q = thermal_relaxation_error(
            t1, t2, float(one_qubit_gate_time_ns)
        )
        # Independent thermal relaxation on the two qubits of a CX.
        thermal_2q = thermal_relaxation_error(
            t1, t2, float(two_qubit_gate_time_ns)
        ).tensor(
            thermal_relaxation_error(t1, t2, float(two_qubit_gate_time_ns))
        )

        one_q_error = thermal_1q if one_q_error is None else one_q_error.compose(thermal_1q)
        two_q_error = thermal_2q if two_q_error is None else two_q_error.compose(thermal_2q)

    if one_q_error is not None:
        noise_model.add_all_qubit_quantum_error(one_q_error, ["ry", "rz"])
    if two_q_error is not None:
        noise_model.add_all_qubit_quantum_error(two_q_error, ["cx"])

    if pr > 0.0:
        readout = ReadoutError([[1.0 - pr, pr], [pr, 1.0 - pr]])
        noise_model.add_all_qubit_readout_error(readout)

    return noise_model


def build_quantum_context(
    backend_type: str = "aer",
    execution_strategy: str = "optimized",
    backend_name: Optional[str] = None,
    min_num_qubits: int = 4,
    shots: int = 4096,
    optimization_level: int = 1,
    estimator_precision: Optional[float] = None,
    channel: Optional[str] = None,
    instance: Optional[str] = None,
    aer_method: Optional[str] = "statevector",
    aer_device: Optional[str] = None,
    aer_noise_enabled: bool = False,
    noise_single_qubit_depolarizing: float = 0.001,
    noise_two_qubit_depolarizing: float = 0.01,
    noise_readout_error: float = 0.02,
    noise_include_thermal_relaxation: bool = False,
    noise_t1_ns: float = 100_000.0,
    noise_t2_ns: float = 80_000.0,
    noise_one_qubit_gate_time_ns: float = 50.0,
    noise_two_qubit_gate_time_ns: float = 300.0,
    noise_seed_simulator: Optional[int] = None,
    report_circuit_stats: bool = False,
    direct_primitive_inference: bool = True,
    direct_primitive_batch_size: int = 4096,
    direct_primitive_fallback_to_qnn: bool = True,
    direct_bind_inference_weights: bool = True,
    direct_fuse_qkv_estimator: bool = True,
    causal_pair_pruning: bool = True,
    ibm_execution_mode: str = "session_parallel",
    ibm_parallel_jobs: int = 5,
    ibm_parallel_min_parameter_rows: int = 128,
    ibm_parallel_min_shot_work: int = 500_000,
    ibm_parallel_max_rows_per_job: int = 2048,
    ibm_deduplicate_parameter_rows: bool = False,
    aer_precision: str = "double",
    aer_shot_branching_enable: bool = False,
    aer_shot_branching_sampling_enable: bool = False,
    aer_runtime_parameter_bind_enable: bool = False,
    aer_batched_shots_gpu: bool = False,
    aer_max_parallel_threads: int = 0,
    aer_max_parallel_experiments: int = 1,
    aer_max_parallel_shots: int = 0,
    aer_num_threads_per_device: int = 1,
    aer_dual_executor: bool = False,
    aer_estimator_shot_branching_enable: bool = False,
    aer_estimator_shot_branching_sampling_enable: bool = False,
    aer_estimator_runtime_parameter_bind_enable: bool = False,
    aer_sampler_shot_branching_enable: bool = False,
    aer_sampler_shot_branching_sampling_enable: bool = False,
    aer_sampler_runtime_parameter_bind_enable: bool = False,
    ibm_use_session: bool = False,
    ibm_resilience_level: int = 0,
    ibm_disable_dynamical_decoupling: bool = True,
    ibm_disable_twirling: bool = True,
    ibm_rep_delay: Optional[float] = None,
    ibm_use_min_rep_delay: bool = False,
) -> QuantumExecutionContext:
    backend_type = str(backend_type).lower()
    execution_strategy = str(execution_strategy).lower()

    if backend_type not in {"aer", "ibm"}:
        raise ValueError("backend_type must be 'aer' or 'ibm'")
    if execution_strategy not in {"optimized", "hardware"}:
        raise ValueError("execution_strategy must be 'optimized' or 'hardware'")
    if backend_type == "ibm" and execution_strategy != "hardware":
        raise ValueError("IBM real hardware must use execution_strategy='hardware'.")
    if aer_noise_enabled and backend_type != "aer":
        raise ValueError("aer_noise_enabled is only valid for backend_type='aer'.")
    if aer_noise_enabled and execution_strategy != "hardware":
        raise ValueError(
            "Noisy validation must use execution_strategy='hardware'; "
            "optimized training remains exact/differentiable."
        )

    ibm_execution_mode = str(ibm_execution_mode).lower()
    if ibm_execution_mode not in {"job", "session", "session_parallel", "batch"}:
        raise ValueError(
            "ibm_execution_mode must be 'job', 'session', 'session_parallel', or 'batch'"
        )
    # Preserve compatibility with old callers that only supplied ibm_use_session.
    if backend_type == "ibm" and ibm_execution_mode == "session" and not ibm_use_session:
        # Explicit new-mode setting takes precedence. No action is needed here.
        pass

    local_noise_model = None

    if backend_type == "aer":
        # IMPORTANT: everything in this branch is local. No QiskitRuntimeService
        # object is created and no IBM backend/calibration data is requested.
        shared_aer_kwargs = {}
        if aer_method is not None:
            shared_aer_kwargs["method"] = aer_method
        if aer_device is not None:
            shared_aer_kwargs["device"] = aer_device

        shared_aer_kwargs["precision"] = str(aer_precision)
        shared_aer_kwargs["max_parallel_threads"] = int(aer_max_parallel_threads)
        shared_aer_kwargs["max_parallel_experiments"] = int(aer_max_parallel_experiments)
        shared_aer_kwargs["max_parallel_shots"] = int(aer_max_parallel_shots)
        shared_aer_kwargs["num_threads_per_device"] = int(aer_num_threads_per_device)
        if str(aer_device).upper() == "GPU":
            shared_aer_kwargs["batched_shots_gpu"] = bool(aer_batched_shots_gpu)

        if aer_noise_enabled:
            local_noise_model = build_local_aer_noise_model(
                single_qubit_depolarizing=noise_single_qubit_depolarizing,
                two_qubit_depolarizing=noise_two_qubit_depolarizing,
                readout_error=noise_readout_error,
                include_thermal_relaxation=noise_include_thermal_relaxation,
                t1_ns=noise_t1_ns,
                t2_ns=noise_t2_ns,
                one_qubit_gate_time_ns=noise_one_qubit_gate_time_ns,
                two_qubit_gate_time_ns=noise_two_qubit_gate_time_ns,
            )
            shared_aer_kwargs["noise_model"] = local_noise_model

        use_dual = bool(aer_dual_executor and aer_noise_enabled and execution_strategy == "hardware")
        if use_dual:
            # Empirical stability guard for the current Aer environment: each
            # native backend may use either branching or runtime binding, but not
            # both simultaneously. The two optimizations are split by primitive.
            if (
                aer_estimator_shot_branching_enable
                and aer_estimator_runtime_parameter_bind_enable
            ):
                raise ValueError(
                    "Dual-Aer Estimator backend cannot enable both shot branching "
                    "and runtime parameter binding in this environment."
                )
            if (
                aer_sampler_shot_branching_enable
                and aer_sampler_runtime_parameter_bind_enable
            ):
                raise ValueError(
                    "Dual-Aer Sampler backend cannot enable both shot branching "
                    "and runtime parameter binding in this environment."
                )
            if (
                aer_estimator_shot_branching_sampling_enable
                and not aer_estimator_shot_branching_enable
            ):
                raise ValueError(
                    "Estimator shot_branching_sampling_enable requires "
                    "shot_branching_enable=True."
                )
            if (
                aer_sampler_shot_branching_sampling_enable
                and not aer_sampler_shot_branching_enable
            ):
                raise ValueError(
                    "Sampler shot_branching_sampling_enable requires "
                    "shot_branching_enable=True."
                )
            # The combination shot_branching_enable=True +
            # runtime_parameter_bind_enable=True reproducibly triggers a native
            # Aer invalid-pointer abort in the user's environment. Split those
            # optimizations across primitive-specific simulators instead.
            estimator_kwargs = dict(shared_aer_kwargs)
            estimator_kwargs["runtime_parameter_bind_enable"] = bool(
                aer_estimator_runtime_parameter_bind_enable
            )
            estimator_kwargs["shot_branching_enable"] = bool(
                aer_estimator_shot_branching_enable
            )
            estimator_kwargs["shot_branching_sampling_enable"] = bool(
                aer_estimator_shot_branching_sampling_enable
            )

            sampler_kwargs = dict(shared_aer_kwargs)
            sampler_kwargs["runtime_parameter_bind_enable"] = bool(
                aer_sampler_runtime_parameter_bind_enable
            )
            sampler_kwargs["shot_branching_enable"] = bool(
                aer_sampler_shot_branching_enable
            )
            sampler_kwargs["shot_branching_sampling_enable"] = bool(
                aer_sampler_shot_branching_sampling_enable
            )

            estimator_backend = AerSimulator(**estimator_kwargs)
            sampler_backend = AerSimulator(**sampler_kwargs)
            backend = estimator_backend  # compatibility/default transpilation target
        else:
            aer_kwargs = dict(shared_aer_kwargs)
            aer_kwargs["runtime_parameter_bind_enable"] = bool(
                aer_runtime_parameter_bind_enable
            )
            aer_kwargs["shot_branching_enable"] = bool(aer_shot_branching_enable)
            aer_kwargs["shot_branching_sampling_enable"] = bool(
                aer_shot_branching_sampling_enable
            )
            backend = AerSimulator(**aer_kwargs)
            estimator_backend = backend
            sampler_backend = backend

        if noise_seed_simulator is not None:
            estimator_backend.set_options(seed_simulator=int(noise_seed_simulator))
            if sampler_backend is not estimator_backend:
                sampler_backend.set_options(seed_simulator=int(noise_seed_simulator))
    else:
        # Deliberately lazy import: this is the only branch that imports and
        # instantiates QiskitRuntimeService or accesses IBM Quantum services.
        from qiskit_ibm_runtime import QiskitRuntimeService

        service_kwargs = {}
        if channel is not None:
            service_kwargs["channel"] = channel
        if instance is not None:
            service_kwargs["instance"] = instance
        service = QiskitRuntimeService(**service_kwargs)

        # ------------------------------------------------------------------
        # Stage-4 IBM account diagnostics.  Never print API tokens/keys.
        # These messages make it explicit which saved/active account, service
        # instance, and physical QPU are being used for remote inference.
        # ------------------------------------------------------------------
        def _print_redacted_mapping(title, mapping):
            print(f"\n=== {title} ===")
            if not mapping:
                print("(none reported)")
                return
            for key, value in mapping.items():
                key_lower = str(key).lower()
                if any(secret_word in key_lower for secret_word in
                       ("token", "api_key", "apikey", "password", "secret")):
                    print(f"{key}: ***REDACTED***")
                else:
                    print(f"{key}: {value}")

        print("\n=== IBM Quantum service connection ===")
        print("Channel:", getattr(service, "channel", None))

        try:
            active_before = service.active_instance()
        except Exception as exc:
            active_before = f"<unavailable: {exc}>"
        print("Active instance before backend selection:", active_before)

        try:
            _print_redacted_mapping(
                "Active IBM Quantum account",
                service.active_account(),
            )
        except Exception as exc:
            print("\n=== Active IBM Quantum account ===")
            print(f"Unable to query active_account(): {exc}")

        # This is informational only.  If channel/instance were supplied
        # explicitly above, the active account can differ from the saved default.
        try:
            defaults = QiskitRuntimeService.saved_accounts(default=True)
            print("\n=== Saved default IBM Quantum account(s) ===")
            if not defaults:
                print("(no saved default account reported)")
            else:
                for saved_name, saved_info in defaults.items():
                    print(f"Account name: {saved_name}")
                    if isinstance(saved_info, dict):
                        for key, value in saved_info.items():
                            key_lower = str(key).lower()
                            if any(secret_word in key_lower for secret_word in
                                   ("token", "api_key", "apikey", "password", "secret")):
                                print(f"  {key}: ***REDACTED***")
                            else:
                                print(f"  {key}: {value}")
                    else:
                        print(f"  {saved_info}")
        except Exception as exc:
            print("\n=== Saved default IBM Quantum account(s) ===")
            print(f"Unable to query saved_accounts(default=True): {exc}")

        if backend_name is None:
            backend = service.least_busy(
                operational=True,
                simulator=False,
                min_num_qubits=min_num_qubits,
            )
        else:
            backend = service.backend(backend_name)
            if getattr(backend, "num_qubits", min_num_qubits) < min_num_qubits:
                raise ValueError(
                    f"Backend {backend_name!r} has {backend.num_qubits} qubits, "
                    f"but {min_num_qubits} are required."
                )

        backend_name_selected = getattr(backend, "name", None)
        if callable(backend_name_selected):
            backend_name_selected = backend_name_selected()

        try:
            active_after = service.active_instance()
        except Exception as exc:
            active_after = f"<unavailable: {exc}>"

        print("\n=== Selected IBM Quantum resource ===")
        print("Backend:", backend_name_selected)
        print("Backend qubits:", getattr(backend, "num_qubits", None))
        print("Active instance after backend selection:", active_after)
        print("IBM execution mode:", str(ibm_execution_mode).lower())
        print("IBM parallel child-job limit:", int(ibm_parallel_jobs))
        print("IBM parallel minimum parameter rows:", int(ibm_parallel_min_parameter_rows))
        print("IBM parallel minimum Sampler shot-work:", int(ibm_parallel_min_shot_work))
        print("IBM parallel maximum rows per child job:", int(ibm_parallel_max_rows_per_job))
        print("IBM cross-episode parameter deduplication:", bool(ibm_deduplicate_parameter_rows))
        print("Dynamic repetition rate enabled:", getattr(backend, "dynamic_reprate_enabled", None))
        print("Backend rep_delay range:", getattr(backend, "rep_delay_range", None))
        print("Backend default rep_delay:", getattr(backend, "default_rep_delay", None))

        if ibm_use_min_rep_delay and ibm_rep_delay is None:
            rep_range = getattr(backend, "rep_delay_range", None)
            dynamic_rate = bool(getattr(backend, "dynamic_reprate_enabled", False))
            if dynamic_rate and rep_range is not None and len(rep_range) >= 1:
                ibm_rep_delay = float(rep_range[0])
                print("Using minimum supported IBM rep_delay:", ibm_rep_delay)
            else:
                warnings.warn(
                    "ibm_use_min_rep_delay=True, but the selected backend does not "
                    "report a dynamic repetition-rate range; backend default retained."
                )

    if backend_type == "ibm":
        estimator_backend = backend
        sampler_backend = backend

    # Build initial primitives in backend/job mode. For IBM Session/session-parallel inference
    # these are replaced only *after* checkpoint-specific circuit preparation,
    # so transpilation does not consume dedicated Session time.
    estimator = Estimator(mode=estimator_backend)
    sampler = Sampler(mode=sampler_backend)

    # This is constructed once. EstimatorQNN/SamplerQNN use it during QNN setup;
    # there is no transpilation in forward().
    pass_manager = generate_preset_pass_manager(
        backend=backend,
        optimization_level=optimization_level,
    )

    context = QuantumExecutionContext(
        backend_type=backend_type,
        execution_strategy=execution_strategy,
        backend=backend,
        estimator=estimator,
        sampler=sampler,
        pass_manager=pass_manager,
        estimator_backend=estimator_backend,
        sampler_backend=sampler_backend,
        dual_aer_executor=bool(
            backend_type == "aer"
            and aer_noise_enabled
            and execution_strategy == "hardware"
            and aer_dual_executor
        ),
        shots=int(shots),
        aer_noise_enabled=bool(aer_noise_enabled),
        noise_model=local_noise_model,
        report_circuit_stats=bool(report_circuit_stats),
        direct_primitive_inference=bool(direct_primitive_inference),
        direct_primitive_batch_size=max(1, int(direct_primitive_batch_size)),
        direct_primitive_fallback_to_qnn=bool(direct_primitive_fallback_to_qnn),
        direct_bind_inference_weights=bool(direct_bind_inference_weights),
        direct_fuse_qkv_estimator=bool(direct_fuse_qkv_estimator),
        causal_pair_pruning=bool(causal_pair_pruning),
        ibm_execution_mode=str(ibm_execution_mode).lower(),
        ibm_parallel_jobs=max(1, int(ibm_parallel_jobs)),
        ibm_parallel_min_parameter_rows=max(1, int(ibm_parallel_min_parameter_rows)),
        ibm_parallel_min_shot_work=int(ibm_parallel_min_shot_work),
        ibm_parallel_max_rows_per_job=max(0, int(ibm_parallel_max_rows_per_job)),
        ibm_deduplicate_parameter_rows=bool(ibm_deduplicate_parameter_rows),
        aer_precision=str(aer_precision),
        aer_shot_branching_enable=bool(aer_shot_branching_enable),
        aer_shot_branching_sampling_enable=bool(aer_shot_branching_sampling_enable),
        aer_runtime_parameter_bind_enable=bool(aer_runtime_parameter_bind_enable),
        aer_batched_shots_gpu=bool(aer_batched_shots_gpu),
        aer_max_parallel_threads=int(aer_max_parallel_threads),
        aer_max_parallel_experiments=int(aer_max_parallel_experiments),
        aer_max_parallel_shots=int(aer_max_parallel_shots),
        aer_num_threads_per_device=int(aer_num_threads_per_device),
        aer_estimator_shot_branching_enable=bool(aer_estimator_shot_branching_enable),
        aer_estimator_shot_branching_sampling_enable=bool(aer_estimator_shot_branching_sampling_enable),
        aer_estimator_runtime_parameter_bind_enable=bool(aer_estimator_runtime_parameter_bind_enable),
        aer_sampler_shot_branching_enable=bool(aer_sampler_shot_branching_enable),
        aer_sampler_shot_branching_sampling_enable=bool(aer_sampler_shot_branching_sampling_enable),
        aer_sampler_runtime_parameter_bind_enable=bool(aer_sampler_runtime_parameter_bind_enable),
        estimator_precision=(None if estimator_precision is None else float(estimator_precision)),
        ibm_use_session=(str(ibm_execution_mode).lower() in {"session", "session_parallel"}),
        ibm_resilience_level=int(ibm_resilience_level),
        ibm_disable_dynamical_decoupling=bool(ibm_disable_dynamical_decoupling),
        ibm_disable_twirling=bool(ibm_disable_twirling),
        ibm_rep_delay=(None if ibm_rep_delay is None else float(ibm_rep_delay)),
    )
    context._configure_runtime_primitives(context.estimator, context.sampler)
    return context


def build_ibm_quantum_context(**kwargs) -> QuantumExecutionContext:
    kwargs.pop("backend_type", None)
    kwargs.pop("execution_strategy", None)
    return build_quantum_context(
        backend_type="ibm",
        execution_strategy="hardware",
        **kwargs,
    )


# -----------------------------------------------------------------------------
# Circuit helpers and metrics
# -----------------------------------------------------------------------------

_PRINTED_STATS_KEYS = set()
_DIRECT_FALLBACK_WARNED = set()
_PAIR_PRUNING_PRINTED = set()


def _append_hardware_ansatz(
    qc: QuantumCircuit,
    input_params,
    weight_params,
    n_qubits: int,
    n_q_layers: int,
) -> None:
    # RY angle encoding
    for q in range(n_qubits):
        qc.ry(input_params[q], q)

    cursor = 0
    for layer in range(n_q_layers):
        # Same trainable RZ-RY-RZ sequence used in both simulator and hardware.
        for q in range(n_qubits):
            qc.rz(weight_params[cursor + 0], q)
            qc.ry(weight_params[cursor + 1], q)
            qc.rz(weight_params[cursor + 2], q)
            cursor += 3

        if n_qubits == 1:
            continue

        distance = 1 if n_qubits == 2 else 1 + (layer % (n_qubits - 1))
        for q in range(n_qubits):
            target = (q + distance) % n_qubits
            if target != q:
                qc.cx(q, target)


def _z_observables(n_qubits: int):
    return [
        SparsePauliOp.from_sparse_list([("Z", [q], 1.0)], num_qubits=n_qubits)
        for q in range(n_qubits)
    ]


def _resolve_params_by_name(circuit: QuantumCircuit, params):
    """Map logical Parameter objects to equivalent parameters in a cached circuit.

    This allows one cached/transpiled topology to be reused by many transformer
    layers even though each layer originally constructed distinct Parameter objects.
    """
    by_name = {p.name: p for p in circuit.parameters}
    missing = [p.name for p in params if p.name not in by_name]
    if missing:
        raise ValueError(f"Cached circuit is missing parameters: {missing}")
    return [by_name[p.name] for p in params]


def circuit_metrics(circuit: QuantumCircuit) -> dict:
    """Return logical/ISA circuit metrics.

    The requested two-qubit depth metric is exactly:
        circuit.depth(lambda x: len(x.qubits) == 2)
    """
    return {
        "num_qubits": circuit.num_qubits,
        "transpiler_layout": circuit.layout,
        "num_classical_bits": circuit.num_clbits,
        "size": circuit.size(),
        "depth": circuit.depth(),
        "two_qubit_depth": circuit.depth(lambda x: len(x.qubits) == 2),
        "nonlocal_gates": circuit.num_nonlocal_gates(),
        "count_ops": dict(circuit.count_ops()),
    }


def print_circuit_metrics(circuit: QuantumCircuit, name: str) -> None:
    m = circuit_metrics(circuit)
    print(f"\n=== {name} ===")
    print("num_qubits       :", m["num_qubits"])
    print("transpiler_layout:", m["transpiler_layout"])
    print("num_classical_bits:", m["num_classical_bits"])
    print("total operations :", m["size"])
    print("total depth      :", m["depth"])
    print("two-qubit depth  :", m["two_qubit_depth"])
    print("nonlocal gates   :", m["nonlocal_gates"])
    print("operation counts :", m["count_ops"])


def _cached_transpile(
    context: QuantumExecutionContext,
    logical_circuit: QuantumCircuit,
    cache_key: tuple,
) -> QuantumCircuit:
    """Transpile one parameterized topology once per execution context.

    The numerical input/weight values remain symbolic, so the returned circuit
    can be reused for every batch and autoregressive inference step.
    """
    if context.execution_strategy != "hardware":
        return logical_circuit
    if cache_key not in context.transpile_cache:
        context.transpile_cache[cache_key] = context.pass_manager.run(logical_circuit)
    return context.transpile_cache[cache_key]


def _maybe_report_circuit(
    context: QuantumExecutionContext,
    logical_circuit: QuantumCircuit,
    key: tuple,
    name: str,
) -> None:
    if not context.report_circuit_stats or key in _PRINTED_STATS_KEYS:
        return
    _PRINTED_STATS_KEYS.add(key)

    print_circuit_metrics(logical_circuit, f"{name} - logical")
    if context.execution_strategy == "hardware":
        isa = _cached_transpile(context, logical_circuit, key)
        if context.backend_type == "ibm":
            suffix = "transpiled/IBM-ISA"
        elif context.aer_noise_enabled:
            suffix = "transpiled/local-noisy-Aer"
        else:
            suffix = "transpiled/local-ideal-Aer"
        print_circuit_metrics(isa, f"{name} - {suffix}")

# -----------------------------------------------------------------------------
# Direct primitive inference helpers
# -----------------------------------------------------------------------------

def _parameter_matrix(circuit: QuantumCircuit, named_columns: dict[str, np.ndarray]) -> np.ndarray:
    """Create parameter rows in *the circuit's actual parameter order*.

    ``named_columns`` maps each Qiskit Parameter.name to either a scalar or a
    length-N vector. This avoids relying on lexical ParameterVector ordering after
    transpilation and is therefore safe for cached ISA circuits.
    """
    params = list(circuit.parameters)
    if not params:
        return np.empty((1, 0), dtype=float)

    n_rows = None
    cols = []
    for p in params:
        if p.name not in named_columns:
            raise ValueError(f"No direct-primitive value supplied for parameter {p.name!r}")
        arr = np.asarray(named_columns[p.name], dtype=float)
        if arr.ndim == 0:
            cols.append(arr)
            continue
        arr = arr.reshape(-1)
        if n_rows is None:
            n_rows = arr.size
        elif arr.size != n_rows:
            raise ValueError("Direct primitive parameter columns have inconsistent lengths")
        cols.append(arr)

    n_rows = 1 if n_rows is None else n_rows
    out = np.empty((n_rows, len(params)), dtype=float)
    for j, arr in enumerate(cols):
        if np.ndim(arr) == 0:
            out[:, j] = float(arr)
        else:
            out[:, j] = arr
    return out


def _estimator_evs(pub_result) -> np.ndarray:
    """Extract expectation values from V2 primitive or executor-like results."""
    data = getattr(pub_result, "data", pub_result)
    if hasattr(data, "evs"):
        return np.asarray(data.evs, dtype=float)
    try:
        if "evs" in data:
            return np.asarray(data["evs"], dtype=float)
    except Exception:
        pass
    try:
        return np.asarray(pub_result["evs"], dtype=float)
    except Exception as exc:
        raise TypeError("Could not extract Estimator expectation values from primitive result") from exc


def _sampler_zero_probabilities(pub_result, n_rows: int, n_qubits: int) -> np.ndarray:
    """Return P(00...0) for every parameter row from a Sampler V2 result."""
    data = getattr(pub_result, "data", pub_result)
    bitdata = None
    if hasattr(data, "meas"):
        bitdata = data.meas
    else:
        try:
            if "meas" in data:
                bitdata = data["meas"]
        except Exception:
            pass
    if bitdata is None:
        try:
            bitdata = pub_result["meas"]
        except Exception as exc:
            raise TypeError("Could not find Sampler 'meas' register in primitive result") from exc

    # Fast SamplerV2 BitArray path.  The raw array stores packed bytes with
    # shape [parameter_shape..., shots, packed_bytes]. P(00...0) is therefore
    # simply the fraction of shots whose packed bytes are all zero. This avoids
    # constructing a Python counts dictionary for every parameter row.
    if hasattr(bitdata, "array"):
        raw = np.asarray(bitdata.array)
        # For our one-dimensional parameter sweep, the leading PUB shape is N.
        if raw.ndim >= 3 and raw.shape[0] == n_rows:
            zero_shots = np.all(raw == 0, axis=-1)
            return zero_shots.mean(axis=-1).astype(float).reshape(n_rows)

    # Compatibility path for BitArray-like implementations that expose counts
    # but not the packed raw array.
    if hasattr(bitdata, "get_counts"):
        out = np.empty(n_rows, dtype=float)
        zero_key = "0" * n_qubits
        for i in range(n_rows):
            try:
                counts = bitdata.get_counts(i)
            except TypeError:
                counts = bitdata.get_counts() if n_rows == 1 else None
            if counts is None:
                raise RuntimeError("Sampler BitArray did not preserve parameter-row shape")
            total = sum(counts.values())
            out[i] = 0.0 if total == 0 else counts.get(zero_key, 0) / total
        return out

    # Executor-style path: ndarray shaped approximately
    # [parameter_sets, shots, register_bits].
    arr = np.asarray(bitdata)
    if arr.ndim == 2 and n_rows == 1:
        arr = arr[None, ...]
    if arr.shape[0] != n_rows:
        raise RuntimeError(
            f"Unexpected direct Sampler result shape {arr.shape}; expected first axis {n_rows}"
        )
    zero = np.all(arr == 0, axis=-1)
    return zero.mean(axis=-1).astype(float)


def _chunks(n: int, batch_size: int):
    for start in range(0, n, batch_size):
        yield start, min(n, start + batch_size)


def _observable_outer_array(observables):
    """Shape observables as [n_obs, 1] for an outer broadcast vs parameter rows.

    A parameter sweep with N rows has broadcast shape [N].  Reshaping the
    observable list to [n_obs, 1] asks EstimatorV2 for all n_obs x N
    expectation values in one PUB instead of one PUB per observable.
    """
    arr = np.empty((len(observables), 1), dtype=object)
    for i, obs in enumerate(observables):
        arr[i, 0] = obs
    return arr


def _estimator_matrix(pub_result, n_rows: int, n_obs: int) -> np.ndarray:
    """Normalize EstimatorV2 broadcast output to [n_rows, n_obs]."""
    evs = np.asarray(_estimator_evs(pub_result), dtype=float)
    evs = np.squeeze(evs)
    if evs.ndim == 0 and n_rows == 1 and n_obs == 1:
        return evs.reshape(1, 1)
    if evs.shape == (n_obs, n_rows):
        return evs.T
    if evs.shape == (n_rows, n_obs):
        return evs
    if n_rows == 1 and evs.size == n_obs:
        return evs.reshape(1, n_obs)
    if n_obs == 1 and evs.size == n_rows:
        return evs.reshape(n_rows, 1)
    if evs.size == n_rows * n_obs:
        # EstimatorV2 outer-broadcast convention is normally [n_obs, n_rows].
        return evs.reshape(n_obs, n_rows).T
    raise RuntimeError(
        f"Unexpected Estimator result shape {evs.shape}; expected "
        f"{n_rows} parameter rows x {n_obs} observables"
    )


def _weight_signature(weights: np.ndarray) -> str:
    """Stable fingerprint used to invalidate lazily bound inference circuits."""
    a = np.ascontiguousarray(np.asarray(weights, dtype=np.float64).reshape(-1))
    return hashlib.sha1(a.view(np.uint8)).hexdigest()

# -----------------------------------------------------------------------------
# Fast differentiable small-qubit statevector evaluator for optimized Aer mode
# -----------------------------------------------------------------------------


def _complex_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.complex128 if dtype == torch.float64 else torch.complex64


def _initial_state(batch: int, n_qubits: int, device, real_dtype) -> torch.Tensor:
    state = torch.zeros(
        batch,
        2 ** n_qubits,
        device=device,
        dtype=_complex_dtype(real_dtype),
    )
    state[:, 0] = 1.0 + 0.0j
    return state


def _index_pairs(n_qubits: int, q: int, device):
    dim = 2 ** n_qubits
    idx = torch.arange(dim, device=device, dtype=torch.long)
    idx0 = idx[(idx & (1 << q)) == 0]
    idx1 = idx0 | (1 << q)
    return idx0, idx1


def _apply_ry(state: torch.Tensor, angles: torch.Tensor, q: int, n_qubits: int):
    idx0, idx1 = _index_pairs(n_qubits, q, state.device)
    a = state.index_select(-1, idx0)
    b = state.index_select(-1, idx1)
    c = torch.cos(angles / 2).to(state.dtype).unsqueeze(-1)
    s = torch.sin(angles / 2).to(state.dtype).unsqueeze(-1)
    out = state.clone()
    out[..., idx0] = c * a - s * b
    out[..., idx1] = s * a + c * b
    return out


def _apply_rz(state: torch.Tensor, angles: torch.Tensor, q: int, n_qubits: int):
    idx0, idx1 = _index_pairs(n_qubits, q, state.device)
    a = state.index_select(-1, idx0)
    b = state.index_select(-1, idx1)
    half = angles / 2
    p0 = torch.complex(torch.cos(half), -torch.sin(half)).to(state.dtype).unsqueeze(-1)
    p1 = torch.complex(torch.cos(half),  torch.sin(half)).to(state.dtype).unsqueeze(-1)
    out = state.clone()
    out[..., idx0] = p0 * a
    out[..., idx1] = p1 * b
    return out


def _apply_cx(state: torch.Tensor, control: int, target: int, n_qubits: int):
    dim = 2 ** n_qubits
    idx = torch.arange(dim, device=state.device, dtype=torch.long)
    control_on = (idx & (1 << control)) != 0
    perm = torch.where(control_on, idx ^ (1 << target), idx)
    # CX is its own inverse, so this permutation gives the output amplitudes.
    return state.index_select(-1, perm)


def _torch_ansatz_state(
    inputs: torch.Tensor,       # [N, n_qubits]
    weights: torch.Tensor,      # [n_layers*n_qubits*3]
    n_qubits: int,
    n_q_layers: int,
) -> torch.Tensor:
    if inputs.dim() != 2 or inputs.size(-1) != n_qubits:
        raise ValueError("inputs must have shape [N, n_qubits]")

    state = _initial_state(inputs.shape[0], n_qubits, inputs.device, inputs.dtype)

    for q in range(n_qubits):
        state = _apply_ry(state, inputs[:, q], q, n_qubits)

    cursor = 0
    for layer in range(n_q_layers):
        for q in range(n_qubits):
            # weights are shared across all samples in the batch.
            w0 = weights[cursor + 0].expand(inputs.shape[0])
            w1 = weights[cursor + 1].expand(inputs.shape[0])
            w2 = weights[cursor + 2].expand(inputs.shape[0])
            state = _apply_rz(state, w0, q, n_qubits)
            state = _apply_ry(state, w1, q, n_qubits)
            state = _apply_rz(state, w2, q, n_qubits)
            cursor += 3

        if n_qubits == 1:
            continue
        distance = 1 if n_qubits == 2 else 1 + (layer % (n_qubits - 1))
        for q in range(n_qubits):
            target = (q + distance) % n_qubits
            if target != q:
                state = _apply_cx(state, q, target, n_qubits)

    return state


def _z_expectations(state: torch.Tensor, n_qubits: int) -> torch.Tensor:
    probs = state.real.square() + state.imag.square()
    dim = 2 ** n_qubits
    idx = torch.arange(dim, device=state.device, dtype=torch.long)
    values = []
    for q in range(n_qubits):
        sign = torch.where(
            (idx & (1 << q)) == 0,
            torch.ones(dim, device=state.device, dtype=probs.dtype),
            -torch.ones(dim, device=state.device, dtype=probs.dtype),
        )
        values.append((probs * sign).sum(dim=-1))
    return torch.stack(values, dim=-1)


# -----------------------------------------------------------------------------
# Q/K/V projection
# -----------------------------------------------------------------------------

class QuantumProjection(nn.Module):
    """Q/K/V quantum projection with optimized inference execution.

    Training keeps the original TorchConnector weights.  Hardware-style
    inference can lazily bind those learned weights, retranspile once, and then
    vectorize only the data angles.  All Z observables are submitted in one
    Estimator PUB per circuit/chunk.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        quantum_context: QuantumExecutionContext,
        n_qubits: int = 4,
        n_q_layers: int = 2,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_qubits = n_qubits
        self.n_q_layers = n_q_layers
        self.quantum_context = quantum_context

        hidden_dim = max(in_dim, n_qubits * 2)
        self.pre_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_qubits),
        )

        x_params = ParameterVector("x", n_qubits)
        theta = ParameterVector("theta", n_q_layers * n_qubits * 3)
        qc = QuantumCircuit(n_qubits)
        _append_hardware_ansatz(qc, x_params, theta, n_qubits, n_q_layers)
        self.logical_circuit = qc
        self._logical_x_params = list(x_params)
        self._logical_theta_params = list(theta)

        _maybe_report_circuit(
            quantum_context,
            qc,
            key=("projection", quantum_context.backend_type, quantum_context.execution_strategy,
                 n_qubits, n_q_layers),
            name="QuantumProjection",
        )

        qnn_circuit = qc
        qnn_observables = _z_observables(n_qubits)
        if quantum_context.execution_strategy == "hardware":
            tkey = ("projection-isa", quantum_context.backend_type,
                    quantum_context.aer_noise_enabled, n_qubits, n_q_layers)
            qnn_circuit = _cached_transpile(quantum_context, qc, tkey)
            layout = getattr(qnn_circuit, "layout", None)
            if layout is not None:
                qnn_observables = [obs.apply_layout(layout) for obs in qnn_observables]

        qnn_input_params = _resolve_params_by_name(qnn_circuit, self._logical_x_params)
        qnn_weight_params = _resolve_params_by_name(qnn_circuit, self._logical_theta_params)

        # Generic direct-primitive circuit (weights remain dynamic).  This is
        # also the path used if inference weight binding is disabled.
        self._direct_circuit = qnn_circuit
        self._direct_observables = qnn_observables
        self._direct_x_names = [p.name for p in qnn_input_params]
        self._direct_theta_names = [p.name for p in qnn_weight_params]

        # Lazily compiled fixed-weight inference circuit.  It is created only
        # after the checkpoint has been loaded and model.eval() starts.
        self._bound_signature = None
        self._bound_direct_circuit = None
        self._bound_direct_observables = None
        self._bound_direct_x_names = None

        qnn = EstimatorQNN(
            circuit=qnn_circuit,
            observables=qnn_observables,
            input_params=qnn_input_params,
            weight_params=qnn_weight_params,
            estimator=quantum_context.estimator,
            input_gradients=True,
            pass_manager=None,
        )
        initial_weights = 0.01 * np.random.randn(len(theta))
        self.q_layer = TorchConnector(qnn, initial_weights=initial_weights)

        self.post_net = nn.Sequential(
            nn.Linear(n_qubits, out_dim),
            nn.GELU(),
        )

    @property
    def _optimized_aer(self) -> bool:
        return (
            self.quantum_context.backend_type == "aer"
            and self.quantum_context.execution_strategy == "optimized"
        )

    @property
    def _use_direct_primitive(self) -> bool:
        return (
            (not self.training)
            and self.quantum_context.execution_strategy == "hardware"
            and self.quantum_context.direct_primitive_inference
        )

    def _prepare_angles(self, x: torch.Tensor):
        batch_size, seq_len, _ = x.shape
        angles = torch.tanh(self.pre_net(x)) * (math.pi / 2.0)
        return angles.reshape(batch_size * seq_len, self.n_qubits), batch_size, seq_len

    def _finish_projection(self, z: torch.Tensor, batch_size: int, seq_len: int):
        z = z.to(
            device=next(self.post_net.parameters()).device,
            dtype=next(self.post_net.parameters()).dtype,
        )
        z = self.post_net(z)
        return z.reshape(batch_size, seq_len, self.out_dim)

    def _ensure_bound_direct_circuit(self):
        """Bind trained theta, retranspile once, and cache the inference circuit."""
        weights_np = self.q_layer.weight.detach().cpu().numpy().reshape(-1)
        signature = _weight_signature(weights_np)
        if self._bound_direct_circuit is not None and signature == self._bound_signature:
            return (
                self._bound_direct_circuit,
                self._bound_direct_observables,
                self._bound_direct_x_names,
            )

        mapping = {
            p: float(weights_np[i])
            for i, p in enumerate(self._logical_theta_params)
        }
        bound = self.logical_circuit.assign_parameters(mapping, inplace=False)
        if self.quantum_context.execution_strategy == "hardware":
            bound = self.quantum_context.pass_manager.run(bound)

        observables = _z_observables(self.n_qubits)
        layout = getattr(bound, "layout", None)
        if layout is not None:
            observables = [obs.apply_layout(layout) for obs in observables]
        x_names = [p.name for p in _resolve_params_by_name(bound, self._logical_x_params)]

        self._bound_signature = signature
        self._bound_direct_circuit = bound
        self._bound_direct_observables = observables
        self._bound_direct_x_names = x_names

        stats_key = (
            "projection-bound-representative",
            self.quantum_context.backend_type,
            self.quantum_context.aer_noise_enabled,
            self.n_qubits,
            self.n_q_layers,
        )
        if self.quantum_context.report_circuit_stats and stats_key not in _PRINTED_STATS_KEYS:
            _PRINTED_STATS_KEYS.add(stats_key)
            print_circuit_metrics(bound, "QuantumProjection - fixed-weight inference ISA (representative)")

        return bound, observables, x_names

    def _direct_components(self):
        if self.quantum_context.direct_bind_inference_weights:
            circuit, observables, x_names = self._ensure_bound_direct_circuit()
            return circuit, observables, x_names, []
        return (
            self._direct_circuit,
            self._direct_observables,
            self._direct_x_names,
            self._direct_theta_names,
        )

    def _parameter_rows(self, angles_np, circuit, x_names, theta_names):
        named = {name: angles_np[:, j] for j, name in enumerate(x_names)}
        if theta_names:
            weights_np = self.q_layer.weight.detach().cpu().numpy().reshape(-1)
            for j, name in enumerate(theta_names):
                named[name] = weights_np[j]
        return _parameter_matrix(circuit, named)

    def _direct_projection(self, angles: torch.Tensor) -> torch.Tensor:
        """Direct Estimator inference with all Zi grouped in each vectorized PUB."""
        angles_np = angles.detach().cpu().numpy()
        n_original = angles_np.shape[0]
        inverse = None
        if self.quantum_context.ibm_deduplicate_parameter_rows and n_original > 1:
            unique_angles, inverse = np.unique(angles_np, axis=0, return_inverse=True)
            if unique_angles.shape[0] < n_original:
                print(
                    f"QuantumProjection parameter deduplication: {n_original} -> "
                    f"{unique_angles.shape[0]} unique rows"
                )
            angles_np = unique_angles
        n = angles_np.shape[0]
        circuit, observables, x_names, theta_names = self._direct_components()
        obs_outer = _observable_outer_array(observables)

        pubs = []
        pub_meta = []
        for start, stop in self.quantum_context.parameter_chunks(n):
            pvals = self._parameter_rows(
                angles_np[start:stop], circuit, x_names, theta_names
            )
            pubs.append((circuit, obs_outer, pvals))
            pub_meta.append((start, stop))

        results = self.quantum_context.run_estimator_pubs(pubs)
        out = np.empty((n, self.n_qubits), dtype=np.float32)
        for pub_result, (start, stop) in zip(results, pub_meta):
            out[start:stop] = _estimator_matrix(
                pub_result, stop - start, self.n_qubits
            )
        if inverse is not None:
            out = out[inverse]
        return torch.from_numpy(out).to(device=angles.device, dtype=angles.dtype)

    @staticmethod
    def forward_fused_qkv(projections, inputs: torch.Tensor):
        """Execute Q/K/V projections in one Estimator.run(...) call.

        With fixed-weight binding enabled, Q/K/V have different bound circuits,
        so each projection contributes vectorized PUB(s), but all PUBs are sent in
        one primitive run.  With binding disabled, all Q/K/V rows share the same
        parameterized circuit and are collapsed into one true parameter sweep.
        """
        if len(projections) != 3:
            raise ValueError("forward_fused_qkv expects [query, key, value] projections")
        qproj, kproj, vproj = projections
        context = qproj.quantum_context
        if not all(p.quantum_context is context for p in projections):
            raise ValueError("Fused Q/K/V projections must share one QuantumExecutionContext")
        if not all(p._use_direct_primitive for p in projections):
            raise RuntimeError("Fused Q/K/V direct projection is inference-only")

        prepared = [p._prepare_angles(inputs) for p in projections]
        angles_list = [x[0] for x in prepared]
        batch_size, seq_len = prepared[0][1], prepared[0][2]
        if any((x[1], x[2]) != (batch_size, seq_len) for x in prepared):
            raise RuntimeError("Q/K/V projection token shapes do not match")
        n_each = angles_list[0].shape[0]

        if context.direct_bind_inference_weights:
            # Fixed weights enable smaller runtime parameter vectors and an extra
            # transpilation/optimization pass. Q/K/V circuits differ after binding.
            # Deduplicate equal episode rows within each projection before sending
            # them to hardware, then scatter results back to the original batch.
            pubs = []
            meta = []
            inverse_maps = []
            unique_counts = []
            for module_idx, (proj, angles) in enumerate(zip(projections, angles_list)):
                angles_np = angles.detach().cpu().numpy()
                inverse = None
                if context.ibm_deduplicate_parameter_rows and angles_np.shape[0] > 1:
                    unique_angles, inverse = np.unique(angles_np, axis=0, return_inverse=True)
                    if unique_angles.shape[0] < angles_np.shape[0]:
                        print(
                            f"QKV projection[{module_idx}] parameter deduplication: "
                            f"{angles_np.shape[0]} -> {unique_angles.shape[0]} unique rows"
                        )
                    angles_np = unique_angles
                inverse_maps.append(inverse)
                unique_counts.append(angles_np.shape[0])
                circuit, observables, x_names, theta_names = proj._direct_components()
                obs_outer = _observable_outer_array(observables)
                for start, stop in context.parameter_chunks(angles_np.shape[0]):
                    pvals = proj._parameter_rows(
                        angles_np[start:stop], circuit, x_names, theta_names
                    )
                    pubs.append((circuit, obs_outer, pvals))
                    meta.append((module_idx, start, stop))

            results = context.run_estimator_pubs(pubs)
            unique_arrays = [
                np.empty((unique_counts[i], p.n_qubits), dtype=np.float32)
                for i, p in enumerate(projections)
            ]
            for pub_result, (module_idx, start, stop) in zip(results, meta):
                unique_arrays[module_idx][start:stop] = _estimator_matrix(
                    pub_result, stop - start, projections[module_idx].n_qubits
                )
            arrays = []
            for arr, inverse in zip(unique_arrays, inverse_maps):
                arrays.append(arr if inverse is None else arr[inverse])
        else:
            # True single-circuit Q/K/V parameter sweep.  The same generic ISA
            # circuit is shared; Q/K/V differences live only in parameter rows.
            shared_circuit = qproj._direct_circuit
            shared_observables = qproj._direct_observables
            obs_outer = _observable_outer_array(shared_observables)
            matrices = []
            for proj, angles in zip(projections, angles_list):
                angles_np = angles.detach().cpu().numpy()
                named = {
                    name: angles_np[:, j]
                    for j, name in enumerate(qproj._direct_x_names)
                }
                weights_np = proj.q_layer.weight.detach().cpu().numpy().reshape(-1)
                for j, name in enumerate(qproj._direct_theta_names):
                    named[name] = weights_np[j]
                matrices.append(_parameter_matrix(shared_circuit, named))
            fused = np.concatenate(matrices, axis=0)
            inverse = None
            fused_exec = fused
            if context.ibm_deduplicate_parameter_rows and fused.shape[0] > 1:
                fused_exec, inverse = np.unique(fused, axis=0, return_inverse=True)
                if fused_exec.shape[0] < fused.shape[0]:
                    print(
                        f"Fused QKV parameter deduplication: {fused.shape[0]} -> "
                        f"{fused_exec.shape[0]} unique rows"
                    )

            pubs = []
            meta = []
            for start, stop in context.parameter_chunks(fused_exec.shape[0]):
                pubs.append((shared_circuit, obs_outer, fused_exec[start:stop]))
                meta.append((start, stop))
            results = context.run_estimator_pubs(pubs)
            unique_out = np.empty((fused_exec.shape[0], qproj.n_qubits), dtype=np.float32)
            for pub_result, (start, stop) in zip(results, meta):
                unique_out[start:stop] = _estimator_matrix(
                    pub_result, stop - start, qproj.n_qubits
                )
            fused_out = unique_out if inverse is None else unique_out[inverse]
            arrays = [
                fused_out[i * n_each:(i + 1) * n_each]
                for i in range(3)
            ]

        outputs = []
        for proj, angles, arr in zip(projections, angles_list, arrays):
            z = torch.from_numpy(arr).to(device=angles.device, dtype=angles.dtype)
            outputs.append(proj._finish_projection(z, batch_size, seq_len))
        return tuple(outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        angles, batch_size, seq_len = self._prepare_angles(x)

        if self._optimized_aer:
            state = _torch_ansatz_state(
                angles,
                self.q_layer.weight,
                self.n_qubits,
                self.n_q_layers,
            )
            z = _z_expectations(state, self.n_qubits)
        elif self._use_direct_primitive:
            try:
                z = self._direct_projection(angles)
            except Exception:
                if not self.quantum_context.direct_primitive_fallback_to_qnn:
                    raise
                key = ("projection", type(self.quantum_context.estimator).__name__)
                if key not in _DIRECT_FALLBACK_WARNED:
                    _DIRECT_FALLBACK_WARNED.add(key)
                    warnings.warn(
                        "Direct Estimator inference failed; falling back to "
                        "EstimatorQNN/TorchConnector. Set "
                        "direct_primitive_fallback_to_qnn=False to expose the "
                        "underlying primitive error while debugging.",
                        RuntimeWarning,
                    )
                z = self.q_layer(angles)
        else:
            z = self.q_layer(angles)

        return self._finish_projection(z, batch_size, seq_len)


# -----------------------------------------------------------------------------
# Fidelity kernel
# -----------------------------------------------------------------------------

class QuantumFidelityKernel(nn.Module):
    """Fidelity kernel with cached new-query-only Sampler inference.

    Hardware-style inference can bind the learned encoder weights once and
    retranspile the measured compute-uncompute circuit. Runtime Sampler sweeps
    then contain only q/k data angles.  The same cache semantics work on Aer and
    IBM hardware.
    """
    def __init__(
        self,
        in_dim: int,
        quantum_context: QuantumExecutionContext,
        n_qubits: int = 4,
        n_q_layers: int = 2,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.n_qubits = n_qubits
        self.n_q_layers = n_q_layers
        self.quantum_context = quantum_context

        hidden_dim = max(in_dim, n_qubits * 2)
        self.q_pre_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, n_qubits)
        )
        self.k_pre_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, n_qubits)
        )

        q_x = ParameterVector("q_x", n_qubits)
        k_x = ParameterVector("k_x", n_qubits)
        q_theta = ParameterVector("q_theta", n_q_layers * n_qubits * 3)
        k_theta = ParameterVector("k_theta", n_q_layers * n_qubits * 3)
        self._logical_q_x = list(q_x)
        self._logical_k_x = list(k_x)
        self._logical_q_theta = list(q_theta)
        self._logical_k_theta = list(k_theta)

        uq = QuantumCircuit(n_qubits, name="Uq")
        _append_hardware_ansatz(uq, q_x, q_theta, n_qubits, n_q_layers)
        uk = QuantumCircuit(n_qubits, name="Uk")
        _append_hardware_ansatz(uk, k_x, k_theta, n_qubits, n_q_layers)

        fidelity_circuit = QuantumCircuit(n_qubits)
        fidelity_circuit.compose(uq, inplace=True)
        fidelity_circuit.compose(uk.inverse(), inplace=True)
        self.logical_circuit = fidelity_circuit

        report_key = ("fidelity", quantum_context.backend_type,
                      quantum_context.execution_strategy,
                      quantum_context.aer_noise_enabled, n_qubits, n_q_layers)
        _maybe_report_circuit(
            quantum_context, fidelity_circuit, key=report_key,
            name="QuantumFidelityKernel compute-uncompute",
        )

        def zero_vs_other(measured_int: int) -> int:
            return 0 if measured_int == 0 else 1

        input_params = self._logical_q_x + self._logical_k_x
        weight_params = self._logical_q_theta + self._logical_k_theta
        qnn_circuit = fidelity_circuit
        if quantum_context.execution_strategy == "hardware":
            tkey = ("fidelity-isa", quantum_context.backend_type,
                    quantum_context.aer_noise_enabled, n_qubits, n_q_layers)
            qnn_circuit = _cached_transpile(quantum_context, fidelity_circuit, tkey)

        qnn_input_params = _resolve_params_by_name(qnn_circuit, input_params)
        qnn_weight_params = _resolve_params_by_name(qnn_circuit, weight_params)

        measured = fidelity_circuit.copy()
        measured.measure_all()
        self._measured_logical_circuit = measured
        if quantum_context.execution_strategy == "hardware":
            dkey = ("fidelity-direct-measured-isa", quantum_context.backend_type,
                    quantum_context.aer_noise_enabled, n_qubits, n_q_layers)
            measured = _cached_transpile(quantum_context, measured, dkey)
        self._direct_circuit = measured
        self._direct_input_names = [p.name for p in _resolve_params_by_name(measured, input_params)]
        self._direct_weight_names = [p.name for p in _resolve_params_by_name(measured, weight_params)]

        self._bound_signature = None
        self._bound_direct_circuit = None
        self._bound_direct_input_names = None

        qnn = SamplerQNN(
            circuit=qnn_circuit,
            sampler=quantum_context.sampler,
            input_params=qnn_input_params,
            weight_params=qnn_weight_params,
            interpret=zero_vs_other,
            output_shape=2,
            input_gradients=True,
            pass_manager=None,
        )
        initial_weights = 0.01 * np.random.randn(len(weight_params))
        self.fidelity_qnn = TorchConnector(qnn, initial_weights=initial_weights)
        self._n_single_encoder_weights = len(q_theta)

    @property
    def _optimized_aer(self) -> bool:
        return (
            self.quantum_context.backend_type == "aer"
            and self.quantum_context.execution_strategy == "optimized"
        )

    @property
    def _use_direct_primitive(self) -> bool:
        return (
            (not self.training)
            and self.quantum_context.execution_strategy == "hardware"
            and self.quantum_context.direct_primitive_inference
        )

    def _q_angles(self, q: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.q_pre_net(q)) * (math.pi / 2.0)

    def _k_angles(self, k: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.k_pre_net(k)) * (math.pi / 2.0)

    def _split_weights(self):
        all_weights = self.fidelity_qnn.weight
        return (all_weights[: self._n_single_encoder_weights],
                all_weights[self._n_single_encoder_weights :])

    def _score_optimized(self, q_angles, k_angles, cached_k_states=None):
        B, H, Tq, _ = q_angles.shape
        Tk = k_angles.size(2)
        q_weights, k_weights = self._split_weights()

        q_state = _torch_ansatz_state(
            q_angles.reshape(B * H * Tq, self.n_qubits),
            q_weights, self.n_qubits, self.n_q_layers,
        ).reshape(B, H, Tq, -1)

        if cached_k_states is None:
            k_state = _torch_ansatz_state(
                k_angles.reshape(B * H * Tk, self.n_qubits),
                k_weights, self.n_qubits, self.n_q_layers,
            ).reshape(B, H, Tk, -1)
        else:
            k_state = cached_k_states

        overlaps = torch.matmul(torch.conj(q_state), k_state.transpose(-2, -1))
        return (torch.abs(overlaps) ** 2).real.clamp(0.0, 1.0), k_state

    def _ensure_bound_direct_circuit(self):
        weights_np = self.fidelity_qnn.weight.detach().cpu().numpy().reshape(-1)
        signature = _weight_signature(weights_np)
        if self._bound_direct_circuit is not None and signature == self._bound_signature:
            return self._bound_direct_circuit, self._bound_direct_input_names

        nq = self._n_single_encoder_weights
        mapping = {}
        for i, p in enumerate(self._logical_q_theta):
            mapping[p] = float(weights_np[i])
        for i, p in enumerate(self._logical_k_theta):
            mapping[p] = float(weights_np[nq + i])

        bound = self._measured_logical_circuit.assign_parameters(mapping, inplace=False)
        if self.quantum_context.execution_strategy == "hardware":
            bound = self.quantum_context.pass_manager.run(bound)
        input_params = self._logical_q_x + self._logical_k_x
        input_names = [p.name for p in _resolve_params_by_name(bound, input_params)]

        self._bound_signature = signature
        self._bound_direct_circuit = bound
        self._bound_direct_input_names = input_names

        stats_key = (
            "fidelity-bound-representative",
            self.quantum_context.backend_type,
            self.quantum_context.aer_noise_enabled,
            self.n_qubits,
            self.n_q_layers,
        )
        if self.quantum_context.report_circuit_stats and stats_key not in _PRINTED_STATS_KEYS:
            _PRINTED_STATS_KEYS.add(stats_key)
            print_circuit_metrics(bound, "QuantumFidelityKernel - fixed-weight measured ISA (representative)")
        return bound, input_names

    def _direct_components(self):
        if self.quantum_context.direct_bind_inference_weights:
            circuit, input_names = self._ensure_bound_direct_circuit()
            return circuit, input_names, []
        return self._direct_circuit, self._direct_input_names, self._direct_weight_names

    def _expanded_pair_mask(self, pair_mask, B, H, Tq, Tk, device):
        """Return a [B,H,Tq,Tk] boolean mask or None.

        Causal-pair pruning is used only by hardware-style execution.  Masked
        entries are never submitted to the Sampler, then are scattered back as
        zeros so downstream normalization is mathematically identical to the
        previous compute-all-then-mask implementation.
        """
        if pair_mask is None or not self.quantum_context.causal_pair_pruning:
            return None
        mask = pair_mask.to(device=device, dtype=torch.bool)
        if mask.ndim != 4 or mask.shape[-2:] != (Tq, Tk):
            raise ValueError(
                f"pair_mask must broadcast as [B,H,{Tq},{Tk}], got {tuple(mask.shape)}"
            )
        try:
            return mask.expand(B, H, Tq, Tk)
        except RuntimeError as exc:
            raise ValueError(
                f"pair_mask shape {tuple(mask.shape)} is not broadcastable to "
                f"[{B},{H},{Tq},{Tk}]"
            ) from exc

    def _build_pruned_pair_inputs(self, q_angles, k_angles, pair_mask=None):
        B, H, Tq, _ = q_angles.shape
        Tk = k_angles.size(2)
        q_pair = q_angles.unsqueeze(3).expand(B, H, Tq, Tk, self.n_qubits)
        k_pair = k_angles.unsqueeze(2).expand(B, H, Tq, Tk, self.n_qubits)
        pair_inputs = torch.cat([q_pair, k_pair], dim=-1).reshape(
            B * H * Tq * Tk, 2 * self.n_qubits
        )

        expanded_mask = self._expanded_pair_mask(
            pair_mask, B, H, Tq, Tk, q_angles.device
        )
        if expanded_mask is None:
            return pair_inputs, None, B, H, Tq, Tk

        valid_flat = expanded_mask.reshape(-1)
        valid_idx = torch.nonzero(valid_flat, as_tuple=False).squeeze(-1)
        pair_inputs = pair_inputs.index_select(0, valid_idx)

        key = (B, H, Tq, Tk, int(valid_idx.numel()))
        if key not in _PAIR_PRUNING_PRINTED:
            _PAIR_PRUNING_PRINTED.add(key)
            full = B * H * Tq * Tk
            kept = int(valid_idx.numel())
            pct = 0.0 if full == 0 else 100.0 * (full - kept) / full
            print(
                f"Causal fidelity-pair pruning: {full} -> {kept} parameter sets "
                f"({pct:.1f}% pruned) for [B,H,Tq,Tk]=[{B},{H},{Tq},{Tk}]"
            )
        return pair_inputs, valid_idx, B, H, Tq, Tk

    def _scatter_pruned_scores(self, values, valid_idx, B, H, Tq, Tk, device, dtype):
        if valid_idx is None:
            return values.reshape(B, H, Tq, Tk)
        full = torch.zeros(B * H * Tq * Tk, device=device, dtype=dtype)
        full.index_copy_(0, valid_idx.to(device=device), values)
        return full.reshape(B, H, Tq, Tk)

    def _score_direct_sampler(self, q_angles, k_angles, pair_mask=None):
        """Sampler sweep over only causally valid query-key pairs when possible."""
        pair_inputs, valid_idx, B, H, Tq, Tk = self._build_pruned_pair_inputs(
            q_angles, k_angles, pair_mask
        )

        pair_np = pair_inputs.detach().cpu().numpy()
        n_original = pair_np.shape[0]
        inverse = None
        if self.quantum_context.ibm_deduplicate_parameter_rows and n_original > 1:
            unique_pairs, inverse = np.unique(pair_np, axis=0, return_inverse=True)
            if unique_pairs.shape[0] < n_original:
                print(
                    f"QuantumFidelity parameter deduplication: {n_original} -> "
                    f"{unique_pairs.shape[0]} unique rows"
                )
            pair_np = unique_pairs
        n = pair_np.shape[0]
        if n == 0:
            return torch.zeros((B, H, Tq, Tk), device=q_angles.device, dtype=q_angles.dtype)
        circuit, input_names, weight_names = self._direct_components()
        weights_np = self.fidelity_qnn.weight.detach().cpu().numpy().reshape(-1)

        pubs = []
        pub_meta = []
        for start, stop in self.quantum_context.parameter_chunks(
            n, shots=self.quantum_context.shots
        ):
            named = {}
            for j, name in enumerate(input_names):
                named[name] = pair_np[start:stop, j]
            for j, name in enumerate(weight_names):
                named[name] = weights_np[j]
            pvals = _parameter_matrix(circuit, named)
            pubs.append((circuit, pvals))
            pub_meta.append((start, stop))

        results = self.quantum_context.run_sampler_pubs(
            pubs, shots=self.quantum_context.shots
        )
        out = np.empty(n, dtype=np.float32)
        for pub_result, (start, stop) in zip(results, pub_meta):
            out[start:stop] = _sampler_zero_probabilities(
                pub_result, stop - start, self.n_qubits
            )
        if inverse is not None:
            out = out[inverse]
        values = torch.from_numpy(out).to(
            device=q_angles.device, dtype=q_angles.dtype
        ).clamp(0.0, 1.0)
        return self._scatter_pruned_scores(
            values, valid_idx, B, H, Tq, Tk, q_angles.device, q_angles.dtype
        )

    def _score_hardware(self, q_angles, k_angles, pair_mask=None):
        if self._use_direct_primitive:
            try:
                return self._score_direct_sampler(q_angles, k_angles, pair_mask=pair_mask)
            except Exception:
                if not self.quantum_context.direct_primitive_fallback_to_qnn:
                    raise
                key = ("fidelity", type(self.quantum_context.sampler).__name__)
                if key not in _DIRECT_FALLBACK_WARNED:
                    _DIRECT_FALLBACK_WARNED.add(key)
                    warnings.warn(
                        "Direct Sampler fidelity inference failed; falling back to "
                        "SamplerQNN/TorchConnector.", RuntimeWarning,
                    )

        pair_inputs, valid_idx, B, H, Tq, Tk = self._build_pruned_pair_inputs(
            q_angles, k_angles, pair_mask
        )
        if pair_inputs.shape[0] == 0:
            return torch.zeros((B, H, Tq, Tk), device=q_angles.device, dtype=q_angles.dtype)
        probs = self.fidelity_qnn(pair_inputs)
        values = probs[..., 0].clamp(0.0, 1.0)
        return self._scatter_pruned_scores(
            values, valid_idx, B, H, Tq, Tk, q_angles.device, q_angles.dtype
        )

    def forward(self, q: torch.Tensor, k: torch.Tensor, pair_mask=None) -> torch.Tensor:
        if q.shape[:-1] != k.shape[:-1] or q.shape[-1] != k.shape[-1]:
            raise ValueError("q and k must have identical [B,H,S,D] shapes")
        q_angles = self._q_angles(q)
        k_angles = self._k_angles(k)
        if self._optimized_aer:
            # Stage 1 deliberately keeps the dense statevector overlap matmul;
            # triangular indexing would be slower than the tiny dense GEMM.
            scores, _ = self._score_optimized(q_angles, k_angles)
            return scores
        return self._score_hardware(q_angles, k_angles, pair_mask=pair_mask)

    def forward_cached(self, q_new, k_new, cache=None):
        """Score only new query positions against cached + new keys."""
        cache = {} if cache is None else cache
        q_angles = self._q_angles(q_new)
        k_new_angles = self._k_angles(k_new)
        old_angles = cache.get("k_angles")
        k_angles = k_new_angles if old_angles is None else torch.cat([old_angles, k_new_angles], dim=2)

        new_cache = {"k_angles": k_angles.detach()}
        if self._optimized_aer:
            _, k_weights = self._split_weights()
            B, H, Tnew, _ = k_new_angles.shape
            new_states = _torch_ansatz_state(
                k_new_angles.reshape(B * H * Tnew, self.n_qubits),
                k_weights, self.n_qubits, self.n_q_layers,
            ).reshape(B, H, Tnew, -1)
            old_states = cache.get("k_states")
            all_states = new_states if old_states is None else torch.cat([old_states, new_states], dim=2)
            scores, _ = self._score_optimized(q_angles, k_angles, cached_k_states=all_states)
            new_cache["k_states"] = all_states.detach()
        else:
            # q_new contains only newly appended query tokens; k_angles contains
            # cached historical keys plus new keys. Old-old fidelity blocks are
            # never resubmitted.  If Tnew>1, also prune future pairs among the
            # newly appended tokens before they reach the Sampler.
            pair_mask = None
            if self.quantum_context.causal_pair_pruning:
                past_len = 0 if old_angles is None else old_angles.size(2)
                Tnew = q_angles.size(2)
                Ttotal = k_angles.size(2)
                q_pos = past_len + torch.arange(Tnew, device=q_angles.device)
                k_pos = torch.arange(Ttotal, device=q_angles.device)
                pair_mask = (k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)).view(
                    1, 1, Tnew, Ttotal
                )
            scores = self._score_hardware(q_angles, k_angles, pair_mask=pair_mask)

        return scores, new_cache

# -----------------------------------------------------------------------------
# Attention and transformer block
# -----------------------------------------------------------------------------

class HybridQuantumSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        quantum_context: QuantumExecutionContext,
        num_heads: int = 8,
        q_proj_qubits: int = 4,
        q_proj_layers: int = 2,
        use_quantum_similarity: bool = False,
        q_sim_qubits: int = 4,
        q_sim_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.projection_dim = embed_dim // num_heads
        self.use_quantum_similarity = use_quantum_similarity

        self.query_proj = QuantumProjection(embed_dim, embed_dim, quantum_context, q_proj_qubits, q_proj_layers)
        self.key_proj = QuantumProjection(embed_dim, embed_dim, quantum_context, q_proj_qubits, q_proj_layers)
        self.value_proj = QuantumProjection(embed_dim, embed_dim, quantum_context, q_proj_qubits, q_proj_layers)
        self.quantum_similarity = (
            QuantumFidelityKernel(self.projection_dim, quantum_context, q_sim_qubits, q_sim_layers)
            if use_quantum_similarity else None
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.combine_heads = nn.Linear(embed_dim, embed_dim)

    def split_heads(self, x, batch_size):
        return x.view(batch_size, -1, self.num_heads, self.projection_dim).transpose(1, 2)

    @staticmethod
    def classical_attention_logits(query, key):
        return torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(key.size(-1))

    @staticmethod
    def _incremental_causal_mask(Tnew, Ttotal, past_len, device):
        q_pos = past_len + torch.arange(Tnew, device=device)
        k_pos = torch.arange(Ttotal, device=device)
        return (k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)).view(1, 1, Tnew, Ttotal)

    def _project_qkv(self, inputs):
        projections = [self.query_proj, self.key_proj, self.value_proj]
        context = self.query_proj.quantum_context
        if (
            context.direct_fuse_qkv_estimator
            and all(p._use_direct_primitive for p in projections)
        ):
            try:
                return QuantumProjection.forward_fused_qkv(projections, inputs)
            except Exception:
                if not context.direct_primitive_fallback_to_qnn:
                    raise
                key = ("fused-qkv", type(context.estimator).__name__)
                if key not in _DIRECT_FALLBACK_WARNED:
                    _DIRECT_FALLBACK_WARNED.add(key)
                    warnings.warn(
                        "Fused Q/K/V Estimator execution failed; falling back to "
                        "the per-projection path.", RuntimeWarning,
                    )
        return (
            self.query_proj(inputs),
            self.key_proj(inputs),
            self.value_proj(inputs),
        )

    def attention(self, query, key, value, attention_mask=None):
        if self.use_quantum_similarity:
            pair_mask = (
                attention_mask
                if self.query_proj.quantum_context.causal_pair_pruning
                else None
            )
            scores = self.quantum_similarity(
                query, key, pair_mask=pair_mask
            ).to(device=value.device, dtype=value.dtype)
            if attention_mask is not None:
                scores = scores.masked_fill(attention_mask == 0, 0.0)
            weights = scores / scores.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        else:
            logits = self.classical_attention_logits(query, key)
            if attention_mask is not None:
                logits = logits.masked_fill(attention_mask == 0, float("-inf"))
            weights = F.softmax(logits, dim=-1)
        weights = self.attn_dropout(weights)
        return torch.matmul(weights, value), weights

    def forward(self, inputs, attention_mask=None):
        B = inputs.size(0)
        q_raw, k_raw, v_raw = self._project_qkv(inputs)
        q = self.split_heads(q_raw, B)
        k = self.split_heads(k_raw, B)
        v = self.split_heads(v_raw, B)
        attention, weights = self.attention(q, k, v, attention_mask)
        attention = attention.transpose(1, 2).contiguous().view(B, -1, self.embed_dim)
        return self.combine_heads(attention), weights

    def forward_cached(self, inputs_new, cache=None):
        """Incremental causal attention without changing block semantics."""
        cache = {} if cache is None else cache
        B, Tnew, _ = inputs_new.shape
        q_raw, k_raw, v_raw = self._project_qkv(inputs_new)
        q_new = self.split_heads(q_raw, B)
        k_new = self.split_heads(k_raw, B)
        v_new = self.split_heads(v_raw, B)

        k_old, v_old = cache.get("key"), cache.get("value")
        past_len = 0 if k_old is None else k_old.size(2)
        k_all = k_new if k_old is None else torch.cat([k_old, k_new], dim=2)
        v_all = v_new if v_old is None else torch.cat([v_old, v_new], dim=2)
        mask = self._incremental_causal_mask(Tnew, k_all.size(2), past_len, inputs_new.device)

        if self.use_quantum_similarity:
            scores, fidelity_cache = self.quantum_similarity.forward_cached(
                q_new, k_new, cache.get("fidelity")
            )
            scores = scores.to(device=v_all.device, dtype=v_all.dtype)
            scores = scores.masked_fill(mask == 0, 0.0)
            weights = scores / scores.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        else:
            logits = self.classical_attention_logits(q_new, k_all)
            logits = logits.masked_fill(mask == 0, float("-inf"))
            weights = F.softmax(logits, dim=-1)
            fidelity_cache = None

        weights = self.attn_dropout(weights)
        attention = torch.matmul(weights, v_all)
        attention = attention.transpose(1, 2).contiguous().view(B, Tnew, self.embed_dim)
        out = self.combine_heads(attention)
        new_cache = {
            "key": k_all.detach(),
            "value": v_all.detach(),
            "fidelity": fidelity_cache,
        }
        return out, weights, new_cache


class QuantumTransformerBlock(nn.Module):
    def __init__(
        self, embed_dim, num_heads, q_proj_qubits, q_proj_layers,
        q_sim_qubits, q_sim_layers, use_quantum_similarity, quantum_context,
    ):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.attn = HybridQuantumSelfAttention(
            embed_dim=embed_dim, num_heads=num_heads,
            q_proj_qubits=q_proj_qubits, q_proj_layers=q_proj_layers,
            use_quantum_similarity=use_quantum_similarity,
            q_sim_qubits=q_sim_qubits, q_sim_layers=q_sim_layers,
            quantum_context=quantum_context, dropout=0.1,
        )
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim), nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim), nn.Dropout(0.1),
        )

    def forward(self, x, attention_mask=None):
        attn_out, _ = self.attn(self.ln_1(x), attention_mask=attention_mask)
        x = x + attn_out
        return x + self.mlp(self.ln_2(x))

    def forward_cached(self, x_new, cache=None):
        attn_out, _, new_cache = self.attn.forward_cached(self.ln_1(x_new), cache=cache)
        x_new = x_new + attn_out
        x_new = x_new + self.mlp(self.ln_2(x_new))
        return x_new, new_cache

if __name__ == "__main__":
    qctx = build_quantum_context(
        backend_type="aer",
        execution_strategy="optimized",
        min_num_qubits=4,
        shots=4096,
        optimization_level=1,
        aer_method="statevector",
        aer_device="CPU",
        report_circuit_stats=True,
    )
    model = QuantumTransformerBlock(
        embed_dim=8,
        num_heads=2,
        q_proj_qubits=4,
        q_proj_layers=1,
        q_sim_qubits=3,
        q_sim_layers=1,
        use_quantum_similarity=True,
        quantum_context=qctx,
    )
    x = torch.randn(1, 2, 8)
    y = model(x)
    print("backend_type:", qctx.backend_type)
    print("execution_strategy:", qctx.execution_strategy)
    print("output shape:", y.shape)
