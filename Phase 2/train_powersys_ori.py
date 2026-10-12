import math
import logging
import time
import multiprocessing as mp

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader

logger = logging.getLogger(__name__)

from mingpt_240_5.utils_ori import sample
from F_240Bus_RLBase_Dyna import GraphEnv


class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1
    lr_decay = False
    warmup_tokens = 375e6
    final_tokens = 260e9
    ckpt_path = None
    num_workers = 1

    # Qiskit hardware-style execution should stay on CPU. Stage 1 optimized
    # statevector training may use CUDA.
    use_data_parallel = False
    force_cpu = False

    # Stage-4 N-episode hardware evaluation. The 240-bus OpenDSS scenario uses
    # module-global DSS state, so independent episodes are isolated in separate
    # POSIX worker processes. Only current model inputs are batched in the parent.
    parallel_episode_evaluation = False
    parallel_episode_workers = 10
    parallel_env_start_method = "spawn"

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def _parallel_240bus_env_worker(conn, graph_data):
    """Own one independent OpenDSS restoration environment in a child process.

    The original 240-bus module stores its OpenDSS feeder at module scope. Running
    several GraphEnv objects in one process would therefore corrupt episode state.
    A separate process gives each episode its own copy of the DSS engine/state.
    """
    try:
        env = GraphEnv(graph_data)
        state = env.reset()
        conn.send({
            "status": "ready",
            "state": state,
            "binary_history": np.asarray(env.binary_history, dtype=np.float32),
        })

        while True:
            msg = conn.recv()
            command = msg.get("command")
            if command == "close":
                break
            if command == "reset":
                state = env.reset()
                conn.send({
                    "status": "ready",
                    "state": state,
                    "binary_history": np.asarray(env.binary_history, dtype=np.float32),
                })
                continue
            if command != "step":
                raise ValueError(f"Unknown environment worker command: {command!r}")

            proposed = list(msg["proposed_action"])
            sampled_all_actions = []
            all_valid_actions = []
            for l in range(5):
                valid_actions = env.get_valid_actions(l, state[l], sampled_all_actions)
                all_valid_actions.append(valid_actions)
                if proposed[l] in valid_actions:
                    sampled_all_actions.append(proposed[l])
                else:
                    sampled_all_actions.append(state[l])

            encoded_action = twetri_to_num(sampled_all_actions)
            state, reward, done, errord, power = env.step(
                sampled_all_actions, all_valid_actions
            )

            if errord:
                # Immediately restore this slot to a clean episode so the parent
                # can keep it in the next batched inference frontier.
                reset_state = env.reset()
                conn.send({
                    "status": "failed_reset",
                    "encoded_action": encoded_action,
                    "state": reset_state,
                    "binary_history": np.asarray(env.binary_history, dtype=np.float32),
                })
            elif done:
                conn.send({
                    "status": "done",
                    "encoded_action": encoded_action,
                    "state": state,
                    "reward": float(reward),
                    "power": float(power),
                    "restoration_period": int(env.restoration_period),
                    "binary_history": np.asarray(env.binary_history, dtype=np.float32),
                })
            else:
                conn.send({
                    "status": "running",
                    "encoded_action": encoded_action,
                    "state": state,
                    "reward": float(reward),
                    "power": float(power),
                    "restoration_period": int(env.restoration_period),
                    "binary_history": np.asarray(env.binary_history, dtype=np.float32),
                })
    except BaseException as exc:
        try:
            conn.send({"status": "worker_error", "error": repr(exc)})
        except Exception:
            pass
        raise
    finally:
        conn.close()


class Trainer:
    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config

        qctx = getattr(self.model, "quantum_context", None)
        hardware_style = (
            qctx is not None
            and getattr(qctx, "execution_strategy", "hardware") == "hardware"
        )

        if getattr(config, "force_cpu", False) or hardware_style or not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda:0")

        self.model = self.model.to(self.device)

        # Qiskit QNN/primitive objects should not be replicated by DataParallel.
        if getattr(config, "use_data_parallel", False):
            if hardware_style:
                raise ValueError(
                    "use_data_parallel=True is not supported for hardware-style quantum execution."
                )
            if torch.cuda.device_count() > 1:
                self.model = torch.nn.DataParallel(self.model)

    def save_checkpoint(self):
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info("saving %s", self.config.ckpt_path)
        # Explicit checkpoint saving is handled by the scenario main file.

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(model, "module") else model
        optimizer = raw_model.configure_optimizers(config)

        self.tokens = 0
        model.train()

        for epoch in range(config.max_epochs):
            loader = DataLoader(
                self.train_dataset,
                shuffle=True,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
            )

            pbar = tqdm(enumerate(loader), total=len(loader))
            for it, (x, y, r, t) in pbar:
                x = x.to(self.device)
                y = y.to(self.device)
                r = r.to(self.device)
                t = t.to(self.device)

                with torch.set_grad_enabled(True):
                    logits, loss = model(x, y, y, r, t)
                    loss = loss.mean()

                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()

                    if config.lr_decay:
                        self.tokens += (y >= 0).sum()
                        if self.tokens < config.warmup_tokens:
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            progress = float(self.tokens - config.warmup_tokens) / float(
                                max(1, config.final_tokens - config.warmup_tokens)
                            )
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = lr
                    else:
                        lr = config.learning_rate

                    pbar.set_description(
                        f"epoch {epoch + 1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}"
                    )

    def evaluate(self, graph_data, ret, N):
        if (
            getattr(self.config, "parallel_episode_evaluation", False)
            and int(N) > 1
        ):
            return self._evaluate_parallel_context1(graph_data, ret, int(N))
        return self._evaluate_serial(graph_data, ret, int(N))

    def _evaluate_serial(self, graph_data, ret, N):
        """Evaluate the 240-bus restoration environment with stable histories.

        Scenario-specific behavior from the original implementation is preserved:
        * state is represented as a 23-way binary indicator for the five active
          locations concatenated with ``env.binary_history``;
        * one scalar action index encodes five base-23 choices;
        * each of the five proposed actions is checked sequentially against the
          environment's valid-action set;
        * failed attempts restart and only 20-period successful restorations count.

        The history tensors and per-step timesteps follow the current V2 cache
        semantics used by the four-stage quantum implementation.
        """
        model, config = self.model, self.config
        model.eval()
        raw_model = model.module if hasattr(model, "module") else model

        graph_data = graph_data.to(self.device)
        env = GraphEnv(graph_data)

        T_rewards = []
        restored_power = []
        i = 0

        evaluation_start = time.perf_counter()
        with torch.inference_mode():
            while i < N:
                episode_start = time.perf_counter()
                print("Test start!")

                if hasattr(raw_model, "reset_inference_cache"):
                    raw_model.reset_inference_cache()

                state = env.reset()

                binary_state = np.zeros(23, dtype=np.float32)
                binary_state[np.asarray(state[:5], dtype=np.int64)] = 1.0
                binary_state_tensor = torch.tensor(
                    binary_state, dtype=torch.float32, device=self.device
                )
                binary_history_tensor = torch.tensor(
                    env.binary_history, dtype=torch.float32, device=self.device
                )
                state_input = torch.cat(
                    (binary_state_tensor, binary_history_tensor)
                ).view(1, 1, -1)

                # [B,T,*] histories. Keeping one timestep per physical state is
                # required by the current exact-prefix quantum KV-cache semantics.
                all_states = state_input
                action_history = None
                rtg_history = torch.tensor(
                    [[[ret]]], dtype=torch.float32, device=self.device
                )
                timestep_history = torch.zeros(
                    (1, 1, 1), dtype=torch.int64, device=self.device
                )

                sampled_action = sample(
                    raw_model,
                    all_states,
                    1,
                    temperature=1.0,
                    sample=True,
                    actions=action_history,
                    rtgs=rtg_history,
                    timesteps=timestep_history,
                )

                j = 0
                reward_sum = 0

                while True:
                    sampled_all_actions = []
                    all_valid_actions = []
                    proposed = num_to_twetri(int(sampled_action.item()))

                    for l in range(5):
                        valid_actions = env.get_valid_actions(
                            l, state[l], sampled_all_actions
                        )
                        all_valid_actions.append(valid_actions)
                        if proposed[l] in valid_actions:
                            sampled_all_actions.append(proposed[l])
                        else:
                            sampled_all_actions.append(state[l])

                    encoded_action = twetri_to_num(sampled_all_actions)
                    new_action = torch.tensor(
                        [[[encoded_action]]], dtype=torch.long, device=self.device
                    )
                    action_history = (
                        new_action
                        if action_history is None
                        else torch.cat([action_history, new_action], dim=1)
                    )

                    state, reward, done, errord, power = env.step(
                        sampled_all_actions, all_valid_actions
                    )
                    reward_sum += reward
                    j += 1

                    if errord:
                        print("This is a failed attempt. System restarted")
                        break

                    if done:
                        if env.restoration_period == 20:
                            T_rewards.append(reward_sum)
                            restored_power.append(power)
                            i += 1
                            print("This is the ", i, "th successful run.")
                            print(
                                f"Episode wall time: {time.perf_counter() - episode_start:.3f} s"
                            )
                        break

                    binary_state = np.zeros(23, dtype=np.float32)
                    binary_state[np.asarray(state[:5], dtype=np.int64)] = 1.0
                    binary_state_tensor = torch.tensor(
                        binary_state, dtype=torch.float32, device=self.device
                    )
                    binary_history_tensor = torch.tensor(
                        env.binary_history, dtype=torch.float32, device=self.device
                    )
                    state_input = torch.cat(
                        (binary_state_tensor, binary_history_tensor)
                    ).view(1, 1, -1)
                    all_states = torch.cat([all_states, state_input], dim=1)

                    next_rtg = rtg_history[:, -1:, :] - float(reward)
                    rtg_history = torch.cat([rtg_history, next_rtg], dim=1)

                    next_t = min(j, self.config.max_timestep)
                    next_t_tensor = torch.full(
                        (1, 1, 1),
                        next_t,
                        dtype=torch.int64,
                        device=self.device,
                    )
                    timestep_history = torch.cat(
                        [timestep_history, next_t_tensor], dim=1
                    )

                    sampled_action = sample(
                        raw_model,
                        all_states,
                        1,
                        temperature=1.0,
                        sample=True,
                        actions=action_history,
                        rtgs=rtg_history,
                        timesteps=timestep_history,
                    )

        print(
            f"Total evaluation wall time: {time.perf_counter() - evaluation_start:.3f} s"
        )
        eval_return = sum(T_rewards) / N
        print(T_rewards)
        print(restored_power)
        print(
            "target return: %f, average eval return: %f with standard deviation %f"
            % (ret, eval_return, np.std(T_rewards))
        )
        print(
            "Average eval restore power: %f with standard deviation %f"
            % (np.mean(restored_power), np.std(restored_power))
        )


    def _evaluate_parallel_context1(self, graph_data, ret, N):
        """Evaluate N successful 240-bus episodes in synchronized model batches.

        This optimization is intentionally restricted to ``context_length == 1``.
        Each active environment contributes one [RTG, state] inference item, so
        episodes may be at different physical restoration periods after a retry
        without padding fake quantum tokens.  The OpenDSS environments themselves
        live in separate processes; the quantum model remains in this parent
        process and sees one batched tensor containing all currently active slots.
        """
        model, config = self.model, self.config
        model.eval()
        raw_model = model.module if hasattr(model, "module") else model

        max_context = raw_model.get_block_size() // 3
        if max_context != 1:
            raise ValueError(
                "parallel 240-bus Stage-4 evaluation currently requires "
                f"context_length=1, got {max_context}."
            )
        if self.device.type != "cpu":
            raise ValueError("parallel OpenDSS hardware evaluation requires CPU model execution")

        requested_workers = max(1, int(getattr(config, "parallel_episode_workers", N)))
        worker_count = min(int(N), requested_workers)
        if worker_count != N:
            # The current implementation maps one successful target episode to one
            # long-lived environment slot. Keep the semantics explicit.
            raise ValueError(
                f"For N={N}, set parallel_episode_workers >= N; got {worker_count}."
            )

        start_method = str(getattr(config, "parallel_env_start_method", "spawn"))
        available = mp.get_all_start_methods()
        if start_method not in available:
            raise RuntimeError(
                f"multiprocessing start method {start_method!r} is unavailable; "
                f"available={available}. The 240-bus OpenDSS isolation path is "
                "configured for an isolated OpenDSS worker start method."
            )
        ctx = mp.get_context(start_method)
        graph_cpu = graph_data.to("cpu")

        parents = []
        workers = []
        slots = []
        T_rewards = []
        restored_power = []
        evaluation_start = time.perf_counter()

        try:
            for slot_id in range(N):
                parent_conn, child_conn = ctx.Pipe()
                proc = ctx.Process(
                    target=_parallel_240bus_env_worker,
                    args=(child_conn, graph_cpu),
                    daemon=True,
                )
                proc.start()
                child_conn.close()
                parents.append(parent_conn)
                workers.append(proc)

            for slot_id, conn in enumerate(parents):
                msg = conn.recv()
                if msg.get("status") == "worker_error":
                    raise RuntimeError(f"OpenDSS worker {slot_id} failed: {msg.get('error')}")
                if msg.get("status") != "ready":
                    raise RuntimeError(f"Unexpected initial worker message: {msg}")
                slots.append({
                    "state": msg["state"],
                    "binary_history": msg["binary_history"],
                    "rtg": float(ret),
                    "timestep": 0,
                    "reward_sum": 0.0,
                    "attempts": 1,
                    "finished": False,
                })

            frontier = 0
            with torch.inference_mode():
                while len(T_rewards) < N:
                    active = [i for i, slot in enumerate(slots) if not slot["finished"]]
                    if not active:
                        break
                    frontier += 1

                    state_rows = []
                    rtg_rows = []
                    timestep_rows = []
                    for idx in active:
                        slot = slots[idx]
                        binary_state = np.zeros(23, dtype=np.float32)
                        binary_state[np.asarray(slot["state"][:5], dtype=np.int64)] = 1.0
                        row = np.concatenate([
                            binary_state,
                            np.asarray(slot["binary_history"], dtype=np.float32),
                        ])
                        state_rows.append(row)
                        rtg_rows.append(slot["rtg"])
                        timestep_rows.append(min(slot["timestep"], config.max_timestep))

                    states_batch = torch.tensor(
                        np.stack(state_rows, axis=0),
                        dtype=torch.float32,
                        device=self.device,
                    ).unsqueeze(1)
                    rtgs_batch = torch.tensor(
                        rtg_rows, dtype=torch.float32, device=self.device
                    ).view(len(active), 1, 1)
                    timesteps_batch = torch.tensor(
                        timestep_rows, dtype=torch.int64, device=self.device
                    ).view(len(active), 1, 1)

                    # context_length=1 cannot reuse a historical KV prefix. Reset
                    # explicitly so changing active-batch membership cannot leak a
                    # cache entry from one set of episode slots into another.
                    if hasattr(raw_model, "reset_inference_cache"):
                        raw_model.reset_inference_cache()

                    wave_start = time.perf_counter()
                    sampled_actions = sample(
                        raw_model,
                        states_batch,
                        1,
                        temperature=1.0,
                        sample=True,
                        actions=None,
                        rtgs=rtgs_batch,
                        timesteps=timesteps_batch,
                    ).reshape(-1)

                    # Submit every environment step before waiting for any one
                    # result. OpenDSS work therefore overlaps across processes and
                    # does not occupy IBM quantum execution time.
                    for local_i, slot_id in enumerate(active):
                        proposed = num_to_twetri(int(sampled_actions[local_i].item()))
                        parents[slot_id].send({
                            "command": "step",
                            "proposed_action": proposed,
                        })

                    for slot_id in active:
                        msg = parents[slot_id].recv()
                        status = msg.get("status")
                        if status == "worker_error":
                            raise RuntimeError(
                                f"OpenDSS worker {slot_id} failed: {msg.get('error')}"
                            )
                        slot = slots[slot_id]

                        if status == "failed_reset":
                            print(
                                f"Episode slot {slot_id + 1} failed; restarting its "
                                "OpenDSS trajectory while other slots continue."
                            )
                            slot.update({
                                "state": msg["state"],
                                "binary_history": msg["binary_history"],
                                "rtg": float(ret),
                                "timestep": 0,
                                "reward_sum": 0.0,
                                "attempts": slot["attempts"] + 1,
                            })
                            continue

                        if status not in {"running", "done"}:
                            raise RuntimeError(f"Unexpected worker message: {msg}")

                        reward = float(msg["reward"])
                        slot["reward_sum"] += reward
                        slot["state"] = msg["state"]
                        slot["binary_history"] = msg["binary_history"]
                        slot["rtg"] -= reward
                        slot["timestep"] += 1

                        if status == "done":
                            if int(msg.get("restoration_period", -1)) != 20:
                                raise RuntimeError(
                                    "240-bus environment reported done before period 20"
                                )
                            slot["finished"] = True
                            T_rewards.append(slot["reward_sum"])
                            restored_power.append(float(msg["power"]))
                            print(
                                f"Successful episode {len(T_rewards)}/{N} from slot "
                                f"{slot_id + 1} after {slot['attempts']} attempt(s)."
                            )

                    print(
                        f"Parallel inference frontier {frontier}: {len(active)} active "
                        f"episode(s), wall time {time.perf_counter() - wave_start:.3f} s"
                    )
        finally:
            for conn in parents:
                try:
                    conn.send({"command": "close"})
                except Exception:
                    pass
            for conn in parents:
                try:
                    conn.close()
                except Exception:
                    pass
            for proc in workers:
                proc.join(timeout=5.0)
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=2.0)

        if len(T_rewards) != N:
            raise RuntimeError(
                f"Parallel evaluation ended with {len(T_rewards)} successful episodes; expected {N}."
            )

        print(f"Total parallel evaluation wall time: {time.perf_counter() - evaluation_start:.3f} s")
        eval_return = sum(T_rewards) / N
        print(T_rewards)
        print(restored_power)
        print(
            "target return: %f, average eval return: %f with standard deviation %f"
            % (ret, eval_return, np.std(T_rewards))
        )
        print(
            "Average eval restore power: %f with standard deviation %f"
            % (np.mean(restored_power), np.std(restored_power))
        )


def num_to_twetri(act):
    """Decode one integer into five base-23 action components."""
    res = []
    ac = int(act)
    while ac > 0:
        res.append(ac % 23)
        ac //= 23
    if len(res) < 5:
        res.extend([0] * (5 - len(res)))
    return res[:5]


def twetri_to_num(action):
    """Encode five base-23 action components into one vocabulary index."""
    return int(sum(int(action[i]) * (23 ** i) for i in range(len(action))))
