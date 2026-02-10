import json
import logging
import multiprocessing as mp
import os
import threading
import time
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict, dataclass, field
import datetime as dt
from pathlib import Path
from pprint import pformat
from typing import Callable, TypedDict
import importlib

# Register VLASH policy configs (pi0/pi05) into LeRobot's config registry
# so `PreTrainedConfig.from_pretrained()` can decode VLASH config.json.
import vlash.configs  # noqa: F401
# Monkey patch for modified transformers library that removed LossKwargs
try:
    from transformers.utils import LossKwargs
except ImportError:
    # Create a placeholder LossKwargs if it doesn't exist
    class LossKwargs(TypedDict, total=False):
        """Placeholder for LossKwargs type hint."""
        pass
    
    # Inject it into transformers.utils
    import transformers.utils
    transformers.utils.LossKwargs = LossKwargs


@dataclass(frozen=True)
class VLASHMethodConfig:
    """VLASH method: use delayed environment observation + current robot state"""
    pass

import einops
import gymnasium as gym
import numpy as np
import torch
from termcolor import colored
from torch import Tensor, nn
from tqdm import tqdm, trange

from lerobot.configs import parser
from lerobot import envs, policies  # noqa: F401
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.factory import make_env
from lerobot.envs.utils import add_envs_task, check_env_attributes_and_types, preprocess_observation
from vlash.policies.factory import make_policy
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import get_device_from_parameters
from lerobot.utils.io_utils import write_video
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import (
    get_safe_torch_device,
    init_logging,
    inside_slurm,
)

from libero.libero.envs import SubprocVectorEnv
from libero.libero import benchmark, get_libero_path
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata


@dataclass
class DatasetConfigForEval:
    """Simplified dataset config for evaluation (all fields optional)."""
    repo_id: str | None = None
    root: str | None = None
    norm_stats_path: str | None = None


@dataclass
class EvalConfig:
    """Eval config extended with nano-lerobot-libero LIBERO eval knobs."""

    n_episodes: int = 50
    batch_size: int = 50
    use_async_envs: bool = False

    # nano-lerobot-libero additions (used by LIBERO eval harness)
    async_delay: int = 0
    action_quant: int = 1
    method_type: str = "vlash"  # VLASH-only

    def __post_init__(self) -> None:
        if self.batch_size > self.n_episodes:
            raise ValueError(
                "The eval batch size is greater than the number of eval episodes "
                f"({self.batch_size} > {self.n_episodes}). As a result, {self.batch_size} "
                f"eval environments will be instantiated, but only {self.n_episodes} will be used. "
                "This might significantly slow down evaluation. To fix this, you should update your command "
                f"to increase the number of episodes to match the batch size (e.g. `eval.n_episodes={self.batch_size}`), "
                f"or lower the batch size (e.g. `eval.batch_size={self.n_episodes}`)."
            )
        if self.method_type != "vlash":
            raise ValueError(
                f"vlash.eval_libero only supports eval.method_type='vlash' (got {self.method_type!r})."
            )


@dataclass
class EvalPipelineConfig:
    """Eval config compatible with nano-lerobot-libero (includes multi-GPU fields)."""

    env: envs.EnvConfig
    eval: EvalConfig = field(default_factory=EvalConfig)
    policy: PreTrainedConfig | None = None
    dataset: DatasetConfigForEval = field(default_factory=DatasetConfigForEval)
    output_dir: Path | None = None
    job_name: str | None = None
    seed: int | None = 1000
    task_description: str | None = ""
    gpu_id: int = 0
    num_gpus: int = 1

    def __post_init__(self):
        # HACK: Parse again the cli args here to get the pretrained path if there was one.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path
        else:
            logging.warning(
                "No pretrained path was provided, evaluated policy will be built from scratch (random weights)."
            )

        if not self.job_name:
            if self.env is None:
                self.job_name = f"{self.policy.type}"
            else:
                self.job_name = f"{self.env.type}_{self.policy.type}"

        if not self.output_dir:
            now = dt.datetime.now()
            eval_dir = f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{self.job_name}"
            self.output_dir = Path("outputs/eval") / eval_dir

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]


def make_libero_env(suite_name: str, task_name: str, init_state_id: int, gym_kwargs: dict):
    # NOTE: We intentionally do NOT depend on the external `gym_libero` package.
    # Instead, we construct the LIBERO env directly via `vlash.libero_gym`.
    from vlash.libero_gym import make_env_from_suite_task

    max_episode_steps = int(gym_kwargs.get("max_episode_steps", 530))
    camera_heights = int(gym_kwargs.get("camera_heights", gym_kwargs.get("camera_height", 256)))
    camera_widths = int(gym_kwargs.get("camera_widths", gym_kwargs.get("camera_width", 256)))

    return make_env_from_suite_task(
        suite_name=suite_name,
        task_name=task_name,
        init_state_id=init_state_id,
        max_episode_steps=max_episode_steps,
        camera_heights=camera_heights,
        camera_widths=camera_widths,
    )

def schedule_envs(task_suite, task_ids, batch_size):
    n_tasks = len(task_ids)
    
    assert 500 >= batch_size > 0
    assert n_tasks > 0
    
    if batch_size < n_tasks:
        n_batches = n_tasks // batch_size + (n_tasks % batch_size > 0)
        
        envs = []
        for i in range(n_batches):
            batch_tasks = task_ids[i * batch_size: (i + 1) * batch_size]
            batch_envs = []
            for task_id in batch_tasks:
                task = task_suite.get_task(task_id)
                n_episodes = len(task_suite.get_task_init_states(task_id))
                batch_envs.append((task.name, task.language, list(range(n_episodes)))) # task name, start ep, end ep
            envs.append(batch_envs)
        
        return envs
            
    elif batch_size >= n_tasks:
        n_task_replicas, rem = divmod(batch_size, n_tasks)
        
        envs = []
        for task_id in task_ids:
            task = task_suite.get_task(task_id)
            init_states = task_suite.get_task_init_states(task_id)
            n_episodes = len(init_states)
            
            n_replicas = n_task_replicas
            if rem > 0:
                n_replicas += 1
                rem -= 1
            
            n_ep_per_replica, ep_rem = divmod(n_episodes, n_replicas)
            # divide episode remainder to the first few replicas
            
            start_ep = 0
            for r in range(n_replicas):
                end_ep = start_ep + n_ep_per_replica + (1 if r < ep_rem else 0)
                envs.append((task.name, task.language, list(range(start_ep, end_ep))))
                start_ep = end_ep
        
        return [envs]
    
    return []        

def merge_observation(obs_list):
    result = {}
    n_envs = len(obs_list)
    for k, v in obs_list[0].items():
        if k == "task_description":
            continue
        if isinstance(v, dict):
            result[k] = merge_observation([obs_list[i][k] for i in range(n_envs)])
        else:
            result[k] = np.stack([obs_list[i][k] for i in range(n_envs)], axis=0)
    
    return result

def rollout(
    env,
    policy,
    seeds = None,
    return_observations = False,
    task_description = None,
    max_steps = 520,
    render_callback = None,
    async_delay: int = 0,
    method: VLASHMethodConfig = VLASHMethodConfig(),
    action_quant: int = 1,
):
    device = get_device_from_parameters(policy)
    
    # Validate that n_action_steps is divisible by action_quant
    if hasattr(policy, 'config') and hasattr(policy.config, 'n_action_steps'):
        n_action_steps = policy.config.n_action_steps
        assert n_action_steps % action_quant == 0, (
            f"n_action_steps ({n_action_steps}) must be divisible by action_quant ({action_quant}). "
            f"Current remainder: {n_action_steps % action_quant}"
        )
    
    # Log rollout start
    print(f"[DEBUG] rollout starting: method={method.__class__.__name__}, async_delay={async_delay}, action_quant={action_quant}, n_envs={len(env)}")
    
    policy.reset()
    observation, info = env.reset(seed=seeds)
    if render_callback is not None:
        render_callback(env)
    
    all_observations = []
    all_actions = []
    all_rewards = []
    all_successes = []
    all_dones = []
    
    done = np.array([False] * len(env))
    
    # Buffer to store previous observations for async_delay
    observation_buffer = []  # List of observation dicts (merged and preprocessed)
    
    # Debug counter
    loop_iteration = 0
    
    while not np.all(done):
        # Process current observation once
        observation = merge_observation(observation)
        observation = preprocess_observation(observation)
        if return_observations:
            all_observations.append(deepcopy(observation))
        
        # Store current observation in buffer (before converting to tensors)
        observation_buffer.append(deepcopy(observation))
        
        # Convert to tensors
        observation = {
            key: observation[key].to(device, non_blocking=device.type == "cuda") for key in observation
        }
        
        # Apply VLASH async_delay logic: delayed env observation + CURRENT robot state
        if async_delay > 0 and len(observation_buffer) > async_delay:
            delay_index = -(async_delay + 1)  # Get observation from async_delay steps ago
            delayed_observation = observation_buffer[delay_index]

            for key in observation.keys():
                if key != "observation.state" and key in delayed_observation:
                    observation[key] = delayed_observation[key].to(device)
        
        observation["task"] = task_description
        
        # Collect action_quant actions from the policy and merge them
        # All actions are taken from the same observation (same time step)
        actions_to_merge = []
        for i in range(action_quant):
            with torch.inference_mode():
                action = policy.select_action(observation)
            
            action = action.to("cpu").numpy()
            assert action.ndim == 2, "Action dimensions should be (batch, action_dim)"
            actions_to_merge.append(action)
        
        # Debug output (only for first loop iteration)
        if loop_iteration == 0:
            print(f"[DEBUG] Loop {loop_iteration}: Collected {len(actions_to_merge)} actions to merge (action_quant={action_quant})")
            print(f"[DEBUG] Each action shape: {actions_to_merge[0].shape}")
        
        # Merge actions by summing (for delta action spaces like LIBERO)
        merged_action = np.sum(actions_to_merge, axis=0)
        
        if loop_iteration == 0:
            print(f"[DEBUG] Merged action shape: {merged_action.shape}")
        
        # Execute the merged action once
        observation, reward, terminated, truncated, info = env.step(merged_action)
        if render_callback is not None:
            render_callback(env)
        
        successes = [info[i]["is_success"] for i in range(len(info))]
        
        done = terminated | truncated | done
        
        # Store the merged action and results (one entry per loop iteration)
        all_actions.append(torch.from_numpy(merged_action))
        all_rewards.append(torch.from_numpy(reward))
        all_dones.append(torch.from_numpy(done))
        all_successes.append(torch.tensor(successes))
        
        loop_iteration += 1
    
    if return_observations:
        observation = preprocess_observation(observation)
        all_observations.append(deepcopy(observation))
    
    # Debug output at end of rollout
    print(f"[DEBUG] Rollout completed: action_quant={action_quant}, total loop iterations={loop_iteration}")
    print(f"[DEBUG] len(all_actions)={len(all_actions)}, len(all_dones)={len(all_dones)}")
    
    ret = {
        "action": torch.stack(all_actions, dim=1),
        "reward": torch.stack(all_rewards, dim=1),
        "success": torch.stack(all_successes, dim=1),
        "done": torch.stack(all_dones, dim=1),
    }
    
    print(f"[DEBUG] Returned action shape: {ret['action'].shape}, done shape: {ret['done'].shape}")
    
    if return_observations:
        stacked_observations = {}
        for key in all_observations[0]:
            stacked_observations[key] = torch.stack([obs[key] for obs in all_observations], dim=1)
        ret["observation"] = stacked_observations

    if hasattr(policy, "use_original_modules"):
        policy.use_original_modules()

    return ret

def eval_policy(
    env_cfg,
    policy,
    schedule,
    start_seed = None,
    task_description = None,
    max_steps = 520,
    max_episodes_rendered = 0,
    videos_dir: Path | None = None,
    progress_callback = None,
    gpu_id = None,
    episode_offset = 0,
    async_delay: int = 0,
    method: VLASHMethodConfig = VLASHMethodConfig(),
    action_quant: int = 1,
):  
    start = time.time()
    policy.eval()
    
    # Log method configuration for debugging
    print(f"eval_policy starting: method=VLASH, async_delay={async_delay}, action_quant={action_quant}")
    
    gym_kwargs = env_cfg.gym_kwargs
    gym_kwargs["max_episode_steps"] = max_steps
    
    # Disable local progress bars if using callback (multi-GPU mode)
    show_progress = progress_callback is None
    
    total_episodes = sum(sum(len(episodes) for _, _, episodes in batch) for batch in schedule)
    
    progbar = trange(
        len(schedule), 
        desc=f"Eval batches", 
        disable=not show_progress,
        dynamic_ncols=True,
        position=0,
    )

    info = {"overall":
        {
            "avg_sum_rewards": [],
            "avg_max_rewards": [],
            "pc_successes": [],
            "avg_episode_length": [],  # Track episode lengths (simulation timesteps)
        }
    }
    descriptions = []
    # Track how many episodes have been rendered per task
    rendered_count_by_task: dict[str, int] = {}
    video_paths_by_task: dict[str, list[str]] = {}
    # Global episode counter for video numbering (starts from episode_offset)
    global_episode_counter = episode_offset
    
    for batch_idx in progbar:
        env_lst = []
        max_n_rollouts = 0
        
        env_limits = []
        env_names = []
        # Pre-assign global episode indices for this batch
        # This maps (env_idx, episode_idx) -> global_episode_number
        batch_episode_map = {}
        
        for env_idx, (task_name, task_language, episodes) in enumerate(schedule[batch_idx]):
            env_lst.append(
                lambda suite_name=env_cfg.task, task_name=task_name, episodes=episodes:
                make_libero_env(suite_name, task_name, episodes[0], gym_kwargs)
            )
            
            env_names.append(task_name)
            
            if task_name not in info:
                info[task_name] = {
                    "avg_sum_rewards": [],
                    "avg_max_rewards": [],
                    "pc_successes": [],
                    "avg_episode_length": [],
                }
            
            descriptions.append(task_language)
            
            env_limits.append(len(episodes))
            max_n_rollouts = max(max_n_rollouts, len(episodes))
            
            # Pre-assign episode indices for this env
            for ep_idx in range(len(episodes)):
                batch_episode_map[(env_idx, ep_idx)] = global_episode_counter
                global_episode_counter += 1
        
        # Create environment with timeout protection
        print(f"[DEBUG] Creating SubprocVectorEnv with {len(env_lst)} environments...")
        import sys
        sys.stdout.flush()  # Force flush before potentially blocking operation
        
        try:
            env = SubprocVectorEnv(env_lst)
            print(f"[DEBUG] SubprocVectorEnv created successfully")
            sys.stdout.flush()
        except Exception as e:
            print(f"[ERROR] Failed to create SubprocVectorEnv: {e}")
            import traceback
            traceback.print_exc()
            raise
        env_limits = np.asarray(env_limits)
        # Determine FPS if available; otherwise default
        try:
            # Try to get metadata from first sub-env
            metas = env.get_env_attr("metadata")
            render_fps = metas[0].get("render_fps", 30) if metas and metas[0] else 30
        except Exception:
            render_fps = 30

        if start_seed is None:
            seeds = None
        else:
            seeds = range(
                start_seed + (batch_idx * len(env_lst)), start_seed + ((batch_idx + 1) * len(env_lst))
            )
        
        for episode in range(max_n_rollouts):
            # Prepare frame buffer if rendering videos is requested
            if max_episodes_rendered > 0 and videos_dir is not None:
                ep_frames: list[np.ndarray] = []
                # Render callback compatible with vectorized subproc envs
                def render_frame(e):
                    frames = e.render()
                    # Stack into (b, h, w, c)
                    ep_frames.append(np.stack(frames))
            else:
                render_frame = None
            rollout_data = rollout(
                env,
                policy,
                seeds=list(seeds) if seeds else None,
                task_description=task_description if task_description else descriptions,
                max_steps=max_steps,
                render_callback=render_frame,
                async_delay=async_delay,
                method=method,
                action_quant=action_quant,
            )

            n_steps = rollout_data["action"].shape[1]
            done_indices = torch.argmax(rollout_data["done"].to(int), dim=1)
            
            mask = (torch.arange(n_steps) <= einops.repeat(done_indices + 1, "b -> b s", s=n_steps)).int()
            batch_sum_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "sum")
            batch_max_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "max")
            batch_successes = einops.reduce((rollout_data["success"] * mask), "b n -> b", "any")
            
            for env_idx in range(len(env)):
                if episode >= env_limits[env_idx]:
                    continue
                
                task_name = env_names[env_idx]
                if task_name not in rendered_count_by_task:
                    rendered_count_by_task[task_name] = 0
                if task_name not in video_paths_by_task:
                    video_paths_by_task[task_name] = []
                
                # Calculate episode length (number of timesteps)
                episode_length = (done_indices[env_idx] + 1).item()
                
                info[task_name]["avg_sum_rewards"].append(batch_sum_rewards[env_idx].item())
                info[task_name]["avg_max_rewards"].append(batch_max_rewards[env_idx].item())
                info[task_name]["pc_successes"].append(batch_successes[env_idx].item())
                info[task_name]["avg_episode_length"].append(episode_length)
                
                info["overall"]["avg_sum_rewards"].append(batch_sum_rewards[env_idx].item())
                info["overall"]["avg_max_rewards"].append(batch_max_rewards[env_idx].item())
                info["overall"]["pc_successes"].append(batch_successes[env_idx].item())
                info["overall"]["avg_episode_length"].append(episode_length)
                
                # Report progress to callback if provided (for multi-GPU coordination)
                if progress_callback is not None:
                    # Send both count and success status
                    success_status = batch_successes[env_idx].item()
                    progress_callback(1, success_status)
            
            # Update progress bar with current accuracy
            if info["overall"]["pc_successes"] and show_progress:
                current_acc = np.mean(info["overall"]["pc_successes"]) * 100
                n_episodes_done = len(info["overall"]["pc_successes"])
                progbar.set_description(f"Eval batches (Acc: {current_acc:.1f}%, {n_episodes_done} eps)")
                progbar.refresh()

            # Save videos for this rollout if requested
            if max_episodes_rendered > 0 and videos_dir is not None and 'ep_frames' in locals() and len(ep_frames) > 0:
                batch_stacked_frames = np.stack(ep_frames, axis=1)  # (b, t, h, w, c)
                for env_idx in range(len(env)):
                    if episode >= env_limits[env_idx]:
                        continue
                    task_name = env_names[env_idx]
                    if rendered_count_by_task[task_name] >= max_episodes_rendered:
                        continue
                    done_index = done_indices[env_idx].item()
                    task_videos_dir = Path(videos_dir) / task_name
                    task_videos_dir.mkdir(parents=True, exist_ok=True)
                    status = "success" if bool(batch_successes[env_idx].item()) else "fail"
                    # Use pre-assigned episode index from the map
                    global_ep_idx = batch_episode_map[(env_idx, episode)]
                    video_filename = f"eval_episode_{global_ep_idx}_{status}.mp4"
                    video_path = task_videos_dir / video_filename
                    # Spawn a thread to write video
                    frames_to_save = batch_stacked_frames[env_idx, : done_index + 1]
                    thread = threading.Thread(
                        target=write_video,
                        args=(
                            str(video_path),
                            frames_to_save,
                            render_fps,
                        ),
                    )
                    thread.start()
                    thread.join()
                    video_paths_by_task[task_name].append(str(video_path))
                    rendered_count_by_task[task_name] += 1
    
    for key in info:
        info[key]["avg_sum_rewards"] = float(np.nanmean(info[key]["avg_sum_rewards"]))
        info[key]["avg_max_rewards"] = float(np.nanmean(info[key]["avg_max_rewards"]))
        info[key]["pc_successes"] = float(np.nanmean(info[key]["pc_successes"]) * 100)
        info[key]["avg_episode_length"] = float(np.nanmean(info[key]["avg_episode_length"]))
    
    info["eval_s"] = time.time() - start
    
    # Attach video paths if any
    if any(len(v) > 0 for v in video_paths_by_task.values()):
        info["video_paths"] = video_paths_by_task
    
    return info
    

@parser.wrap()
def eval_main(cfg: EvalPipelineConfig):
    max_step_dict = {
        "libero_spatial": 230,
        "libero_object": 290,
        "libero_goal": 310,
        "libero_10": 530,
        "libero_90": 410
    }
    
    logging.info(pformat(asdict(cfg)))
    device = get_safe_torch_device(cfg.policy.device, log=True)
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    
    # We do not require the external `gym_libero` package; env construction is integrated
    # in `vlash.libero_gym`.

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.env.task]()
    task_ids = list(range(task_suite.n_tasks))
    
    logging.info(f"Evaluating {len(task_ids)} tasks: {task_ids}")
    
    # Multi-GPU support: split episodes across GPUs
    if cfg.num_gpus > 1:
        # In multi-GPU mode, we need to distribute episodes evenly across GPUs
        # First, calculate how many episodes each task should contribute per GPU
        n_episodes_per_gpu = cfg.eval.n_episodes // cfg.num_gpus
        remainder_episodes = cfg.eval.n_episodes % cfg.num_gpus
        
        # This GPU gets base episodes plus one extra if it's in the remainder
        if cfg.gpu_id < remainder_episodes:
            gpu_n_episodes = n_episodes_per_gpu + 1
            start_episode = cfg.gpu_id * (n_episodes_per_gpu + 1)
        else:
            gpu_n_episodes = n_episodes_per_gpu
            start_episode = cfg.gpu_id * n_episodes_per_gpu + remainder_episodes
        
        # Store start_episode for video numbering
        cfg.start_episode = start_episode
        
        # Now distribute these episodes across tasks
        # Each task gets proportional episodes based on total available init states
        task_episodes = []
        for task_id in task_ids:
            task = task_suite.get_task(task_id)
            init_states = task_suite.get_task_init_states(task_id)
            n_init_states = len(init_states)
            
            # Calculate how many episodes this task gets for this GPU
            # We distribute proportionally based on available init states
            task_n_episodes = max(1, gpu_n_episodes // len(task_ids))
            
            # Determine which init state episodes this GPU handles
            start_idx = start_episode % n_init_states
            episodes_for_task = []
            for i in range(task_n_episodes):
                episodes_for_task.append((start_idx + i) % n_init_states)
            
            task_episodes.append((task.name, task.language, episodes_for_task))
        
        # Create schedule with one batch containing all tasks for this GPU
        schedule = [task_episodes]
        
        total_episodes = sum(len(episodes) for _, _, episodes in task_episodes)
        logging.info(
            colored(f"Multi-GPU mode: ", "cyan", attrs=["bold"]) +
            f"GPU {cfg.gpu_id}/{cfg.num_gpus} processing {total_episodes} episodes "
            f"(episode range: {start_episode}-{start_episode + total_episodes - 1})"
        )
    else:
        # Single GPU mode: use original scheduling
        schedule = schedule_envs(task_suite, task_ids, cfg.eval.batch_size)
    
    logging.info("Making policy...")
    policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env)
    policy.eval()
    
    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        # Get progress callback if it exists (used by multi-GPU setup)
        progress_callback = getattr(cfg, '_progress_callback', None)
        gpu_id = getattr(cfg, 'gpu_id', None)
        episode_offset = getattr(cfg, 'start_episode', 0)
        
        # VLASH-only method selection
        method = getattr(cfg.eval, "method", None)
        if method is None:
            method_type = getattr(cfg.eval, "method_type", "vlash")
            if method_type != "vlash":
                raise ValueError(
                    f"vlash.eval_libero only supports eval.method_type='vlash' (got {method_type!r})."
                )
            method = VLASHMethodConfig()
        elif not isinstance(method, VLASHMethodConfig):
            raise ValueError(
                f"vlash.eval_libero only supports VLASHMethodConfig (got {type(method).__name__})."
            )
        
        # Get async_delay and action_quant from config
        async_delay = getattr(cfg.eval, 'async_delay', 0)
        action_quant = getattr(cfg.eval, 'action_quant', 1)
        
        info = eval_policy(
            env_cfg=cfg.env,
            policy=policy,
            schedule=schedule,
            start_seed=cfg.seed,
            task_description=cfg.task_description,
            max_steps=max_step_dict[cfg.env.task],
            max_episodes_rendered=50,
            videos_dir=Path(cfg.output_dir) / "videos",
            progress_callback=progress_callback,
            gpu_id=gpu_id,
            episode_offset=episode_offset,
            async_delay=async_delay,
            method=method,
            action_quant=action_quant,
        )
    
    print(info)
    
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Only save eval_results.json in single-GPU mode
    # In multi-GPU mode, only the merged_results.json is saved by the main process
    is_multi_gpu_worker = hasattr(cfg, 'gpu_id') and hasattr(cfg, 'num_gpus') and cfg.num_gpus > 1
    if not is_multi_gpu_worker:
        # Reorganize to put 'overall' first
        ordered_info = {}
        if 'overall' in info:
            ordered_info['overall'] = info['overall']
        for key in sorted(info.keys()):
            if key not in ['overall', 'eval_s', 'video_paths']:
                ordered_info[key] = info[key]
        # Add metadata at the end
        if 'eval_s' in info:
            ordered_info['eval_s'] = info['eval_s']
        if 'video_paths' in info:
            ordered_info['video_paths'] = info['video_paths']
        
        with open(Path(cfg.output_dir) / "eval_results.json", "w") as f:
            json.dump(ordered_info, f, indent=2)
    
    logging.info("End of eval")
    
    return info  # Return results for multi-GPU aggregation


def worker_process(gpu_id, num_gpus, cfg_dict, result_queue, progress_queue):
    """Worker process for a single GPU in multi-GPU mode."""
    import sys
    
    # IMPORTANT: Set environment variables BEFORE any imports that might use them
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.environ['MUJOCO_GL'] = 'egl'
    # Disable tokenizers parallelism to avoid fork warnings
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    # Also explicitly select the CUDA device. In some environments, relying on
    # CUDA_VISIBLE_DEVICES alone can still result in all workers defaulting to cuda:0.
    try:
        import torch

        if torch.cuda.is_available():
            # Use physical GPU id (works even if CUDA_VISIBLE_DEVICES doesn't take effect as expected).
            torch.cuda.set_device(gpu_id)
    except Exception:
        # If torch is unavailable or CUDA init fails, let the worker error later with clearer logs.
        pass
    
    # Redirect stdout/stderr to avoid cluttering main process
    # Keep logs in separate GPU directories for debugging
    log_dir = Path(cfg_dict['output_dir']) / "logs" / f"gpu_{gpu_id}"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = open(log_dir / "worker.log", 'w', buffering=1)
    sys.stdout = log_file
    sys.stderr = log_file
    
    try:
        print(f"GPU {gpu_id} worker starting...")
        
        # Reconstruct config - need to handle nested dataclasses
        # Import configs from this module (keeps behavior consistent between main and workers)
        from vlash.eval_libero import EvalPipelineConfig, DatasetConfigForEval, EvalConfig
        # NOTE: Some lerobot versions do not ship all env config classes (e.g. CalvinEnv).
        # This eval script is LIBERO-specific, so we only need LiberoEnv here.
        from lerobot.envs.configs import LiberoEnv
        
        # Extract configs
        policy_dict = cfg_dict.get('policy', {})
        policy_type = cfg_dict.get('policy_type', 'pi0')
        dataset_dict = cfg_dict.get('dataset', {})
        eval_dict = cfg_dict.get('eval', {})
        env_dict = cfg_dict.get('env', {})
        output_dir_str = cfg_dict.get('output_dir', 'outputs/eval')
        seed = cfg_dict.get('seed', 100)
        task_description = cfg_dict.get('task_description', None)
        
        # Dynamically import the correct policy config class based on policy type.
        # IMPORTANT: Use VLASH policy config classes here (pi0/pi05) so worker reconstruction
        # matches the VLASH checkpoint config fields.
        if policy_type == "pi05":
            from vlash.policies.pi05 import PI05Config

            PolicyConfigClass = PI05Config
        elif policy_type == "pi0":
            from vlash.policies.pi0 import PI0Config

            PolicyConfigClass = PI0Config
        else:
            # Keep compatibility with other policy types if present; fallback to pi0.
            from vlash.policies.pi0 import PI0Config

            PolicyConfigClass = PI0Config
            print(f"Warning: Unknown policy type '{policy_type}', falling back to PI0Config")
        
        # Reconstruct config objects
        policy_cfg = PolicyConfigClass(**policy_dict) if policy_dict else None
        dataset_cfg = DatasetConfigForEval(**dataset_dict) if dataset_dict else None
        eval_cfg = EvalConfig(**eval_dict)
        
        # Get env type and create env config
        import copy
        env_dict_copy = copy.deepcopy(env_dict)
        env_type = env_dict_copy.pop('type', 'libero')
        
        # Remove features and features_map - they will be recreated in __post_init__
        env_dict_copy.pop('features', None)
        env_dict_copy.pop('features_map', None)
        
        if env_type != "libero":
            raise ValueError(
                f"vlash.eval_libero worker only supports env.type='libero' but got env.type={env_type!r}"
            )
        env_cfg = LiberoEnv(**env_dict_copy)
        
        # Create main config
        cfg = EvalPipelineConfig(
            env=env_cfg,
            eval=eval_cfg,
            policy=policy_cfg,
            dataset=dataset_cfg,
            output_dir=output_dir_str,
            seed=seed,
            task_description=task_description,
        )
        
        cfg.gpu_id = gpu_id
        cfg.num_gpus = num_gpus
        # Use shared output directory for all GPUs
        cfg.output_dir = output_dir_str
        if cfg.policy:
            cfg.policy.device = "cuda"
        
        print(f"GPU {gpu_id}: Config reconstructed")
        print(f"GPU {gpu_id}: Task = {cfg.env.task}")
        print(f"GPU {gpu_id}: Output = {cfg.output_dir}")
        print(f"GPU {gpu_id}: Method = {getattr(cfg.eval, 'method_type', 'unknown')}")
        print(f"GPU {gpu_id}: Async delay = {getattr(cfg.eval, 'async_delay', 0)}")
        print(f"GPU {gpu_id}: Action quant = {getattr(cfg.eval, 'action_quant', 1)}")
        
        # Create a progress callback to send updates to main process
        def progress_callback(n_episodes, success_status=None):
            """Send progress update to main process."""
            try:
                if success_status is not None:
                    progress_queue.put(('progress', gpu_id, n_episodes, success_status))
                else:
                    progress_queue.put(('progress', gpu_id, n_episodes, None))
            except:
                pass  # Queue might be closed, ignore
        
        # Monkey-patch the callback into cfg so eval_main can use it
        cfg._progress_callback = progress_callback
        
        print(f"GPU {gpu_id}: Starting evaluation...")
        
        # Run evaluation.
        # IMPORTANT: `eval_main` is decorated with `@parser.wrap()`, which uses a strict
        # `type(args[0]) is argtype` check and can mis-handle cfg objects reconstructed
        # across processes. Call the undecorated function directly.
        result_info = eval_main.__wrapped__(cfg)
        
        print(f"GPU {gpu_id}: Evaluation complete!")
        
        # Send results
        result_queue.put(('success', gpu_id, result_info))
        
    except Exception as e:
        print(f"GPU {gpu_id}: Error occurred - {str(e)}")
        result_queue.put(('error', gpu_id, str(e)))
        import traceback
        traceback.print_exc()
    finally:
        log_file.close()


def merge_results(results_dict):
    """Merge results from multiple GPUs."""
    merged = {}
    
    # Collect all unique keys
    all_keys = set()
    for result in results_dict.values():
        if result is not None:
            all_keys.update(result.keys())
    
    # Remove special keys
    all_keys.discard('eval_s')
    all_keys.discard('video_paths')
    
    # Merge each key
    for key in all_keys:
        values = []
        for gpu_id, result in results_dict.items():
            if result is not None and key in result:
                # If it's already aggregated (single value), just collect it
                if isinstance(result[key], dict):
                    values.append(result[key])
        
        if values:
            # Merge dictionaries
            if all(isinstance(v, dict) for v in values):
                merged[key] = {}
                for metric in ['avg_sum_rewards', 'avg_max_rewards', 'pc_successes', 'avg_episode_length']:
                    if metric in values[0]:
                        # Average across GPUs
                        merged[key][metric] = float(np.mean([v[metric] for v in values]))
    
    return merged


@parser.wrap()
def main(cfg: EvalPipelineConfig):
    """Main entry point supporting both single-GPU and multi-GPU modes."""
    
    # Check if multi-GPU mode is requested
    num_gpus = getattr(cfg, 'num_gpus', 1)
    num_gpus = min(num_gpus, torch.cuda.device_count()) if num_gpus > 1 else 1
    
    if num_gpus <= 1:
        # Single GPU mode - run directly
        logging.info("Running in single-GPU mode")
        eval_main(cfg)
        return
    
    # Multi-GPU mode
    logging.info(colored(f"\n{'='*60}", "cyan", attrs=["bold"]))
    logging.info(colored(f"Multi-GPU Evaluation", "cyan", attrs=["bold"]))
    logging.info(colored(f"{'='*60}", "cyan", attrs=["bold"]))
    logging.info(f"Model: {cfg.policy.path if hasattr(cfg.policy, 'path') else 'N/A'}")
    logging.info(f"Task: {cfg.env.task}")
    logging.info(f"GPUs: {num_gpus}")
    logging.info(f"Episodes per GPU: ~{cfg.eval.n_episodes // num_gpus}")
    logging.info(f"Total Episodes: {cfg.eval.n_episodes}")
    logging.info(f"Batch size: {cfg.eval.batch_size}")
    logging.info(colored(f"{'='*60}\n", "cyan", attrs=["bold"]))
    
    # Create output directory
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert config to dict for passing to workers
    # We need to manually build the dict to ensure proper serialization
    cfg_dict = {
        'policy': asdict(cfg.policy) if cfg.policy else {},
        'policy_type': cfg.policy.type if cfg.policy and hasattr(cfg.policy, 'type') else 'pi0',
        'dataset': asdict(cfg.dataset) if cfg.dataset else {},
        'eval': asdict(cfg.eval),
        'env': asdict(cfg.env),
        'output_dir': str(cfg.output_dir),
        'seed': cfg.seed if hasattr(cfg, 'seed') else 100,
        'task_description': cfg.task_description if hasattr(cfg, 'task_description') else None,
    }
    
    # Create queues for results and progress
    result_queue = mp.Queue()
    progress_queue = mp.Queue()
    
    # Start worker processes
    processes = []
    for gpu_id in range(num_gpus):
        p = mp.Process(
            target=worker_process,
            args=(gpu_id, num_gpus, cfg_dict, result_queue, progress_queue)
        )
        p.start()
        processes.append(p)
        logging.info(f"Started GPU {gpu_id} (PID: {p.pid})")
    
    # Monitor progress with unified progress bar
    print("\n")
    total_episodes = cfg.eval.n_episodes
    pbar = tqdm(
        total=total_episodes,
        desc="Overall Progress",
        unit="episode",
        position=0,
        colour="green",
        dynamic_ncols=True
    )
    
    # Track per-GPU progress
    gpu_progress = {i: 0 for i in range(num_gpus)}
    gpu_status = {i: "running" for i in range(num_gpus)}
    
    # Collect results and real-time success data
    results = {}
    completed = 0
    all_successes_realtime = []  # Track success status in real-time
    
    start_time = time.time()
    
    # Monitor until all processes complete
    while completed < num_gpus:
        # Check for progress updates
        while not progress_queue.empty():
            msg = progress_queue.get()
            if msg[0] == 'progress':
                msg_type, gpu_id, value = msg[0], msg[1], msg[2]
                success_status = msg[3] if len(msg) > 3 else None
                
                gpu_progress[gpu_id] += value
                pbar.update(value)
                
                # Track success status if provided
                if success_status is not None:
                    all_successes_realtime.append(success_status)
        
        # Check for results
        while not result_queue.empty():
            msg_type, gpu_id, data = result_queue.get()
            
            if msg_type == 'success':
                results[gpu_id] = data
                gpu_status[gpu_id] = "done"
                completed += 1
                logging.info(colored(f"GPU {gpu_id} completed successfully", "green"))
                
            elif msg_type == 'error':
                gpu_status[gpu_id] = f"error"
                completed += 1
                logging.error(colored(f"GPU {gpu_id} failed: {data}", "red"))
        
        # Check for dead processes (crashed without sending error message)
        for i, p in enumerate(processes):
            if gpu_status[i] == "running" and not p.is_alive():
                logging.error(colored(f"GPU {i} process died unexpectedly (exit code: {p.exitcode})", "red"))
                gpu_status[i] = "crashed"
                completed += 1
                # Try to get any remaining messages from the log
                log_file = Path(cfg.output_dir) / "logs" / f"gpu_{i}" / "worker.log"
                if log_file.exists():
                    logging.error(f"Check log file for details: {log_file}")
        
        # Calculate and display current average accuracy from real-time data
        if all_successes_realtime:
            current_acc = np.mean(all_successes_realtime) * 100
            n_episodes_done = len(all_successes_realtime)
            pbar.set_description(f"Overall Progress (Acc: {current_acc:.1f}%, {n_episodes_done}/{total_episodes} eps)")
            pbar.refresh()
        
        time.sleep(0.1)
    
    pbar.close()
    
    # Wait for all processes to finish
    for p in processes:
        p.join()
    
    elapsed = time.time() - start_time
    
    print("\n")
    logging.info(colored(f"{'='*60}", "cyan", attrs=["bold"]))
    
    # Count successful vs failed GPUs
    successful_gpus = [i for i, status in gpu_status.items() if status == "done"]
    failed_gpus = [i for i, status in gpu_status.items() if status in ["error", "crashed"]]
    
    if len(successful_gpus) == num_gpus:
        logging.info(colored(f"Evaluation Complete!", "cyan", attrs=["bold"]))
    else:
        logging.warning(colored(f"Evaluation Completed with Failures!", "yellow", attrs=["bold"]))
        logging.warning(f"Successful GPUs: {successful_gpus}")
        logging.warning(f"Failed GPUs: {failed_gpus}")
    
    logging.info(colored(f"{'='*60}", "cyan", attrs=["bold"]))
    logging.info(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    if elapsed > 0:
        actual_episodes = sum(gpu_progress.values())
        logging.info(f"Completed episodes: {actual_episodes}/{total_episodes}")
        logging.info(f"Throughput: {actual_episodes/elapsed:.2f} episodes/second")
    
    # Merge results
    if results:
        logging.info("\nMerging results from all GPUs...")
        merged_results = merge_results(results)
        
        # Reorganize to put 'overall' first, tasks in the middle, metadata at the end
        ordered_results = {}
        if 'overall' in merged_results:
            ordered_results['overall'] = merged_results['overall']
        for key in sorted(merged_results.keys()):
            if key not in ['overall', 'eval_s', 'video_paths']:
                ordered_results[key] = merged_results[key]
        # Add metadata at the end
        if 'eval_s' in merged_results:
            ordered_results['eval_s'] = merged_results['eval_s']
        if 'video_paths' in merged_results:
            ordered_results['video_paths'] = merged_results['video_paths']
        
        # Save merged results
        output_file = output_dir / "merged_results.json"
        with open(output_file, 'w') as f:
            json.dump(ordered_results, f, indent=2)
        
        logging.info(f"Results saved to: {output_file}")
        
        # Print summary
        print("\n" + colored("="*60, "cyan", attrs=["bold"]))
        print(colored("Final Results:", "cyan", attrs=["bold"]))
        print(colored("="*60, "cyan", attrs=["bold"]))
        
        if 'overall' in ordered_results:
            overall = ordered_results['overall']
            success_rate = f"{overall['pc_successes']:.1f}%"
            print(f"Success Rate: {colored(success_rate, 'green', attrs=['bold'])}")
            print(f"Avg Rewards:  {overall['avg_sum_rewards']:.3f}")
            print(f"Max Rewards:  {overall['avg_max_rewards']:.3f}")
            if 'avg_episode_length' in overall:
                print(f"Avg Episode Length: {overall['avg_episode_length']:.1f} timesteps")
        
        # Per-task results
        print("\nPer-task results:")
        for key in sorted(ordered_results.keys()):
            if key not in ['overall', 'eval_s', 'video_paths']:
                task_result = ordered_results[key]
                ep_len_str = f", {task_result['avg_episode_length']:.1f} steps" if 'avg_episode_length' in task_result else ""
                print(f"  {key}: {task_result['pc_successes']:.1f}% success{ep_len_str}")
        
        print(colored("="*60, "cyan", attrs=["bold"]))
    
    logging.info("\nDone!")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    init_logging()
    main()