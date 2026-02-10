import math
import os
from collections import OrderedDict
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


MAX_STEP_DICT: dict[str, int] = {
    "libero_spatial": 230,
    "libero_object": 290,
    "libero_goal": 310,
    "libero_10": 530,
    "libero_90": 410,
}


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """
    Copied from robosuite:
    https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    quat = np.asarray(quat).copy()
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(float(quat[3]))) / den


def _resolve_task(suite_name: str, task_name: str):
    bench = benchmark.get_benchmark_dict()
    if suite_name not in bench:
        raise ValueError(
            f"Unknown LIBERO suite {suite_name!r}. Available: {sorted(bench.keys())}"
        )
    suite = bench[suite_name]()
    for task_id in range(suite.n_tasks):
        task = suite.get_task(task_id)
        if task.name == task_name:
            return suite, task_id, task
    raise ValueError(f"Task {task_name!r} not found in suite {suite_name!r}")


class LiberoEnvWrapper(gym.Env):
    """
    Integrated replacement for the external `gym_libero` package.

    This wrapper matches the observation/action formats used by our eval harness,
    while constructing the underlying LIBERO env directly via `OffScreenRenderEnv`.
    """

    metadata = {"render_modes": ("human", "rgb_array", "cameras"), "render_fps": 30}

    def __init__(
        self,
        *,
        suite_name: str,
        task_name: str,
        init_state_id: int = 0,
        max_episode_steps: int | None = None,
        camera_heights: int = 256,
        camera_widths: int = 256,
        **env_kwargs: Any,
    ):
        self.suite_name = suite_name
        self.task_name = task_name

        self.task_suite, self.task_id, self.task_obj = _resolve_task(suite_name, task_name)
        self.init_states = self.task_suite.get_task_init_states(self.task_id)
        self.init_state_id = int(init_state_id)

        self.num_wait_steps = 10
        self.metadata["render_fps"] = int(env_kwargs.get("render_fps", 30))

        self.image_height = int(camera_heights)
        self.image_width = int(camera_widths)

        # Create underlying LIBERO env
        bddl_file_name = os.path.join(
            get_libero_path("bddl_files"),
            self.task_obj.problem_folder,
            self.task_obj.bddl_file,
        )

        # OffScreenRenderEnv expects LIBERO / robosuite style kwargs; keep this minimal.
        self._env = OffScreenRenderEnv(
            bddl_file_name=bddl_file_name,
            camera_heights=self.image_height,
            camera_widths=self.image_width,
        )

        # Episode length handling (Gym-style)
        self._max_episode_steps = int(max_episode_steps) if max_episode_steps is not None else int(
            MAX_STEP_DICT.get(suite_name, 530)
        )
        self._elapsed_steps = 0

        # Expose language instruction
        self.task = self.task_obj.language
        self.task_description = self.task

        # Action/obs spaces (kept compatible with prior gym_libero wrapper)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "pixels": spaces.Dict(
                    {
                        "image": spaces.Box(
                            low=0,
                            high=255,
                            shape=(self.image_height, self.image_width, 3),
                            dtype=np.uint8,
                        ),
                        "wrist_image": spaces.Box(
                            low=0,
                            high=255,
                            shape=(self.image_height, self.image_width, 3),
                            dtype=np.uint8,
                        ),
                    }
                ),
                "agent_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float64),
            }
        )

        self.last_image = np.zeros((self.image_height, self.image_width * 2, 3), dtype=np.uint8)
        self.reset_count = 0

    def _obs_reformat(self, obs: dict[str, Any]) -> OrderedDict:
        state = np.concatenate(
            [
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            ]
        )

        # Match historical gym_libero orientation.
        agentview = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wristview = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

        out = OrderedDict()
        out["pixels"] = {"image": agentview, "wrist_image": wristview}
        out["agent_pos"] = state

        self.last_image = np.concatenate([agentview, wristview], axis=1)
        return out

    def _action_reformat(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action).copy()
        action[-1] = np.sign(action[-1])
        return action

    def _get_info(self) -> dict[str, Any]:
        return {"is_success": bool(self._env.check_success())}

    def reset(self, **kwargs: Any):
        init_state_id = kwargs.pop("init_state_id", None)
        seed = kwargs.get("seed", None)

        _ = self._env.reset()
        if init_state_id is not None:
            self.init_state_id = int(init_state_id)

        obs = self._env.set_init_state(self.init_states[self.init_state_id])
        obs = self._obs_reformat(obs)

        # Delay for a few steps to let the env settle
        dummy_action = np.asarray([0.0] * 6 + [-1.0], dtype=np.float32)
        for _ in range(self.num_wait_steps):
            obs, _, _, _ = self._env.step(dummy_action)
            obs = self._obs_reformat(obs)

        # Gym-style step counter
        self._elapsed_steps = 0

        # Keep parity with external gym_libero: if seed provided, rotate init_state_id.
        if seed is not None:
            self.reset_count += 1
            self.init_state_id = (self.init_state_id + 1) % len(self.init_states)

        return obs, {}

    def step(self, action: np.ndarray):
        self._elapsed_steps += 1

        action = self._action_reformat(action)
        obs, reward, done, info = self._env.step(action)
        obs = self._obs_reformat(obs)

        terminated = bool(done)
        truncated = bool((not terminated) and (self._elapsed_steps >= self._max_episode_steps))
        done_out = terminated or truncated
        info_out = self._get_info()

        return obs, float(reward), done_out, done_out, info_out

    def render(self, *args: Any, **kwargs: Any):
        return self.last_image

    def get_language_instruction(self) -> str:
        return str(self.task)

    def close(self):
        return self._env.close()


def make_env_from_suite_task(
    *,
    suite_name: str,
    task_name: str,
    init_state_id: int,
    max_episode_steps: int,
    camera_heights: int = 256,
    camera_widths: int = 256,
) -> LiberoEnvWrapper:
    return LiberoEnvWrapper(
        suite_name=suite_name,
        task_name=task_name,
        init_state_id=init_state_id,
        max_episode_steps=max_episode_steps,
        camera_heights=camera_heights,
        camera_widths=camera_widths,
    )


