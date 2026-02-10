#!/usr/bin/env bash
set -euo pipefail

POLICY_PATH=<your policy path>

N_EPISODES=500
BATCH_SIZE=32
NUM_GPUS=8
N_ACTION_STEPS=5
SEED=42

export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false

SUITES=(libero_spatial libero_object libero_goal libero_10)

for suite in "${SUITES[@]}"; do
  for async_delay in 1 2 3 4; do
    out="outputs/eval/pi05_async_libero/${suite}/async_delay_${async_delay}_action_${N_ACTION_STEPS}/vlash"
    echo "[RUN] suite=${suite} async_delay=${async_delay} -> ${out}"

    python -m vlash.cli eval-libero \
      --policy.path="${POLICY_PATH}" \
      --output_dir="${out}" \
      --env.type=libero --env.task="${suite}" \
      --eval.n_episodes="${N_EPISODES}" --eval.batch_size="${BATCH_SIZE}" --eval.use_async_envs=True \
      --policy.device=cuda --policy.use_amp=false --policy.n_action_steps="${N_ACTION_STEPS}" \
      --eval.async_delay="${async_delay}" --eval.method_type=vlash \
      --seed="${SEED}" \
      --num_gpus="${NUM_GPUS}"
  done
done
