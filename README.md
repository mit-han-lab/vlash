# VLASH-simulation-benchmark-LIBERO

This repo contains a VLASH fine-tuning and evaluation for **LIBERO** simulation benchmarks.

## Install

1. **Create and activate your environment (example)**

    ```bash
    conda create -n vlash-libero python=3.10
    conda activate vlash-libero
    ```

2. **Install LIBERO**

   Follow the installation instructions from this repo:  
   https://github.com/Lifelong-Robot-Learning/LIBERO#installation

3. **Modify LIBERO Benchmark Initialization**

   In `LIBERO/libero/libero/benchmark/__init__.py`, replace:
   ```python
   init_states = torch.load(init_states_path)
   ```
   with:
   ```python
   init_states = torch.load(init_states_path, weights_only=False)
   ```

4. **Install VLASH**

    ```bash
    cd vlash
    pip install -e .
    ```

5. **Check numpy version if you encounter segfaults**

   If you experience segmentation faults, ensure you have numpy version 1.24.4:
   ```bash
   pip install numpy==1.24.4
   ```

## Fine-tuning (LIBERO)

Example training config:

```bash
vlash train examples/train/pi05/libero.yaml
```


## Evaluate (LIBERO)

Example (multi-GPU, 4 suites, async_delay sweep):

```bash
bash libero-eval-scripts/run.sh
```