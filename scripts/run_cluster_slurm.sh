#!/bin/bash
# ==============================================================================
# SLURM Job Submission Script for HKBU GPU Cluster / High-Performance Computing
# Project: "Where does the image matter? Translating visually grounded comics"
# Includes Weights & Biases (W&B) experiment metric tracking
# ==============================================================================
#
# Resource Request Details:
#   --job-name:       Descriptive name for monitoring in squeue
#   --output:         File where console stdout logs are written
#   --error:          File where console stderr logs are written
#   --partition:      GPU partition (e.g., gpu, a100, or faculty partition)
#   --nodes:          1 physical compute node
#   --ntasks:         1 primary task
#   --cpus-per-task:  8 CPU cores for fast parallel data/image loading
#   --gres=gpu:1      1 high-performance GPU (NVIDIA A100-40GB/80GB or RTX 4090)
#   --mem:            32GB system RAM
#   --time:           02:00:00 (2 hours wallclock time - ample for 500 units)
#
# ==============================================================================

#SBATCH --job-name=comic_vlm_eval
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00

echo "=========================================================="
echo "Job Started: $(date)"
echo "Node:        $(hostname)"
echo "Assigned GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================================="

# 1. Load Cluster Environment Modules
module purge
module load cuda/12.2
module load python/3.10

# 2. Activate Python Virtual Environment
source ~/envs/comic_vlm_env/bin/activate

# 3. Verify PyTorch sees the GPU
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0)}')"

# 4. Create directories for logs and predictions
mkdir -p logs outputs/qwen2.5_vl_7b outputs/llama_vision_11b

# 5. Run Publication Benchmark across all RQs and live stream metrics to W&B
echo "Running Publication Benchmark and Streaming Live Metrics to W&B..."
python run_benchmark.py --wandb

# 6. Detailed Tier 1 & Tier 2 Model Inference Runs (Optional for per-bubble logging)
# python -m src.pipelines.vlm_end_to_end --model Qwen/Qwen2.5-VL-7B-Instruct --wandb
# python -m src.pipelines.vlm_end_to_end --model meta-llama/Llama-3.2-11B-Vision-Instruct --wandb

echo "=========================================================="
echo "Job Finished: $(date)"
echo "=========================================================="
