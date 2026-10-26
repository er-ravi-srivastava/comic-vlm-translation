#!/bin/bash
# ==============================================================================
# SLURM Job Submission Script for HKBU GPU Cluster / High-Performance Computing
# Project: "Where does the image matter? Translating visually grounded comics"
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

# 1. Load Cluster Environment Modules (adapt module names to cluster environment)
module purge
module load cuda/12.2
module load python/3.10

# 2. Activate Python Virtual Environment
source ~/envs/comic_vlm_env/bin/activate

# 3. Verify PyTorch sees the GPU
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0)}')"

# 4. Create log directories
mkdir -p logs outputs/qwen2.5_vl_7b outputs/llama_vision_11b

# 5. Run Tier 1: Qwen2.5-VL-7B-Instruct across all 4 visual conditions
echo "Running Tier 1: Qwen2.5-VL-7B across 4 conditions..."
python -m src.pipelines.vlm_end_to_end \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --data data/annotations/curated_benchmark_500.json \
    --conditions text_only correct_image mismatched_image noise_image \
    --target_lang es \
    --output_dir outputs/qwen2.5_vl_7b/

# 6. Run Tier 2: Llama-3.2-11B-Vision-Instruct across all 4 visual conditions
echo "Running Tier 2: Llama-3.2-11B-Vision across 4 conditions..."
python -m src.pipelines.vlm_end_to_end \
    --model meta-llama/Llama-3.2-11B-Vision-Instruct \
    --data data/annotations/curated_benchmark_500.json \
    --conditions text_only correct_image mismatched_image noise_image \
    --target_lang es \
    --output_dir outputs/llama_vision_11b/

# 7. Compute Automated Quality Estimation (COMET-Kiwi) and Bootstrap Statistics
echo "Running Automated Evaluation Suite..."
python -m src.evaluation.compute_metrics \
    --hypotheses outputs/qwen2.5_vl_7b/predictions.json \
    --gold data/annotations/curated_benchmark_500.json

echo "=========================================================="
echo "Job Finished: $(date)"
echo "=========================================================="
