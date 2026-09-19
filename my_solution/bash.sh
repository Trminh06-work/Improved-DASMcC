#!/bin/bash
# Arguments after the script name are passed straight to train.py. The
# autoencoder consumes the CNN's features, so chain the two stages:
#
    # cnn=$(sbatch --parsable --job-name=cnn bash.sh \
    #     --model cnn --epochs 500 --batch-size 64 --lr 1e-4 \
    #     --out checkpoints/cnn.pt)

    # sbatch --dependency=afterok:$cnn --job-name=ae16 bash.sh \
    #     --model ae --epochs 500 --batch-size 64 --lr 1e-4 --latent-dim 16 \
    #     --cnn-checkpoint checkpoints/cnn.pt --out checkpoints/ae16.pt
#
#SBATCH --job-name=train
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

set -eo pipefail

cd "$SLURM_SUBMIT_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jupyter_env

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "$(hostname): train.py $*"
python train.py "$@"
