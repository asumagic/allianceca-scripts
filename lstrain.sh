#!/bin/bash
#SBATCH --job-name="SB-FastConformer-L-v4"
#SBATCH --gpus-per-node=2
######SBATCH --gpus-per-node=1
#SBATCH --array=0-6%1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --account=rrg-ravanelm
#SBATCH --time=11:00:00

set -ex

scripts_root=$(dirname "$0")

sb_root='/home/sdelang/sb'
librispeech_root='/home/sdelang/projects/def-ravanelm/datasets/librispeech/'

hparam_file='hparams/conformer_transducer.yaml'

nvidia-smi

. ${scripts_root}/toolkits/speechbrain/load-modules.sh
${scripts_root}/toolkits/speechbrain/setup-environment.sh $sb_root
${scripts_root}/datasets/librispeech-extract-local.sh ${librispeech_root}/*.tar.gz

cd "${sb_root}/recipes/LibriSpeech/ASR/transducer"

# FIXME: specifying this causes an error (thread count mismatch)
# export OMP_NUM_THREADS=8

# export CUDA_LAUNCH_BLOCKING=1  # use if debugging CUDA errors
# export TORCH_DISTRIBUTED_DEBUG=info  # use if debugging DDP errors

srun torchrun --nproc_per_node=2 \
    train.py ${hparam_file} \
    --data_folder "/home/sdelang/LibriSpeech-symlink/LibriSpeech" \
    --skip_prep False \
    --precision fp16
