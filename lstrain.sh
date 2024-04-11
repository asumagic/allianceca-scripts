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

sb_root='/home/sdelang/sb'

hparam_file='hparams/conformer_transducer.yaml'

nvidia-smi

module load StdEnv/2020

# TODO: update to support 2023 (probably need to update the CUDA module versions)
# cudacore no longer seems to be a thing
#module load StdEnv/2023
module load cuda/11.7
module load cudacore/.11.7.0  # for numba; provides libnvvm; unsure why separate
module load python/3.11

# Create virtual environment on the local scratch
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

# TODO: maybe the requirements.txt file simply doesn't need to be used:
# the setup.py installs dependencies itself

# For requirement files, remove package pins.
# This is hacky, but we need to rely on whatever version the cluster has available
function reqs_override_pins() {
    sed -E -e 's/(==|^SoundFile).*//' $1 >$2
    diff $1 $2 || true
}
reqs_override_pins $sb_root/requirements.txt $SLURM_TMPDIR/requirements.txt
reqs_override_pins $sb_root/lint-requirements.txt $SLURM_TMPDIR/lint-requirements.txt

# Install dependencies
pip install --no-index --upgrade pip
pip install --no-index -r $SLURM_TMPDIR/requirements.txt
pip install --no-index numba
pip install --no-index --no-dependencies --editable $sb_root
pip list

cd $SLURM_TMPDIR
for f in ~/projects/def-ravanelm/datasets/librispeech/*.tar.gz; do
	tar -xf $f # &   # will extract into /LibriSpeech
done

#wait < <(jobs -p)

ls -l LibriSpeech

# hacky, but making sure we have a consistent path to the dataset even when
# extracted to a local scratch
unlink ~/LibriSpeech-symlink || true
ln -s $SLURM_TMPDIR/ ~/LibriSpeech-symlink || true

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
