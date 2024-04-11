#!/bin/bash

# setup-environment.sh /path/to/speechbrain

sb_root=$1

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
reqs_override_pins $1/requirements.txt $SLURM_TMPDIR/requirements.txt
reqs_override_pins $1/lint-requirements.txt $SLURM_TMPDIR/lint-requirements.txt

# Install dependencies
pip install --no-index --upgrade pip
pip install --no-index -r $SLURM_TMPDIR/requirements.txt
pip install --no-index numba
pip install --no-index --no-dependencies --editable $sb_root
pip list