#!/bin/bash

# librispeech-extract-local.sh ~/projects/def-ravanelm/datasets/librispeech/*.tar.gz

cd $SLURM_TMPDIR
for f in $@; do
	tar -xf $f
done

#wait < <(jobs -p)

ls -l LibriSpeech

# hacky, but making sure we have a consistent path to the dataset even when
# extracted to a local scratch
unlink ~/LibriSpeech-symlink || true
ln -s $SLURM_TMPDIR/ ~/LibriSpeech-symlink || true