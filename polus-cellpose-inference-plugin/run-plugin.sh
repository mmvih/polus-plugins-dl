#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize ../data)

# Inputs
inpDir=/data/input
diameter=0
diameterMode=EveryImage
pretrainedModel=nuclei

# Output paths
outDir=/data/output

# Remove the --gpus all to test on CPU
docker run --mount type=bind,source=${datapath},target=/data/ \
            --user $(id -u):$(id -g) \
            --gpus all \
            labshare/polus-cellpose-inference-plugin:${version} \
            --inpDir ${inpDir} \
            --diameter ${diameter} \
            --diameterMode ${diameterMode} \
            --pretrainedModel ${pretrainedModel} \
            --outDir ${outDir}