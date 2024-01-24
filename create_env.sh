#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh

export env_name="kbtu-nlp"
conda create -n $env_name python=3.10 -y
conda activate $env_name
pip install -r requirements.txt
