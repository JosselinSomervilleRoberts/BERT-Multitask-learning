#!/usr/bin/env bash

conda create --name cs224n_dfp_final --file spec-file.txt
conda activate cs224n_dfp_final

pip install -r requirements.txt
