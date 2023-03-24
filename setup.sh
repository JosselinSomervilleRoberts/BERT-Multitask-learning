#!/usr/bin/env bash

conda create -n cs224n_dfp2 python=3.8
conda activate cs224n_dfp2
pip install pytorch==1.8.0 torchvision torchaudio cudatoolkit=10.1 -c pytorch
pip install -r requirements.txt
