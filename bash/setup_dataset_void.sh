#!/bin/bash

mkdir -p data

gdown https://drive.google.com/uc?id=1Keem2q_3XKGb5Z5QpSKCNkWic5tU4ma2
unzip void_release.zip
mv void_release data/void_release

python setup/setup/setup_dataset_void.py