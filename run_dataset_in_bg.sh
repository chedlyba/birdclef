#!/bin/bash

python3 dataset.py > logs/dataset.log  2>&1 &

disown