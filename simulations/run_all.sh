#!/usr/bin/env bash

echo "Running FFN"
python linear.py

echo "Running ReLU"
python relu.py

echo "Running GeLU"
python gelu.py

echo "Running LayerNorm"
python layernorm.py

echo "Running Dropout"
python dropout.py

echo "Running Softmax"
python softmax.py

echo "Running SHA"
python SHA.py