#!/bin/bash

# Start the training pipeline in the background using nohup
# Assumes you are in the sdr_intrusion_detection directory

echo "Starting ablation study training in the background..."

nohup python -m src.train --train_all > training.log 2>&1 &

echo "Training started! PID: $!"
echo "Logs are being written to training.log"
echo "To view the progress in real-time, run:"
echo "tail -f training.log"
