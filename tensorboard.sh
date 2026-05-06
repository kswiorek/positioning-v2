#!/bin/bash

#SBATCH --job-name=tensorboard
#SBATCH --output=tb_job_%j.log       # The IP will be printed in this file
#SBATCH --time=5:00:00              # Keep it alive for 24 hours
#SBATCH --cpus-per-task=2            # TB doesn't need much CPU
#SBATCH --mem=8G                     # TB doesn't need much RAM
#SBATCH --partition=l4               # Or whatever your default partition is

# 1. Grab the hostname and IP address of the compute node
NODE_NAME=$(hostname)
NODE_IP=$(hostname -I | awk '{print $1}')

# 2. Print them clearly to the log file
echo "=========================================================="
echo "TensorBoard has started!"
echo "Because you are on the VPN, you can access it directly at:"
echo "URL 1: http://$NODE_NAME:6006"
echo "URL 2: http://$NODE_IP:6006"
echo "=========================================================="

# 3. Start TensorBoard
singularity exec \
    -B ~/positioning-v2/venv:/scratch/venv \
    /ceph/container/pytorch/pytorch_25.08.sif \
    /bin/bash -c "source /scratch/venv/bin/activate && tensorboard --logdir ./training_engine/logs --port 6006 --bind_all"