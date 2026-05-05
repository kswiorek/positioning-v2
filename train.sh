#!/bin/bash

#SBATCH --job-name=pytorch_train
#SBATCH --output=training_%j.log    # %j adds the job ID to the filename
#SBATCH --error=training_%j.err     # Separates errors into a different file
#SBATCH --time=2:00:00             # Max time limit (HH:MM:SS) - set to 24 hours
#SBATCH --cpus-per-task=16          # Number of CPU cores for fast data loading
#SBATCH --mem=32G                   # System RAM (gives you plenty of headroom over 17GB)
#SBATCH --gres=gpu:1                # Request 1 GPU

# Execute the training command inside your specific container
singularity exec --nv \
    -B ~/positioning-v2/venv:/scratch/venv \
    /ceph/container/pytorch/pytorch_25.08.sif \
    /bin/bash -c "source /scratch/venv/bin/activate && python -m training_engine.train --config training_engine/training_config.example.json"