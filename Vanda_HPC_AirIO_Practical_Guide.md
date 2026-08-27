# NUS Vanda HPC Air-IO Practical Guide

This guide covers setup and TLab fine-tuning for Air-IO on the NUS Vanda GPU cluster.

## 1. Log in and enter Air-IO

```bash
ssh e1234567@vanda.nus.edu.sg
cd ~/Air-IO
git status
git pull
```

Use the login node for setup, PBS submission, status checks, and log inspection. Run training through PBS.

## 2. Run the one-time setup

The setup script downloads the two TLab datasets, the pretrained Air-IO EuRoC model, and the Python requirements.

```bash
cd ~/Air-IO
chmod +x setup_airio.sh
./setup_airio.sh
```

The expected pretrained checkpoint is:

```text
AirIO_EuRoC/AirIO_checkpoint/best_model.ckpt
```

Verify the inputs:

```bash
ls T-Lab_31st_July_dataset
ls T-Lab_28th_July_dataset
ls -lh AirIO_EuRoC/AirIO_checkpoint/best_model.ckpt
```

To use a different checkout or container image:

```bash
AIRIO_REPO_DIR=/path/to/Air-IO ./setup_airio.sh
AIRIO_IMAGE=/path/to/pytorch.sif ./setup_airio.sh
```

## 3. Check the Air-IO configuration

Fine-tuning uses `configs/TLab/finetune_motion_body.conf`. The PBS file overrides the config's original `/content/Air-IO/...` checkpoint value with the repository-relative Vanda path. Confirm the TLab dataset paths with:

```bash
grep -nE 'data_root|rot_path|pretrained_ckpt' \
  configs/datasets/TLab/tlab_body.conf \
  configs/TLab/finetune_motion_body.conf
```

The training and evaluation dataset paths are repository-relative. The `rot_path` belongs to the inference section and is not read during fine-tuning.

## 4. Configure Weights & Biases

W&B login is normally needed only once:

```bash
module load apptainer/1.4.1
image=/app1/common/singularity-img/vanda/pytorch_2.5_cuda_12.4_unsloth.sif
apptainer exec -e "$image" python3 -m wandb login
apptainer exec -e "$image" python3 -m wandb status
```

In this repository, adding `--log` to the training command disables W&B logging.

## 5. Submit Air-IO fine-tuning

Submit from the repository so `PBS_O_WORKDIR` points to Air-IO:

```bash
cd ~/Air-IO
qsub airio_finetune.pbs
```

The job requests one GPU, 36 CPU cores, 250 GB RAM, and 24 hours. It runs:

```bash
python3 -u train_motion_finetune.py \
  --config configs/TLab/finetune_motion_body.conf \
  --pretrained_ckpt AirIO_EuRoC/AirIO_checkpoint/best_model.ckpt \
  --device cuda:0
```

The PBS job name is `airio_finetune`, so output logs normally start with `airio_finetune.o`.

Optional submission-time overrides:

```bash
qsub -v AIRIO_CONFIG=configs/TLab/finetune_motion_body.conf airio_finetune.pbs
qsub -v AIRIO_IMAGE=/path/to/pytorch.sif airio_finetune.pbs
qsub -v AIRIO_CHECKPOINT=/path/to/best_model.ckpt airio_finetune.pbs
```

## 6. Monitor and manage the job

```bash
qstat
qstat -x
qstat -fx JOB_ID
```

Typical states are `Q` (queued), `R` (running), and `F` (finished).

Watch the newest Air-IO log:

```bash
tail -f "$(ls -t airio_finetune.o* | head -1)"
```

Press `Ctrl-C` to stop watching; the PBS job continues. Read the completed log with:

```bash
cat "$(ls -t airio_finetune.o* | head -1)"
```

Cancel a job:

```bash
qdel JOB_ID
```

Inspect resource use and exit status:

```bash
qstat -fx JOB_ID | grep -E 'Resource_List|resources_used|Exit_status'
```

## 7. Useful Vanda commands

```bash
hpc gpu
hpc pbs help
hpc project
hpc gstat
qstat -Q
```

## Quick reference

```bash
ssh e1349884@vanda.nus.edu.sg
cd ~/Air-IO
git pull
./setup_airio.sh
qsub airio_finetune.pbs
qstat
tail -f "$(ls -t airio_finetune.o* | head -1)"
qstat -fx JOB_ID
```
