# LeRobot (SO-100) + Isaac-GR00T Training & Inference Pipeline

This guide explains how to **collect teleoperation data**, **train GR00T**, and **run inference** on the **SO-100 robot**, using **LeRobot** for data workflows and **Isaac-GR00T** for model training and execution.

---


## Version Compatibility (Important)

LeRobot and Hugging Face update dataset schemas frequently. GR00T training code expects **LeRobot dataset schema `v2.0`**. To overcome this issue, we use a different Isaac Gr00t repo which has the fix.

---

## Overview

Use the following repository + branch:

| Repo | Link | Purpose | Branch / Notes |
|------|------|---------|-----------------|
| Isaac-GR00T | git@github.com:pengjujin/Isaac-GR00T.git | Model training and policy inference | `dataset_3_0` branch *(required)* |
| LeRobot | https://github.com/huggingface/lerobot.git/tree/main | Hardware drivers, teleoperation, dataset recording | `main` branch (required) |
| HomeLeRobot | https://github.com/LF-Luis/HomeLeRobot/tree/main | LeRobot Utils / Evaluation Scripts | Optional but recommended |

---

## 1) Environment(s) Setup

### Local Robot Machine (Data Collection)

Setup miniconda if not already setup
```bash
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```


```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot

conda create -n lerobot python=3.10
conda activate lerobot

pip install draccus
pip install -e .
pip install -e ".[feetech]"   # Enables SO-100/SO-101 servo communication
conda install ffmpeg -c conda-forge
conda deactivate
cd ..

# Clone and setup Gr00t (also install lerobot as pkg)
git clone git@github.com:pengjujin/Isaac-GR00T.git
cd Isaac-GR00T
git checkout dataset_3_0

conda create -n gr00t python=3.10
conda activate gr00t

pip install --upgrade setuptools
pip install -e .[base]
pip install --no-build-isolation flash-attn==2.7.1.post4 
cd ..

cd lerobot
pip install -e .
cd ..

```

## 2) Data Collection Workflow

### Step A — Motor Setup
1. Find ports on robot cpu corresponding to leader and follower (https://huggingface.co/docs/lerobot/enso101#1-find-the-usb-ports-associated-with-each-arm) 
2. Register motors ids for leader and follower (https://huggingface.co/docs/lerobot/en/so101?example=Linux#2-set-the-motors-ids-and-baudrates)
3. Calibrate leader and follower (https://huggingface.co/docs/lerobot/en/so101?example=Linux#calibrate)


### Step B — Identify Camera Devices

```bash
lerobot-find-cameras opencv
```

### Step C — Leader-Follower Teleoperation
Install Rerun.io
Use this for validating leader follower calibration + teleop practice
```bash
lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port=/dev/<follower port> \
  --robot.id=<follower name (as used in calibration script)> \
  --robot.cameras="{ front: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 30} }" \
  --teleop.type=so101_leader \
  --teleop.port=/dev/<leader port> \
  --teleop.id=<leader name (as used in calibration script)> \
  --display_data=true
```

### Step D - Record dataset
```bash
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/<follower port> \
  --robot.id=<follower name (as used in calibration script)> \
  --robot.cameras="{ front: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30} }" \
  --teleop.type=so101_leader \
  --teleop.port=/dev/<leader port> \
  --teleop.id=<leader name (as used in calibration script)> \
  --dataset.repo_id=argus_assignment/nov_6_dataset_final_11 \
  --dataset.num_episodes=5 \
  --dataset.single_task="put pineapple in bowl" \
  --display_data=true
```

### Step E - Upload data to HF dataset
Create hf token (read+write permissions) and add to .bashrc
```bash
echo 'export HUGGINGFACE_HUB_TOKEN="<token from HF>"' >> ~/.bashrc
source ~/.bashrc
conda activate lerobot

hf auth login --token "$HUGGINGFACE_HUB_TOKEN"

# Update the HF_NAMESPACE, BASE_DIR, DATASET_NAME_PREFIX in the script
python auto_upload.py
```

## 3) Model Training (Lambda)

### Step A - Run step #1 on lambda

### Step B - Download dataset from huggingface locally
Create hf token (read+write permissions) and add to .bashrc
```bash
echo 'export HUGGINGFACE_HUB_TOKEN="<token from HF>"' >> ~/.bashrc
source ~/.bashrc
conda activate lerobot

mkdir datasets
hf auth login --token "$HUGGINGFACE_HUB_TOKEN"
python "$HOME/HomeLeRobot/get_hf_data.py" \
  --datasets "<repo id on HF>" \
  --base_dir "$HOME/datasets/train_dataset"
  --cam dual

conda deactivate
```

### Step C - Train in gr00t env
```bash
conda activate gr00t

python scripts/gr00t_finetune.py \
  --dataset-path ~/datasets/final_train_dataset/nov_6_dataset_final_* \
  --num-gpus 1 \
  --output-dir /home/ubuntu/model_ft_output \
  --max-steps 10000 \
  --data-config so100_dualcam \
  --video-backend torchvision_av

conda deactivate
``` 

### Step  - Upload checkpoint to HF
```bash
conda activate lerobot

hf auth login --token "$HUGGINGFACE_HUB_TOKEN"
hf repo create <HF_username>/<model_checkpoint_name> --repo-type model

python upload_model_to_hf.py <path_to_root_of_model_checkpoints_dir>/<checkpoint_dir_to_upload>/ <HF_username>/<model_checkpoint_name>

conda deactivate
```
## 4) Inference

### Step A - Start GR00T Policy Server (Lambda)
```bash
conda activate gr00t

python scripts/inference_service.py \
  --model_path /home/ubuntu/model_ft_output/checkpoint-10000 \ --server  \
  --data_config so100_dualcam  \
  --embodiment_tag new_embodiment  \
  --port 5555 \
  --host 0.0.0.0

conda deactivate
```

### Step B - Forward Port to Robot Machine
Setup a pem for lambda and store it on local machine. Put it in ~/.ssh dir
```bash
ssh -i ~/.ssh/lambda.pem -N -L 5555:127.0.0.1:5555 ubuntu@<lambda-ip>
```

### Step C - Run Robot with Trained Policy
Run evaluation from Gr00t
```bash
conda activate gr00t

python $HOME/robotics/Isaac-GR00T/examples/SO-100/eval_lerobot.py \ 
--robot.type=so101_follower \ 
--robot.port=/dev/<follower port> \ 
--robot.id=follower \ --robot.cameras="{ front: {type: opencv, index_or_path: 3, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 2, width: 640,  height: 480, fps: 30} }" \
--policy_host=localhost \
--policy_port=5555 \ 
--lang_instruction="put pineapple in bowl"

conda deactivate
```


## Final dataset used

The full training dataset consists of **11 teleoperated episodes sets**, each stored as a separate Hugging Face dataset.  
All of these together form the **SO-100 Nov 6 Final Dataset**.

| Dataset Name | Hugging Face Link | Notes |
|--------------|------------------|-------|
| `nov_6_dataset_final_1` | https://huggingface.co/datasets/KavitShah1998/nov_6_dataset_final_1 | ✅ Uploaded |
| `nov_6_dataset_final_2` | https://huggingface.co/datasets/KavitShah1998/nov_6_dataset_final_2 | ✅ Uploaded |
| `nov_6_dataset_final_3` | https://huggingface.co/datasets/KavitShah1998/nov_6_dataset_final_3 | ✅ Uploaded |
| `nov_6_dataset_final_4` | https://huggingface.co/datasets/KavitShah1998/nov_6_dataset_final_4 | ✅ Uploaded |
| `nov_6_dataset_final_5` | https://huggingface.co/datasets/KavitShah1998/nov_6_dataset_final_5 | ✅ Uploaded |
| `nov_6_dataset_final_6` | https://huggingface.co/datasets/KavitShah1998/nov_6_dataset_final_6 | ✅ Uploaded |
| `nov_6_dataset_final_7` | https://huggingface.co/datasets/KavitShah1998/nov_6_dataset_final_7 | ✅ Uploaded |
| `nov_6_dataset_final_8` | https://huggingface.co/datasets/KavitShah1998/nov_6_dataset_final_8 | ✅ Uploaded |
| `nov_6_dataset_final_9` | https://huggingface.co/datasets/KavitShah1998/nov_6_dataset_final_9 | ✅ Uploaded |
| `nov_6_dataset_final_10` | https://huggingface.co/datasets/KavitShah1998/nov_6_dataset_final_10 | ✅ Uploaded |
| `nov_6_dataset_final_11` | https://huggingface.co/datasets/KavitShah1998/nov_6_dataset_final_11 | ✅ Uploaded |

To download all of them at once (training machine):
```bash
python ~/HomeLeRobot/get_hf_data.py \
  --datasets "KavitShah1998/nov_6_dataset_final_1 KavitShah1998/nov_6_dataset_final_2 KavitShah1998/nov_6_dataset_final_3 KavitShah1998/nov_6_dataset_final_4 KavitShah1998/nov_6_dataset_final_5 KavitShah1998/nov_6_dataset_final_6 KavitShah1998/nov_6_dataset_final_7 KavitShah1998/nov_6_dataset_final_8 KavitShah1998/nov_6_dataset_final_9 KavitShah1998/nov_6_dataset_final_10 KavitShah1998/nov_6_dataset_final_11" \
  --base_dir "$HOME/datasets/final_train_dataset" \
  --cam dual
```

## Fine-tuned Model Link

| Model | Hugging Face Link | Notes |
|-------|------------------|-------|
| `gr00t-so100-nov6-v3` | https://huggingface.co/KavitShah1998/gr00t-so100-nov6-v3 | ✅ Used for inference & deployment |

## Evaluation

This section summarizes the **real-world performance** of the trained GR00T policy on the SO-100 robot.  
Evaluations were run using **live teleop-free execution** with language instructions.

---

#### Eval Setup

| Item | Details |
|------|---------|
| Robot | SO-100 / SO-101 |
| Cameras Used | Front (external) + Wrist (eye-in-hand) |
| Task Instruction Format | Natural language (e.g., `"put pineapple in bowl"`) |
| Execution Mode | Closed-loop policy inference streamed from Lambda via GR00T inference server |
| Number of Trials | `10` per task |
| Number of Tasks | `1` |
| Environment | `tabletop` |

---

#### Tasks Set

| Task Name | Example Instruction | Object(s) Used |
|----------|---------------------|----------------|
| Place object in container | `"put pineapple in bowl"` | Pineapple toy, bowl |


---

#### Metrics

We measured:

| Metric | Description |
|--------|-------------|
| **TCS — Task Completion Score** | Whether the final state satisfies the task. |
| **Success Rate** | Fraction of trials completing the task correctly |
| **Goal Reached** | Robot reached correct final pose |
| **Grasp Stability** | Object remained stable during motion |

---

#### TCS (Task Completion Score)

For the task **"Put Object in Container"**, we evaluate performance as progress through the following **four sequential stages**:

| TCS Score | Stage | Description | Success Condition |
|----------|--------|-------------|------------------|
| **1** | **Reach Object** | The robot moves from its initial position toward the object. | End-effector reaches the object's vicinity (within grasp envelope). |
| **2** | **Grasp & Lift Object** | The robot grasps the object and lifts it without dropping. | Object is securely held and lifted off the surface. |
| **3** | **Move Above Container** | The robot transports the object and positions it above the container opening. | Wrist pose is centered above container opening (within allowed margin). |
| **4** | **Place Object in Container** (**Success**) | The robot correctly releases the object into the container. | Object ends inside the container with no drop/fall outside. ✅ |

> **TCS = 4** indicates **full task success**.  
> Partial scores (0, 1, 2, 3) help diagnose *where* behavior failed.

#### Quantitative Results

| Trial | TCS Score | Stage Reached |
|-------|----------|---------------|
| 1 | 0 | Arm moved to wrong position |
| 2 | 2 | Grasp failed / unstable lift |
| 3 | 3 | Alignment above container inaccurate |
| 5 | 0 | Arm moved to wrong position |
| 4 | 3 | Alignment above container inaccurate |
| 6 | 0 | Arm moved to wrong position |
| 7 | 1 | Reached object successfully but didnt pick up|
| 8 | 1 | Grasp failed / unstable lift |
| 9 | 1 | Reached object successfully but didnt pick up |
| 10 | 0 | Arm moved to wrong position |


---

#### Observations (Qualitative)

**Strengths**
- Whenever grasp is strong, arm moves object near bowl
- Once object is released (may or may not drop it in bowl) arm resets back
- After reset, arm keeps retrying to pick object if object is found outside vicinity of bowl

**Failure Modes / Limitations**
- Arm keeps moving to default position if it does not see object from wrist camera at start and it becomes un-recoverable
- Sideways approach seemed to trigger grasp more reliably than top down (directly correlates to grasp stability)
- Arm behaves better with higher action horizon (15) compared to default (8)
- Arm did not learn to place INSIDE the bowl, it triggers termination near the bowl
- Arm gets locked in initial configuration (geometry limit of arm) when object is closer to arm's init config (extra torque needed during teleop to bring the arm out of init config .. robot didnt learn)


**Potential Fixes / Next Steps**
- Add more distribution for top down grasp from more set of locations
- Update data such that arm is first moved out of initial config .. go to pick
- Add scan behavior (showing search for object) before picking
- User better contrast between table, background & bowl

---

#### Model Card Summary (for publishing)

| Property | Value |
|---------|-------|
| Training Sources | `nov_6_dataset_final_[1–11]` |
| Total Episodes | 10 |
| Policy Type | GR00T n1.5 (dual-camera config) |
| Dataset Type | LeRobot V3.0 |
| Deployment | SO-101 follower mode, live inference from Lambda |
| License | `MIT` 