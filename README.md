
# Mudjacking: Patching Backdoor Vulnerabilities in Foundation Models

This repository contains the core code of Mudjacking, which patches foundation models to remove backdoors.

## Citation

If you find the code helpful, please cite:

```
@inproceedings{liu2024mudjacking,
  title={Mudjacking: Patching Backdoor Vulnerabilities in Foundation Models},
  author={Liu, Hongbin and Reiter, Michael K and Gong, Neil Zhenqiang},
  booktitle={USENIX Security Symposium},
  year={2024}
}
```

## Requirements

```bash
cd Mudjacking
conda env create -f environment.yml
conda activate Mudjacking
```

## Prepare the Datasets

Please download the datasets and place them in the respective folders:

- CIFAR-10: `./data/cifar10`
- STL-10: `./data/stl10`

Google Drive link:

## Pre-trained Backdoored and Patched Image Foundation Models

Download the pre-trained models and place them in the respective folder:

- Pre-trained models: `./output/cifar10`

Google Drive link: 

## Run Experiments

To run the experiments, follow these steps:

### Step 1: Compute Attribution Score Given a Bug

Run the following script to compute the attribution score:

```bash
python3 ./scripts/run_reverse_engineering_attribute.py
```

The bug file is located in `./data_bugs/BadEncoder/pretrain_cifar10_downstream_stl10/backdoor_bug_1/0/`.

### Step 2: K-means to Reverse Engineer Backdoor Trigger

Run the following script to reverse engineer the backdoor trigger using K-means:

```bash
python3 ./scripts/run_reverse_engineering_k_means.py
```

### Step 3: Run Mudjacking to Patch the Backdoored Vision Foundation Model

To patch the backdoored foundation model using Mudjacking, run:

```bash
python3 ./scripts/run_patching.py
```

### Step 4: Train Downstream Classifiers

To train the downstream classifiers before and after applying the patch, run:

```bash
python3 ./scripts/run_training_downstream_classifier.py
```

## Acknowledgement

We would like to acknowledge the BadEncoder repository, which can be found here:

[BadEncoder (S&P 2022)](https://github.com/jinyuan-jia/BadEncoder)
