# Fine-tuning Stable Diffusion on Fashion with LoRA

This repository contains code and resources for fine-tuning the Stable Diffusion XL model on fashion-related data using Low-Rank Adaptation (LoRA). The main objective is to determine the optimal LoRA configuration to balance performance and efficiency for fine-tuning, enabling the generation of high-quality images while minimizing the number of trainable parameters and computational requirements.

---

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Training](#training)
  - [Evaluation](#evaluation)

---

## Introduction
This project demonstrates the application of LoRA for fine-tuning Stable Diffusion XL to create realistic, high-resolution fashion images. 

---

## Features
- **LoRA Fine-Tuning**: Efficient parameter-efficient fine-tuning using LoRA.
- **Custom Dataset Support**: Seamlessly integrate fashion-specific datasets.
- **High-Resolution Outputs**: Generate images up to 1024x1024 pixels.
- **Optimized for Efficiency**: Designed for resource-constrained environments.

---

## Project Structure
```
├── configs
│   ├── experiment1
│   │   ├── eval_config_sdxl_w_refiner.json
│   │   ├── eval_config_sdxl_wo_refiner.json
│   │   ├── train_config_sdxl1.json
│   │   ├── train_config_sdxl2.json
│   │   └── train_config_sdxl3.json
│   ├── experiment2
│   ├── ...
│
├── data
│   ├── <dataset_name>
│   │   ├── images/
│   │   │   └── <all images>
│   │   └── captions/
│   │       └── captions.json
│
├── model_logs
├── notebooks
│   ├── 01-look-at-data.ipynb
│   ├── 02-look-at-prepared-data.ipynb
│   ├── 03-look-at-model-outputs.ipynb
│   ├── 04-look-at-model-param-count.ipynb
│   └── ...
├── scripts
│   ├── train_ldm_lora.sh
│   ├── eval_ldm_lora.sh
├── src
│   ├── data_prep/
│   ├── models/
│   ├── train_ldm_lora.py
│   ├── evaluate_ldm_lora.py
│   ├── utils.py
│   └── ...
├── requirements.txt
└── README.md
```

---

## Installation
1. **Clone the Repository**
   ```bash
   git clone https://github.com/<your-repo-name>/Stable-Diffusion-Lora-Finetuning.git
   cd Stable-Diffusion-Lora-Finetuning
   ```

2. **Install Dependencies**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
---

## Usage

### Data Preparation
**Dataset Structure**
   ```
   data/<dataset_name>/
   ├── images/
   │   └── <all images>
   └── captions/
       └── captions.json
   ```
   - **Images**: Place all images in the `images/` folder.
   - **Captions**: Provide a `captions.json` file in the `captions/` folder with the following format:
     ```json
     {
       "image1.jpg": "A woman wearing a red dress.",
       "image2.jpg": "A man in a blue jacket."
     }
     ```
     
### Training
Run the training script:


```bash
bash scripts/train_ldm_lora.sh
```
- Ensure the correct configuration file (e.g., `train_config_sdxl1.json`) and data directory are specified in the script.

### Evaluation
Evaluate the fine-tuned model:

```bash
bash scripts/eval_ldm_lora.sh
```
- Use the appropriate evaluation configuration file (e.g., `eval_config_sdxl_w_refiner.json`), data directory, and specify the trained model checkpoint.
