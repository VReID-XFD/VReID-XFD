# VSLA-CLIP Baseline for VReID-XFD Workshop (WACV 2026)

This repository contains the **baseline implementation** for the **VReID-XFD: Video-based Human Recognition at Extreme Far Distances** workshop, accepted at **WACV 2026**.  
It is built upon the [VSLA-CLIP](https://github.com/FHR-L/VSLA-CLIP) framework and provides reproducible training and evaluation pipelines for long-range person recognition tasks on the **DetReIDX** dataset.

---

## 🧩 Environment Setup

```bash
conda create -n vslaclip_new python=3.8
conda activate vslaclip_new
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
pip install yacs timm scikit-image tqdm ftfy regex
```

---

## 🚀 Training

To train the baseline model:

```bash
CUDA_VISIBLE_DEVICES=0 python train_reidadapter.py --config_file configs/adapter/vit_adapter.yml
```

---

## 🧪 Evaluation

Three cross-view matching scenarios are supported.

To evaluate all cases together:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_all_cases.py \
  --config_file configs/adapter/vit_adapter.yml \
  --model_path logs/ViT-B-16_5.pth
```

After evaluation, results will be saved as:

```
logs/evaluation_rankings.csv
```

This file contains combined rankings across all test cases, ready for **competition submission**.

---

## 💻 Hardware Requirements

- Verified on **NVIDIA A100 GPUs**  
- Supports any CUDA-capable GPU with adequate memory (≥16 GB recommended)

---

## 🧠 Workshop Information

### 📍 WACV 2026 Workshop
**Title:** Video-based Human Recognition at Extreme Far Distances (VReID-XFD)  
**Type:** Half-day workshop with competition track  
**Website:** Coming soon  
**Contact:** Prof. Hugo Proença – hugomcp@ubi.pt  
**Dataset:** [DetReIDX](https://www.it.ubi.pt/DetReIDX/)

---

### 🧾 Call for Papers

We invite original research on:

- UAV-based and long-range person re-identification  
- Domain adaptation and viewpoint generalization  
- Multi-modal and low-resolution re-ID  
- Tracking, detection, and recognition under extreme conditions  

Accepted **full papers (6–8 pages)** will appear in **CVF Open Access proceedings**.

---

### 🎯 Motivation and Impact

Most existing UAV-based re-identification benchmarks consider moderate distances and short-term sessions.  
The **VReID-XFD workshop** and its **DetReIDX Challenge** focus instead on **extreme distances (up to 120m)**, with drastic **appearance, illumination, and viewpoint changes**.

Objectives:
- Release the **DetReIDX benchmark** for long-range video re-ID  
- Provide reproducible **baselines and evaluation protocols**  
- Build a community around **extreme-distance person recognition**

---

### 👥 Organizers

- **Prof. Hugo Proença**, University of Beira Interior, Portugal  
- **Prof. Fernando Alonso-Fernandez**, Halmstad University, Sweden  
- **Prof. Mayank Vatsa**, IIT Jodhpur, India  
- **Dr. Kien Nguyen**, Queensland University of Technology, Australia  
- **Prof. Vitomir Štruc**, University of Ljubljana, Slovenia  
- **Kailash Hambarde**, Instituto de Telecomunicações, Portugal  

---

### 🙏 Acknowledgment

This baseline implementation is adapted from [VSLA-CLIP](https://github.com/FHR-L/VSLA-CLIP).  
We sincerely thank the original authors for their contribution.
