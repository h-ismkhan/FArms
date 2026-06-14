# In the name of God
# FArms & wArms — Federated Multi-Modal Learning with Simulating Arms for Missing Modalities

> Official implementation of the paper **"FArms: Federated Multi-Modal Learning with Simulating Arms for Missing Modalities"** (to be submitted as soon as possible).

FArms handles arbitrary missing modalities in multi-modal learning by training lightweight **simulating arms** — bidirectional pairwise simulators and multi-head-attention-based fusion simulators — that reconstruct missing modality embeddings on the fly from available ones. The framework supports both **centralized** (`wArms`) and **federated** (`FArms`) settings, and works for both classification and regression tasks.

---

## Repository Structure

```
FArms_wArms/
├── FArm_wArm/
│   ├── FArm-MMIMDb/          # Federated setting — MM-IMDb (image + text classification)
│   │   ├── farm/
│   │   │   ├── task.py       # Model, dataset, train/test logic
│   │   │   ├── client_app.py # Flower ClientApp
│   │   │   └── server_app.py # Flower ServerApp (runs all missing configs)
│   │   └── pyproject.toml    # Flower app config & hyperparameters
│   │
│   ├── FArm-MOSI/            # Federated setting — CMU-MOSI (text + audio + video regression)
│   │   ├── farm/
│   │   │   ├── task.py       # Model, dataset, train/test logic
│   │   │   ├── mosi_reg.py   # MOSI dataset loader
│   │   │   ├── client_app.py # Flower ClientApp
│   │   │   └── server_app.py # Flower ServerApp with two-stage round-aware strategy
│   │   └── pyproject.toml
│   │
│   └── wArm-MOSI/            # Centralized setting — CMU-MOSI (standalone script)
│       ├── mosi_train_reg.py # Full training + evaluation script
│       └── mosi_reg.py       # MOSI dataset loader
```

---

## Method Overview

FArms separates missing-modality handling from the final task into a modular **blue-box / green-box** design:

- **Blue-box (simulating arms):** Pairwise simulators `S_{i→j}` and fusion-based simulators trained with ℓ₂ reconstruction losses across all modality availability cases.
- **Green-box (task head):** Multi-head self-attention fusion followed by a task-specific MLP head for classification or regression.
- **Two-stage optimization:** Simulators are trained first (`Rsim` rounds/epochs), then frozen while the task head is fine-tuned. This stabilizes cross-modal reconstruction before task-specific training and reduces communication overhead in the federated setting.
- **Frozen encoders:** Pre-trained encoders (CLIP, BERT, WavLM) are kept frozen throughout, keeping per-client compute low.

---

## Datasets

### MM-IMDb (Multi-label movie genre classification)
- Modalities: **image** + **text**
- 27 genre classes (multi-label)
- Download: [MM-IMDb](https://www.kaggle.com/datasets/eduardschipatecua/mmimdb)

Expected directory layout:
```
mmimdb/
├── split.json
└── dataset/
    ├── <id>.json
    └── <id>.jpeg
```

### CMU-MOSI (Sentiment regression / binary classification)
- Modalities: **text** + **audio** + **video**
- Encoders: BERT (`bert-base-uncased`), WavLM-Large (`microsoft/wavlm-large`), CLIP ViT-B/32
- Download: [CMU-MOSI](http://multicomp.cs.cmu.edu/resources/cmu-mosi-dataset/)

Expected directory layout:
```
Raw - CMU Multimodal Opinion Sentiment Intensity/
├── Audio/WAV_16000/Segmented/
├── Video/Segmented/
├── Transcript/Segmented/
└── mosi_splits-70train.json
```

---

## Installation

### 1. Create environment

```bash
conda create -n farms python=3.12
conda activate farms
```

### 2. Install PyTorch

```bash
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install shared dependencies

```bash
pip install transformers scikit-learn Pillow numpy tqdm matplotlib seaborn scipy
```

### 4. Install CLIP (for MM-IMDb)

```bash
pip install git+https://github.com/openai/CLIP.git
```

### 5. For audio (CMU-MOSI only)

```bash
pip install torchaudio soundfile opencv-python
```

### 6. For federated experiments (`FArm-MMIMDb` and `FArm-MOSI`)

```bash
pip install "flwr[simulation]>=1.22.0" "flwr-datasets[vision]>=0.5.0" hatchling
```

---

## Configuration

### Dataset paths

Before running, update the dataset root paths inside the relevant script:

| Script | Variable to update |
|--------|-------------------|
| `FArm-MMIMDb/farm/task.py` | `MMIMDB_ROOT` |
| `FArm-MOSI/farm/task.py` | `_dpath` |
| `wArm-MOSI/mosi_train_reg.py` | `dsspath` |

### Missing modality patterns

**MM-IMDb** uses two pattern formats:
- `α_image_β_text` — e.g., `100_image_20_text` means 100% image present, 20% text present
- `complex_γ_α_β` — e.g., `complex_20_40_40` means 20% both present, 40% image only, 40% text only

**CMU-MOSI** uses two pattern formats:
- `X_text_Y_audio_Z_video` — e.g., `20_text_100_audio_20_video`
- `complex_TAV_T_A_V_TA_TV_AV` — e.g., `complex_20_20_20_10_10_10_10`

---

## Running the Code

### Centralized — wArms on CMU-MOSI

```bash
cd FArm_wArm/wArm-MOSI
python mosi_train_reg.py
```

To run specific missing configurations, pass them as command-line arguments:

```bash
python mosi_train_reg.py 100_text_100_audio_100_video 20_text_100_audio_20_video
```

The script runs **10 independent runs** per configuration by default and reports averaged F-micro with statistical significance (paired t-test).

**Default configurations evaluated:**
```
100_text_100_audio_100_video
100_text_20_audio_20_video
20_text_100_audio_20_video
20_text_20_audio_100_video
20_text_100_audio_100_video
100_text_100_audio_20_video
100_text_20_audio_100_video
complex_20_20_20_10_10_10_10
```

---

### Federated — FArms on MM-IMDb

```bash
cd FArm_wArm/FArm-MMIMDb
flwr run .
```

Configure runs via `pyproject.toml`:

```toml
[tool.flwr.app.config]
num-server-rounds = 2          # Total FL rounds (first 2 train simulators only)
num-supernodes = 10            # Number of federated clients
fraction-train = 0.5           # Fraction of clients selected per round
local-epochs = 5
lr = 0.001
alpha = 1.0                    # Weight for simulation loss
beta = 1.0                     # Weight for classification loss
missing-configs = "100_image_100_text,100_image_20_text,20_image_100_text,complex_20_40_40"
```

The server iterates over all `missing-configs` automatically, saving per-config model checkpoints and results:
```
final_model-n_10-r_2-e_5-<config>.pt
results-n_10-r_2-e_5-<config>.json
```

---

### Federated — FArms on CMU-MOSI

```bash
cd FArm_wArm/FArm-MOSI
flwr run .
```

Configure runs via `pyproject.toml`:

```toml
[tool.flwr.app.config]
num-server-rounds = 5
num-rounds-for-sim-train = 2   # Rounds dedicated to simulator training (two-stage)
fraction-train = 0.5
local-epochs = 8
num-supernodes = 10
lr = 0.0005
alpha = 0.1
beta = 1.0
missing-configs = "20_text_100_audio_100_video, 100_text_20_audio_100_video, 100_text_100_audio_20_video"
```

> **Note on two-stage training:** In both federated experiments, the first `num-rounds-for-sim-train` rounds optimize only the simulation arms (blue-box). The remaining rounds freeze the simulators and train only the task head (green-box). This matches Algorithm 2 in the paper.

---

## Hyperparameters

| Parameter | MM-IMDb | CMU-MOSI |
|-----------|---------|----------|
| Encoder | CLIP ViT-B/32 (512-dim) | BERT (768-dim) + WavLM-Large (1024-dim) + CLIP ViT-B/32 (512-dim) |
| Simulator hidden size | 256 | — |
| Shared embedding dim | 256 | 256 |
| Task head | 256→128→64→27 (MLP + ReLU + 0.1 dropout) | MLP + ReLU + dropout |
| Optimizer | Adam | Adam |
| Learning rate | 5×10⁻⁵ (federated) | 5×10⁻⁴ (centralized), 5×10⁻⁴ (federated) |
| Local epochs | 10 (federated) | 8 (federated), — (centralized) |
| FL rounds | 4 | 5–10 |
| Simulator-only rounds | 2 | 2 |
| Clients | 10 (50% per round) | 10 (50% per round) |

---

## Results Summary

FArms achieves state-of-the-art performance under severe missing modality regimes (>50% aggregate absence), outperforming ShaSpec, M³Care, PmcmFL, Flex-MoE, and PEPSY on both datasets.

| Dataset | Metric | FArms (federated) |
|---------|--------|-------------------|
| MM-IMDb | F-micro | **0.93–0.94** (all configs) |
| CMU-MOSI | F-micro | **0.67–0.82** (all configs) |

FArms converges in as few as **5–10 communication rounds** under non-IID data with 50% client participation per round, compared to 50–200 rounds required by prior federated methods.

See the paper for full tables, t-test / Wilcoxon significance results, and t-SNE embedding visualizations.

---

## Citation

```bibtex
@inproceedings{farms2026,
  title     = {FArms: Federated Multi-Modal Learning with Simulating Arms for Missing Modalities},
  booktitle = {?},
  year      = {2026}
}
```

---

## License

This project is licensed under the Apache 2.0 License. See `LICENSE` for details.
