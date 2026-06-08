<div align="center">

<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
<img src="https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" />
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
<img src="https://img.shields.io/badge/License-MIT-10B981?style=for-the-badge" />
<img src="https://img.shields.io/badge/Open%20Source-%E2%9D%A4-F43F5E?style=for-the-badge" />

<br /><br />

```
███╗   ███╗██╗  ██╗███████╗██████╗       ████████╗██████╗
████╗ ████║██║  ██║██╔════╝██╔══██╗      ╚══██╔══╝██╔══██╗
██╔████╔██║███████║█████╗  ██║  ██║         ██║   ██║  ██║
██║╚██╔╝██║██╔══██║██╔══╝  ██║  ██║         ██║   ██║  ██║
██║ ╚═╝ ██║██║  ██║███████╗██████╔╝         ██║   ██████╔╝
╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝╚═════╝          ╚═╝   ╚═════╝
```

# Mental Health Emotion Detection from Textual Data

**A fine-tuned RoBERTa transformer for 28-class multi-label emotion detection,  
with a real-time Streamlit dashboard, psychometric Wellness Score, and PDF reporting.**

[🚀 Quick Start](#-quick-start) • [📖 Docs](#-project-structure) • [🤝 Contribute](#-contributing) • [🐛 Report Bug](#-reporting-bugs--issues)

</div>

---

## 📋 Table of Contents

- [About the Project](#-about-the-project)
- [Key Features](#-key-features)
- [Demo](#-demo)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Model Details](#-model-details)
- [Dataset](#-dataset)
- [Performance](#-performance)
- [Wellness Score Formula](#-wellness-score-formula)
- [Configuration](#-configuration)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [Reporting Bugs & Issues](#-reporting-bugs--issues)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

---

## 🧠 About the Project

MHED-TD (**M**ental **H**ealth **E**motion **D**etection from **T**extual **D**ata) is an open-source NLP system that goes beyond simple positive/negative sentiment analysis. Traditional digital wellness tools reduce human emotion to 3–5 coarse labels. Real emotional experiences are far more nuanced — a single journal entry might contain **grief, remorse, love, and loneliness simultaneously**.

MHED-TD solves this with a **multi-label transformer architecture** trained on Google's GoEmotions corpus, detecting **28 fine-grained emotions** at once and translating them into an actionable, real-time **Wellness Score (0–100)**.

```
Traditional Approach          MHED-TD Approach
─────────────────────         ──────────────────────────────────────
"I feel sad"                  "I feel sad"
    │                              │
    ▼                              ▼
 Sad: 0.9                    Grief:        0.65 ██████░░░░
 Happy: 0.1                  Remorse:      0.45 ████░░░░░░
                             Love:         0.30 ███░░░░░░░
                             Loneliness:   0.55 █████░░░░░
                                  │
                                  ▼
                          Wellness Score: 42 / 100
                          → Moderate Distress Detected
```

> ⚠️ **Disclaimer:** MHED-TD is a supportive emotional awareness tool — **not a clinical diagnostic instrument**. It does not replace professional psychological evaluation or therapy.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🎯 **28-Label Multi-Label Classification** | Detects all GoEmotions categories simultaneously using Sigmoid activation (not Softmax) |
| 🧮 **Psychometric Wellness Score** | Algorithmic formula maps raw probabilities → intuitive 0–100 wellness index |
| 📊 **Interactive Visualizations** | Real-time Radar, Bar, and Pie charts powered by Plotly |
| 🕒 **Session Timeline Tracking** | Tracks and charts emotional trends across multiple entries per session |
| 📄 **On-Demand PDF Reports** | ReportLab-generated clinical summary reports downloadable in-browser |
| 🔒 **Privacy-by-Design** | All data processed in-memory only — nothing written to disk or sent to any server |
| ⚡ **Low Latency** | Inference completes in < 2 seconds (GPU) / < 5 seconds (CPU) |
| 🎨 **Glassmorphism UI** | Custom CSS-injected Streamlit dashboard with frosted glass aesthetic |
| 🚨 **Risk Threshold Alerts** | Automatic distress alerts with evidence-based coping suggestions |

---

## 🎬 Demo

```
╔══════════════════════════════════════════════════════════╗
║         MHED-TD Diagnostic Dashboard                    ║
╠══════════════════════════════════════════════════════════╣
║  Input:  "I'm so grateful for everyone around me,       ║
║           feeling truly blessed today."                 ║
╠══════════════════════════════════════════════════════════╣
║  Wellness Score:  [ 94 / 100 ] ✅ Positive Balance      ║
║                                                          ║
║  Top Emotions Detected:                                  ║
║  ┌─────────────┬──────────────────────────┬──────────┐  ║
║  │ Gratitude   │ ██████████████████████░░ │  94%     │  ║
║  │ Love        │ ██████████████████░░░░░░ │  76%     │  ║
║  │ Optimism    │ ████████████░░░░░░░░░░░░ │  48%     │  ║
║  │ Joy         │ █████████░░░░░░░░░░░░░░░ │  38%     │  ║
║  └─────────────┴──────────────────────────┴──────────┘  ║
╚══════════════════════════════════════════════════════════╝
```

---

## 🛠 Tech Stack

```
┌─────────────────────────────────────────────────┐
│                 Streamlit Dashboard              │
│         Glassmorphism CSS  ·  Plotly Charts      │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│              PyTorch Deep Learning               │
│      Fine-tuned RoBERTa-base  ·  CUDA Inference  │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│           Hugging Face Transformers              │
│       BPE Tokenizer  ·  Safetensors Cache        │
└─────────────────────────────────────────────────┘
```

| Layer | Library | Version |
|---|---|---|
| Deep Learning | `torch` | 2.1+ |
| Transformers | `transformers` | 4.35+ |
| Dataset | `datasets` | 2.14+ |
| Metrics | `evaluate` | 0.4+ |
| Web UI | `streamlit` | 1.28+ |
| Visualization | `plotly` | 5.17+ |
| PDF Generation | `reportlab` | 4.0+ |
| Data Processing | `numpy`, `pandas`, `scikit-learn` | latest |
| Hardware Acceleration | CUDA Toolkit | 11.8 / 12.1 |

---

## 📁 Project Structure

```
MHED-TD/
│
├── 📂 app/
│   ├── app.py                  # Main Streamlit application entry point
│   ├── styles.css              # Custom Glassmorphism CSS styles
│   └── components/
│       ├── charts.py           # Plotly radar, bar, pie chart renderers
│       ├── sidebar.py          # Session history sidebar
│       └── recommendations.py  # Wellness threshold & coping suggestions
│
├── 📂 src/
│   ├── model.py                # RobertaMultiLabelClassifier class definition
│   ├── predict.py              # EmotionPredictionPipeline + Wellness Score
│   ├── preprocess.py           # Tokenization, data cleaning utilities
│   └── report.py               # ReportLab PDF generation module
│
├── 📂 training/
│   ├── train.py                # Fine-tuning script (Trainer API)
│   ├── dataset.py              # GoEmotions loader & multi-hot encoder
│   ├── class_weights.py        # Inverse frequency weight computation
│   └── evaluate.py             # Per-class F1, Precision, Recall metrics
│
├── 📂 models/
│   └── mhed_roberta/           # Fine-tuned model weights (Safetensors)
│       ├── config.json
│       ├── tokenizer_config.json
│       └── model.safetensors   # ← download separately (see Quick Start)
│
├── 📂 tests/
│   ├── test_wellness.py        # Unit tests for Wellness Score algorithm
│   ├── test_predict.py         # Inference pipeline tests
│   └── test_report.py          # PDF generation tests
│
├── 📂 outputs/
│   └── logs/                   # Training logs and eval outputs
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- pip 23.0+
- (Optional but recommended) NVIDIA GPU with CUDA 11.8+

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/MHED-TD.git
cd MHED-TD
```

### 2. Create a Virtual Environment

```bash
# Unix / macOS
python -m venv mhed_env
source mhed_env/bin/activate

# Windows
python -m venv mhed_env
mhed_env\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Or install manually:

```bash
pip install torch transformers datasets evaluate scikit-learn \
            numpy pandas streamlit plotly reportlab accelerate
```

### 4. Download the Fine-Tuned Model

> The model weights are hosted separately due to file size. Download and place in `models/mhed_roberta/`.

```bash
# Using Hugging Face Hub (recommended)
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='yourusername/mhed-roberta', local_dir='models/mhed_roberta')
"
```

> Alternatively, you can **train from scratch** — see [Training from Scratch](#training-from-scratch).

### 5. Launch the Dashboard

```bash
streamlit run app/app.py
```

Open your browser at `http://localhost:8501` 🎉

---

### Training from Scratch

If you want to fine-tune the model yourself on the GoEmotions dataset:

```bash
# Downloads GoEmotions automatically via Hugging Face Datasets
python training/train.py \
  --model_name roberta-base \
  --output_dir models/mhed_roberta \
  --num_train_epochs 4 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --fp16
```

Expected training time: ~2 hours on an NVIDIA RTX 3060 / ~8 hours on CPU.

---

## 🤖 Model Details

### Architecture

```
Input Text
    │
    ▼
RobertaTokenizerFast  (BPE, max_length=128, vocab=50K)
    │
    ▼
RoBERTa-base Encoder
  ├── 12 Transformer Layers
  ├── 768 Hidden Units per Layer
  └── 12 Self-Attention Heads
    │
    ▼
[CLS] Pooler Output  (768-dim)
    │
    ▼
Dropout (p=0.2)
    │
    ▼
Linear Layer  (768 → 28)
    │
    ▼
Sigmoid Activation  (independent per class)
    │
    ▼
28 Probability Scores  [0.0 – 1.0]
```

### Why RoBERTa over BERT?

| Improvement | BERT | RoBERTa |
|---|---|---|
| Masking Strategy | Static (preprocessed once) | Dynamic (new mask per epoch) |
| NSP Objective | Yes | Removed |
| Training Corpus | 16 GB | **160 GB** (CC-News, OpenWebText, Stories) |
| Batch Size | 256 | Up to **8,000** |
| BPE Vocab Size | 30K | **50K** |
| Casual Text Handling | Weak | Strong |

### Training Configuration

```python
TrainingArguments(
    num_train_epochs        = 4,
    per_device_train_batch_size = 32,
    learning_rate           = 2e-5,
    warmup_ratio            = 0.1,
    weight_decay            = 0.01,
    fp16                    = True,
    metric_for_best_model   = "f1",
    load_best_model_at_end  = True,
)
```

### Training Loss Progression

```
Epoch 1/4:  Train Loss: 0.1242  │  Eval Loss: 0.0984  │  F1-Micro: 0.514
Epoch 2/4:  Train Loss: 0.0864  │  Eval Loss: 0.0842  │  F1-Micro: 0.548
Epoch 3/4:  Train Loss: 0.0621  │  Eval Loss: 0.0792  │  F1-Micro: 0.569
Epoch 4/4:  Train Loss: 0.0412  │  Eval Loss: 0.0764  │  F1-Micro: 0.582
```

---

## 📊 Dataset

The model is trained on the **[GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions)** dataset published by Google Research (2020).

| Property | Value |
|---|---|
| Total Samples | 58,009 Reddit comments |
| Emotion Classes | 28 (27 discrete + Neutral) |
| Human Raters | 82 independent annotators |
| Annotation Method | Consensus multi-label |
| Split | Train / Validation / Test |

### Emotion Taxonomy

| Group | Emotions |
|---|---|
| 🟢 **Positive** | Admiration, Amusement, Approval, Caring, Desire, Excitement, Gratitude, Joy, Love, Optimism, Pride, Relief |
| 🔴 **Negative / Stress** | Anger, Annoyance, Disappointment, Disapproval, Disgust, Embarrassment, Fear, Grief, Nervousness, Remorse, Sadness |
| 🔵 **Neutral / Ambiguous** | Confusion, Curiosity, Realization, Surprise, Neutral |

### Class Imbalance Note

The dataset is highly imbalanced. `Neutral` has ~14,210 samples while `Grief` has only ~77. We address this using **inverse class-frequency weighted loss**:

```python
# Compute class weights
from sklearn.utils.class_weight import compute_class_weight
weights = compute_class_weight('balanced', classes=np.arange(28), y=flat_labels)
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weights))
```

---

## 📈 Performance

Evaluated on the GoEmotions test split (6,329 samples):

### Aggregate Metrics

| Metric | Score |
|---|---|
| Micro-Average F1 | **0.58** |
| Micro-Average Precision | **0.72** |
| Micro-Average Recall | **0.49** |
| Macro-Average F1 | 0.44 |
| Weighted-Average F1 | 0.56 |

### Per-Class Highlights

| Emotion | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Gratitude | 0.94 | 0.89 | **0.92** | 352 |
| Amusement | 0.80 | 0.83 | **0.82** | 264 |
| Love | 0.76 | 0.85 | **0.80** | 238 |
| Remorse | 0.61 | 0.86 | **0.71** | 56 |
| Fear | 0.77 | 0.59 | **0.67** | 78 |
| Neutral | 0.75 | 0.55 | **0.64** | 1787 |
| Grief | 0.00 | 0.00 | **0.00** ⚠️ | 6 |
| Nervousness | 0.00 | 0.00 | **0.00** ⚠️ | 23 |

> ⚠️ **Known Limitation:** Rare classes (Grief, Nervousness, Pride, Relief) show 0.00 F1 due to extreme long-tail sparsity. This is an active area for contribution — see [Roadmap](#-roadmap).

---

## 🧮 Wellness Score Formula

The Wellness Score `W` aggregates the 28 predicted probabilities into a single 0–100 index:

```
W = 100 × [ α·P̄ + β·Ā − γ·S̄ ]

Where:
  P̄  = Mean probability of Positive Affect emotions
  Ā  = Mean probability of Neutral/Ambiguous emotions × 0.5
  S̄  = Mean probability of Stress/Negative emotions

  α  = 0.60   (positive weight)
  β  = 0.20   (ambiguous weight)
  γ  = 0.80   (stress penalty)
```

### Wellness Thresholds

| Score Range | Status | System Response |
|---|---|---|
| **W ≥ 60** | ✅ Stable Baseline | Gratitude journaling & mindfulness suggestions |
| **40 ≤ W < 60** | 🟡 Moderate Distress | Guided breathing & cognitive-reframing techniques |
| **W < 40** | 🔴 Critical Distress | Immediate support resources & crisis hotline display |

---

## ⚙️ Configuration

Key settings can be modified in `src/predict.py`:

```python
# Inference threshold — lower = more emotions detected, higher = more precise
PREDICTION_THRESHOLD = 0.3

# Device selection
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Max token length
MAX_LENGTH = 128
```

Wellness Score weights can be tuned in `src/predict.py`:

```python
WELLNESS_WEIGHTS = {
    "positive_alpha": 0.60,
    "ambiguous_beta": 0.20,
    "stress_gamma":   0.80,
}
```

---

## 🗺 Roadmap

Planned features and known improvements:

- [ ] 🔧 **Fix rare-class F1** — oversample Grief, Nervousness, Pride, Relief using SMOTE or data augmentation
- [ ] 🌐 **Multilingual support** — fine-tune `xlm-roberta-base` for Hindi, Spanish, French
- [ ] 🎙️ **Multimodal analysis** — fuse text embeddings with voice tone features
- [ ] ☁️ **Encrypted cloud sync** — optional end-to-end encrypted session history
- [ ] 🏥 **Clinical integration API** — REST endpoint for therapist portal integration
- [ ] 📱 **Mobile-responsive UI** — improve Streamlit layout for small screens
- [ ] 🧪 **Expand test coverage** — add edge case tests for adversarial/sarcastic inputs
- [ ] 🐳 **Docker image** — containerized one-command deployment
- [ ] 📦 **PyPI package** — installable `pip install mhed-td` CLI

---

## 🤝 Contributing

**MHED-TD is open source and we warmly welcome contributions of all kinds!**

Whether you're fixing a typo, improving model performance, adding a new feature, or translating the UI — your help is valued. Here's how to get started:

### Ways to Contribute

- 🐛 **Bug fixes** — found something broken? Open an issue or submit a fix
- ✨ **New features** — check the [Roadmap](#-roadmap) for ideas or propose your own
- 📈 **Model improvements** — better training strategies, augmentation, new architectures
- 🌍 **Translations** — help make MHED-TD accessible in more languages
- 📖 **Documentation** — improve clarity, add examples, fix typos
- 🧪 **Tests** — expand test coverage for edge cases

### Contribution Workflow

```bash
# 1. Fork the repository on GitHub
# 2. Clone your fork
git clone https://github.com/YOUR-USERNAME/MHED-TD.git
cd MHED-TD

# 3. Create a new branch
git checkout -b feature/your-feature-name
#   or
git checkout -b fix/bug-description

# 4. Make your changes and commit
git add .
git commit -m "feat: add multilingual support for Hindi"
#   or
git commit -m "fix: correct wellness score calculation for neutral-only inputs"

# 5. Push your branch
git push origin feature/your-feature-name

# 6. Open a Pull Request on GitHub
```

### Commit Message Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

| Prefix | Use For |
|---|---|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `docs:` | Documentation only |
| `test:` | Adding or fixing tests |
| `refactor:` | Code change that isn't a fix or feature |
| `perf:` | Performance improvement |
| `chore:` | Build system, dependencies |

### Pull Request Guidelines

- Keep PRs focused — one feature or fix per PR
- Include a clear description of what changed and why
- Add or update tests where applicable
- Ensure `python -m pytest tests/` passes before submitting
- Reference any related issue numbers (e.g. `Closes #42`)

### Development Setup

```bash
# Install dev dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests
python -m pytest tests/ -v

# Format code
black src/ app/ training/

# Lint
flake8 src/ app/ training/ --max-line-length=100
```

---

## 🐛 Reporting Bugs & Issues

Found a bug? We want to know! Please open an issue on GitHub:

**[👉 Open a Bug Report](https://github.com/yourusername/MHED-TD/issues/new?template=bug_report.md)**

### What to Include in a Bug Report

```markdown
## Bug Description
A clear description of what went wrong.

## Steps to Reproduce
1. Go to '...'
2. Enter text '...'
3. Click '...'
4. See error

## Expected Behaviour
What you expected to happen.

## Actual Behaviour
What actually happened.

## Environment
- OS: [e.g. Ubuntu 22.04 / Windows 11 / macOS Sonoma]
- Python version: [e.g. 3.10.12]
- PyTorch version: [e.g. 2.1.0]
- GPU: [e.g. RTX 3060 / CPU only]

## Error Log / Traceback
Paste the full traceback here.
```

### Feature Requests

Have an idea to improve MHED-TD?

**[👉 Open a Feature Request](https://github.com/yourusername/MHED-TD/issues/new?template=feature_request.md)**

---

## 📜 License

This project is licensed under the **MIT License** — you are free to use, modify, and distribute this code for personal or commercial purposes.

```
MIT License

Copyright (c) 2026 MHED-TD Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software...
```

See the full [LICENSE](./LICENSE) file for details.

---

## 🙏 Acknowledgements

- **[Google Research](https://github.com/google-research/google-research/tree/master/goemotions)** — for the GoEmotions dataset
- **[Hugging Face](https://huggingface.co/)** — for the Transformers library and model hub
- **[Liu et al. (2019)](https://arxiv.org/abs/1907.11692)** — for the RoBERTa architecture
- **[Streamlit](https://streamlit.io/)** — for the web application framework
- **[Plotly](https://plotly.com/)** — for the interactive visualization library
- **[CORE-OM / PSS](https://novopsych.com/assessments/outcome-monitoring/clinical-outcomes-in-routine-evaluation-core-om/)** — for the clinical threshold inspiration

---

<div align="center">

**Made with ❤️ and open to the world**

If MHED-TD helped you, please consider giving it a ⭐ on GitHub — it helps others find this project!

[⭐ Star on GitHub](https://github.com/yourusername/MHED-TD) • [🐛 Report Bug](https://github.com/yourusername/MHED-TD/issues) • [💡 Request Feature](https://github.com/yourusername/MHED-TD/issues)

</div>
