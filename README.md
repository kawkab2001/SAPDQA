# ERPCRAFT: Parameter-Efficient Domain Adaptation for Context-Aware ERP Question Answering

ERPCRAFT is a research codebase for building and evaluating **domain-adapted, context-aware Question Answering (QA) systems for Enterprise Resource Planning (ERP) documentation** (e.g., SAP Help content). The repository contains the data collection and preprocessing pipeline used to construct a SAP-domain QA dataset, together with fine-tuning and evaluation notebooks/scripts for several extractive and generative QA models (BERT, RoBERTa, T5, Qwen) as well as a zero-shot ("without fine-tuning") baseline.

## Repository Structure

```
ERPCRAFT/
├── code/
│   ├── preprocessing/
│   │   ├── collection with scroll.py        # Web scraping w/ scroll-based pagination (SAP Help pages)
│   │   ├── collection without scroll.py      # Web scraping without scroll-based pagination
│   │   ├── combine.py                        # Merges raw collected data into a unified corpus
│   │   ├── preprocessing 1.py                # Cleans / normalizes scraped text, builds QA pairs
│   │   ├── title.py                          # Extracts/derives document titles
│   │   ├── sample of html code of sap page web _what.html   # Example raw source page
│   │   └── sample of colecct and step one of generate in _wich/  # Example intermediate collection output
│   └── fine_tuning/
│       ├── bert.ipynb                        # Fine-tuning & evaluation: BERT
│       ├── roberta.ipynb                     # Fine-tuning & evaluation: RoBERTa
│       ├── t5.ipynb                          # Fine-tuning & evaluation: T5
│       ├── qwen.ipynb                        # Parameter-efficient fine-tuning & evaluation: Qwen
│       └── without_fine_tuning.py            # Zero-shot / no fine-tuning baseline
├── dataset/
│   ├── dataset_QA.json                       # SQuAD-style ERP/SAP QA dataset (title/context/QA pairs)
│   └── dataset_QA (1).json                   # Additional dataset variant/version
├── environment                               # Hardware / OS / driver notes for the experiments
└── requirements                              # Python dependencies (pip-installable)
```

## Dataset

`dataset/dataset_QA.json` is a **SQuAD-style JSON** dataset built from ERP (SAP Help) documentation. Each entry contains a `title`, one or more `paragraphs`, each with a `context` (a documentation passage) and associated question–answer pairs, enabling both extractive and generative QA training.

The dataset was produced with the scripts in `code/preprocessing/`:

1. **Collection** — `collection with scroll.py` / `collection without scroll.py` scrape SAP Help / ERP documentation pages (see the included HTML sample for the source page structure).
2. **Combination** — `combine.py` merges the raw collected pages into a single corpus.
3. **Cleaning & QA generation** — `preprocessing 1.py` and `title.py` normalize the text and construct the final title/context/QA structure saved to `dataset/dataset_QA.json`.

## Models

The `code/fine_tuning/` folder contains one notebook/script per model family evaluated in the paper:

| File | Model | Adaptation strategy |
|---|---|---|
| `bert.ipynb` | BERT | Full/extractive fine-tuning |
| `roberta.ipynb` | RoBERTa | Full/extractive fine-tuning |
| `t5.ipynb` | T5 | Fine-tuning (generative QA) |
| `qwen.ipynb` | Qwen | Parameter-efficient fine-tuning (PEFT/LoRA via Unsloth) |
| `without_fine_tuning.py` | — | Zero-shot baseline (no fine-tuning) |

Evaluation uses standard QA metrics (Exact Match / F1, ROUGE, BLEU/SacreBLEU) and semantic similarity via Sentence-Transformers embeddings.

## Environment

Experiments were run across two setups (see `environment` for full details):

- **Local (CPU) — preprocessing:** Intel Core i5-1335U, 16 GB RAM, Python 3.10, Windows 11, no GPU.
- **Remote (GPU) — fine-tuning, preprocessing, testing:** NVIDIA Tesla V100-PCIE-16GB, CUDA 12.6, Driver 560.35.03, PyTorch 2.6.0, Ubuntu, virtual environment `.ve-jhub`.

## Installation & Reproducibility

To reproduce the experiments, follow these steps:

### 1. Clone the repository

```bash
git clone https://github.com/ELTE-DSED/ERPCRAFT.git
cd ERPCRAFT
```

### 2. Create and activate a virtual environment

```bash
# Python 3.10 recommended
python3.10 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements
```

Key dependencies (see `requirements` for exact pinned versions):

- **Core:** `torch==2.6.0`, `transformers==4.53.2`/`4.46.0`, `datasets==3.5.0`
- **PEFT/fine-tuning:** `unsloth==2025.3.18`, `unsloth_zoo==2025.3.16`, `peft==0.15.2`, `trl==0.15.2`, `accelerate==1.5.2`, `bitsandbytes==0.45.5`
- **Evaluation:** `nltk==3.9.1`, `rouge==1.0.1`, `scikit-learn==1.6.1`, `sacrebleu==2.5.1`, `evaluate==0.4.3`, `sentence-transformers==4.1.0`
- **Utilities:** `matplotlib==3.10.1`, `tqdm==4.67.1`, `numpy==2.2.4`

Additional one-time setup:

```bash
python -c "import nltk; nltk.download('punkt')"
```

If you plan to run the zero-shot baseline in `without_fine_tuning.py` with local LLMs, install [Ollama](https://ollama.com) manually and pull the required model(s).

### 4. (Optional) Regenerate the dataset from scratch

```bash
cd code/preprocessing
python "collection with scroll.py"        # or "collection without scroll.py"
python combine.py
python "preprocessing 1.py"
python title.py
```

This reproduces `dataset/dataset_QA.json`. Since the target site's content and structure may change over time, using the provided `dataset/dataset_QA.json` directly is recommended for exact reproducibility of the reported results.

### 5. Reproduce the fine-tuning / evaluation results

A GPU with ≥16 GB VRAM (e.g., NVIDIA V100-16GB) is recommended, matching the paper's setup.

```bash
cd code/fine_tuning
jupyter notebook bert.ipynb       # BERT
jupyter notebook roberta.ipynb    # RoBERTa
jupyter notebook t5.ipynb         # T5
jupyter notebook qwen.ipynb       # Qwen (PEFT/LoRA)
python without_fine_tuning.py     # Zero-shot baseline
```

Each notebook/script loads `dataset/dataset_QA.json`, splits it into train/validation/test sets, fine-tunes (where applicable) the corresponding model, and reports QA/generation metrics (Exact Match, F1, ROUGE, BLEU/SacreBLEU, and semantic similarity).

> **Note:** Notebooks may need path adjustments (e.g., dataset location, output directories) depending on where you run them (local machine vs. remote GPU/Jupyter server).

## Citation

If you use this repository or dataset, please cite:

```bibtex
@article{erpcraft2026,
  title   = {ERPCRAFT: Parameter-Efficient Domain Adaptation for Context-Aware ERP Question Answering},
  author  = {Bouressace, Kawkab and Arafat, Md Easin and Saha, Sourav and Orosz, Tam\'as and Bouressace, Hassina},
  year    = {2026}
}
```

## Authors

- **Kawkab Bouressace**\* — kawkab@inf.elte.hu — Eötvös Loránd University (ELTE)
- **Md Easin Arafat**\* (Corresponding author) — arafatmdeasin@inf.elte.hu — Eötvös Loránd University (ELTE)
- **Sourav Saha** — research.srv.sh@gmail.com
- **Tamás Orosz** — orosztamas@inf.elte.hu — Eötvös Loránd University (ELTE)
- **Hassina Bouressace** — bouressace.hassina@univ-guelma.dz — University 8 Mai 1945, Guelma, Algeria

\* Equal contribution.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
