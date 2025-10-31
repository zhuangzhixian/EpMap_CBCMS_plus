# EpMap for CBCMS+ (Enhanced PDL Mapping Pipeline)

This repository implements the **Enhanced Policy Definition Language (PDL) Mapping Pipeline**, referred to as **EpMap**, for the CBCMS+ system. EpMap leverages a **RoBERTa-based** model to extract structured policy signals from multilingual legal texts and map them to machine-readable PDL constructs that support cross-jurisdictional compliance.

## Features

- **Transformer Backbone (RoBERTa)**: Fine-tuned contextual representations tailored to regulatory text.
- **Structured PDL Mapping**: Converts unstructured clauses into policy labels aligned with the PDL schema.
- **Multi-Label Outputs**: Supports multiple policy tags per clause to capture composite obligations.
- **Reproducible Workflow**: Clear data layout, fixed seeds in code, and deterministic preprocessing.

## Project Structure

```
EpMap_CBCMS_plus/
├── data/
│   ├── raw/                        # Directory for storing raw data (CSV format: one column for text, one for labels).
│   ├── EpMap_Annotation_Manual.pdf # Detailed annotation manual to guide users in preparing their datasets.
│   └── README.md                   # Instructions for preparing and organizing data files.
├── src/
│   ├── __init__.py
│   ├── models/                     # Model definition, training, and evaluation code.
│   │   ├── __init__.py
│   │   ├── model.py                # Definition of the RoBERTa-based model.
│   │   ├── train.py                # Training script for the model.
│   │   └── evaluate.py             # Evaluation script for the model.
│   ├── data/                       # Data preprocessing and loading scripts.
│   │   ├── __init__.py
│   │   ├── preprocess.py           # Data preprocessing pipeline.
│   │   ├── dataset.py              # Data loading utilities.
│   └── main.py                     # Main script to coordinate the overall workflow.
├── roberta/
│   ├── roberta-base/               # Directory for storing RoBERTa model files.
│   │   └── README.md               # Instructions for organizing RoBERTa-related files.
├── output/
│   ├── README.md                   # Description of the output directory and files.
├── results/
│   └── README.md                   # Instructions for interpreting experimental results.
├── requirements.txt                # List of Python dependencies.
├── .gitignore                      # Configuration for Git to ignore unnecessary files.
└── README.md                       # Project overview and instructions.
```

## Installation

1. Clone the repository

```bash
git clone <repository-url>
cd EpMap_CBCMS_plus
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** Download the required RoBERTa model files and place them under `roberta/roberta-base/` as described in `roberta/README.md`.

## Usage

EpMap follows a straightforward workflow. After preparing the data (see below), run:

```bash
python src/main.py
```

This command will execute the default training/evaluation pipeline as implemented in `src/main.py`, using the configurations defined in the source code.

## Data Preparation

1. Place your dataset under `data/raw/`.
2. Follow the format described in `data/README.md` and the **annotation manual** `data/EpMap_Annotation_Manual.pdf`.

- **text**: a legal clause or sentence.
- **labels**: a serialized list of PDL tags associated with the clause.

> **Please read first:** `data/EpMap_Annotation_Manual.pdf` — it defines the PDL tag set, annotation rules, and examples.

## Output Files

See `output/README.md` and `results/README.md` for details.

## Example Commands

### Quick Start

```bash
pip install -r requirements.txt
python src/main.py
```

### With Custom Data Location (if configured in code)

Adjust paths inside the code or config files (no CLI flags required) and re-run:

```bash
python src/main.py
```

## Performance & Reproducibility

- Preprocessing and tokenization are deterministic; seeds are fixed in the code where applicable.
- GPU acceleration is supported if PyTorch detects CUDA (no extra flags required).

## Troubleshooting

- **Model files**: Ensure RoBERTa weights are correctly placed under `roberta/roberta-base/`.
- **Data format**: Verify that CSV columns match `text` and `labels` and labels are valid PDL tags per the manual.
- **Dependencies**: Re-install via `pip install -r requirements.txt` if import errors occur.
- **Paths**: Confirm that your working directory is the repo root when running `python src/main.py`.

## Notes

- The raw dataset is not included in this repository.
- For accurate replication, **always** consult `data/EpMap_Annotation_Manual.pdf` before preparing data.