# EpMap for CBCMS+ (Enhanced PDL Mapping Pipeline)

This repository contains the implementation of the Enhanced PDL Mapping Pipeline (EpMap) for the CBCMS+ system. EpMap leverages deep learning models to extract structured policies from legal texts efficiently and accurately, supporting multiple languages and legal frameworks.

## Directory Structure
```
EpMap_CBCMS_plus/
├── data/
│   ├── raw/                       # Directory for storing raw data (CSV format: one column for text, one for labels).
│   ├── EpMap_Annotation_Manual.pdf # Detailed annotation manual to guide users in preparing their datasets.
│   └── README.md                  # Instructions for preparing and organizing data files.
├── src/
│   ├── __init__.py                # Marks the directory as a Python package.
│   ├── models/                    # Model definition, training, and evaluation code.
│   │   ├── __init__.py
│   │   ├── model.py               # Definition of the RoBERTa-based model.
│   │   ├── train.py               # Training script for the model.
│   │   └── evaluate.py            # Evaluation script for the model.
│   ├── data/                      # Data preprocessing and loading scripts.
│   │   ├── __init__.py
│   │   ├── preprocess.py          # Data preprocessing pipeline.
│   │   ├── dataset.py             # Data loading utilities.
│   └── main.py                    # Main script to coordinate the overall workflow.
├── roberta/
│   ├── roberta-base/              # Directory for storing RoBERTa model files.
│   │   └── README.md              # Instructions for organizing RoBERTa-related files.
├── output/
│   ├── README.md                  # Description of the output directory and files.
├── results/
│   └── README.md                  # Instructions for interpreting experimental results.
├── requirements.txt               # List of Python dependencies.
├── .gitignore                     # Configuration for Git to ignore unnecessary files.
└── README.md                      # Project overview and instructions.
```

## How to Use

### 1. Set Up the Environment

Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Prepare the Data

- Place your dataset in the `data/raw/` directory.
- Ensure the dataset follows the format described in `data/README.md`.

### 3. Train and Evaluate the Model

Run the training script:

``` bash
python src/main.py
```

## Notes

- The raw dataset is not included in this repository.
- However, we provide a detailed annotation manual to guide users in preparing their datasets. The manual includes comprehensive instructions, definitions, examples, and common pitfalls for labeling data according to the Policy Definition Language (PDL) used in the Enhanced PDL Mapping Pipeline (EpMap).
- For detailed usage instructions and additional configuration, see the other documents in the directory.