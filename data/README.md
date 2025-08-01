# Data Directory

This directory is used to store and manage the data required for training and evaluating the EpMap model.

## Structure
```
data/
├── raw/                       # Directory for raw data files (not provided)
├── EpMap_Annotation_Manual.pdf # Detailed annotation manual to guide users in preparing their datasets
└── README.md                  # Data directory description (this file)
```
## Raw Data

The `raw/` directory should contain the raw dataset files. The dataset is expected to be in CSV format, where:
- Each line corresponds to a data instance.
- The first column is the `text`, and the second column is the `label`.

### Example CSV File Format

"text","label"
"This is a sample text.","1,0...,1"
"Another example text.","0,1...,0"

## Notes

- The raw data is not provided in this repository.
- We provide a detailed annotation manual, including comprehensive instructions, definitions, examples, and common pitfalls for labeling data according to PDL in EpMap.
- Users should place their dataset files in the `raw/` directory before running any preprocessing or training scripts.
- Ensure the dataset adheres to the specified format for compatibility.