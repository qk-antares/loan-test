# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a loan distribution ML system that intelligently routes loan applications to multiple partner institutions to maximize approval rates and revenue. The project uses Python with scikit-learn for machine learning experiments with different feature engineering approaches.

## Running the Models

To run the different ML experiments:

```bash
# Run the base random forest model
python ml_model_design.py

# Run with one-hot encoding for categorical features
python ml_model_design_onehot.py

# Run with MLP-based dimensionality reduction
python ml_model_design_mlp.py
```

## Data Processing

Before running models, process the raw data:

```bash
python data_processor.py
```

Place the raw CSV file (format: 20250903.csv) in the `data/` directory before processing.

## Core Architecture

### Data Flow
1. **Raw Data** → `data/` directory (CSV files from partners)
2. **Processing** → `data_processor.py` reads, cleans, and transforms data
3. **Feature Engineering** → Different approaches in ml_model_design variants
4. **Model Training** → RandomForest-based multi-label classification
5. **Results** → Stored in `log/` directory

### Key Components

- **DataProcessor** (data_processor.py): Handles JSON parsing, data cleaning, field validation, and feature extraction from loan application data
- **LoanDistributionModel** (ml_model_design.py): Multi-label classification model that predicts approval probability for each partner
- **Feature Engineering Variants**:
  - Base model: Label encoding for categorical features
  - One-hot model: One-hot encoding for better categorical representation
  - MLP model: Neural network-based dimensionality reduction

### Model Strategy

The system uses a multi-label classification approach where:
- Each partner is treated as a separate label
- RandomForestClassifier predicts approval probability per partner
- Ranking and constraint optimization select optimal partner combinations
- Class imbalance handled via SMOTE/ADASYN oversampling

### Data Schema

Key fields processed from loan applications:
- Personal info: age, gender, education, marital status
- Financial: income level, loan amount, purpose
- Employment: company, industry, occupation
- Identity: ID validation dates, address parsing
- Relationships: emergency contacts

## Important Notes

- No dependency management files (requirements.txt) exist - ensure scikit-learn, pandas, numpy, imbalanced-learn are installed
- Data files in `data/`, `partner_csv/`, and `processed_data/` directories are gitignored
- Experimental results logged in `log/` directory with Chinese filenames