# Integrated Deep Learning and Multi-Objective Pareto Optimization for Retail Supply Chains

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Paper:** *An Integrated Deep Learning and Multi-Objective Pareto Optimization Framework for Retail Supply Chains*  
> **Authors:** Mohammed Zizi, Anas Chafi, Mohammed El Hammoumi  
> **Affiliation:** Laboratory of Industrial Techniques, Faculty of Sciences and Techniques, Sidi Mohamed Ben Abdellah University, Fez, Morocco  

---

## Overview

This repository provides the complete implementation for reproducing the results reported in our paper. The framework integrates **ensemble deep learning forecasting** with **multi-objective Pareto optimization** to simultaneously optimize four supply chain objectives: inventory cost, service level, delivery efficiency, and production smoothness.

### Key Results

| Metric | Value |
|--------|-------|
| Forecast Accuracy (R²) | **0.9299** (23.8% improvement over naive) |
| Best Total Cost | **$2,936,505** (61.5% reduction) |
| Best Service Level | **97.29%** |
| Delivery Efficiency | **70.7%** improvement |
| Production Smoothness | **70.2%** improvement |
| Projected Annual Savings | **$5.26M** (ROI: 1,052%) |

## Framework Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA PIPELINE                             │
│  Data Loading → Feature Engineering → Feature Selection      │
│  (913K obs.)    (31 features)         (Top 30 via F-stat)   │
├─────────────────────────────────────────────────────────────┤
│                  FORECASTING MODELS                          │
│  ┌──────────────┐ ┌──────────┐ ┌─────────────┐             │
│  │   Quantile   │ │  BiLSTM  │ │ Transformer │             │
│  │  Regression  │ │ (64+32)  │ │  (128-dim)  │             │
│  │ (q10,q50,q90)│ │          │ │             │             │
│  └──────┬───────┘ └────┬─────┘ └──────┬──────┘             │
│         └──────────────┼──────────────┘                      │
│              CV-Optimized Ensemble                            │
│           (w = 0.42, 0.31, 0.27)                            │
├─────────────────────────────────────────────────────────────┤
│                   OPTIMIZATION                               │
│  Single-Objective (Newsvendor) → Multi-Objective (Pareto)   │
│  4 objectives: Cost, Service, Delivery Eff., Prod. Smooth.  │
│  → 4 Pareto-optimal solutions                               │
└─────────────────────────────────────────────────────────────┘
```

## Repository Structure

```
dl-moo-supply-chain/
├── README.md                        # This file
├── LICENSE                          # MIT License
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Git ignore rules
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py             # Data loading, feature engineering, selection
│   ├── models.py                    # DL model architectures (QR, BiLSTM, Transformer)
│   ├── training.py                  # Model training and ensemble optimization
│   ├── evaluation.py                # Forecast evaluation metrics (R², MAE, CRPS, etc.)
│   ├── inventory_optimization.py    # Single-objective newsvendor inventory model
│   ├── pareto_optimization.py       # Multi-objective Pareto optimization engine
│   ├── financial_analysis.py        # ROI and financial projections
│   └── visualization.py            # Publication-quality figure generation
├── notebooks/
│   └── full_pipeline.ipynb          # Complete Jupyter notebook (original)
├── main.py                          # Main execution script
└── figures/                         # Generated figures (after running)
```

## Installation

### Prerequisites
- Python 3.8+
- NVIDIA GPU (recommended) or CPU

### Setup

```bash
# Clone the repository
git clone https://github.com/mzizi-supply-chain/dl-moo-framework.git
cd dl-moo-framework

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dataset
Download the [Store Item Demand Forecasting](https://www.kaggle.com/competitions/demand-forecasting-kernels-only) dataset from Kaggle and place `train.csv` in the project root:

```bash
# Using Kaggle CLI
kaggle competitions download -c demand-forecasting-kernels-only
unzip demand-forecasting-kernels-only.zip
```

## Usage

### Quick Start (Full Pipeline)

```bash
python main.py
```

This runs the complete pipeline: data loading → feature engineering → model training → evaluation → inventory optimization → multi-objective Pareto optimization → financial analysis → report generation.

### Module-by-Module Usage

```python
from src.data_pipeline import load_and_split_data, engineer_features, select_features
from src.models import build_quantile_model, build_lstm, build_transformer
from src.training import train_advanced_models
from src.evaluation import evaluate_with_stats
from src.inventory_optimization import optimize_inventory
from src.pareto_optimization import multi_objective_optimization

# Step 1: Data
train, val, test, df = load_and_split_data('train.csv')
train, val, test = engineer_features(train, val, test)
selected_features, all_features = select_features(train, val, test, k=40)

# Step 2: Train models
all_preds, y_test = train_advanced_models(train, val, test, selected_features, baselines)

# Step 3: Evaluate
results = evaluate_with_stats(y_test, all_preds)

# Step 4: Optimize
inv_results = {name: optimize_inventory(y_test, pred) for name, pred in all_preds.items()}
pareto_solutions, baseline_scores, del_std, prod_std = multi_objective_optimization(
    y_test, all_preds['Weighted Ensemble'], test, baselines
)
```

### Jupyter Notebook

For an interactive walkthrough with inline visualizations:

```bash
jupyter notebook notebooks/full_pipeline.ipynb
```

## Methodology

### Forecasting Models
- **Quantile Regression NN**: 3-layer feedforward (256→128→64) with multi-head outputs for q10, q50, q90
- **Bidirectional LSTM**: 2-layer BiLSTM (64, 32 units) with dropout and L2 regularization  
- **Transformer**: Self-attention with 128-dim embedding, feed-forward sublayer, residual connections
- **Weighted Ensemble**: Cross-validation-optimized weights minimizing validation MAE

### Multi-Objective Optimization
Four formally defined objectives (see paper Equations 8–13):
1. **Total Cost** (minimize): holding + stockout costs
2. **Service Level** (maximize): fraction of demand fulfilled
3. **Delivery Efficiency** (maximize): reduction in store-level order variability
4. **Production Smoothness** (maximize): reduction in daily production variability

### Evaluation Metrics
- Point metrics: R², MAE, RMSE, MAPE, 95% CI
- Probabilistic: CRPS, Pinball Loss
- Inventory: Total cost, Service level, Safety stock
- Computational: Training time, inference latency, memory

## Results Summary

### Forecasting Performance

| Model | R² | MAE | RMSE | MAPE (%) |
|-------|-----|-----|------|----------|
| Naive (t-1) | 0.7509 | 11.87 | 16.27 | 22.06 |
| Moving Avg (7d) | 0.8492 | 9.55 | 12.65 | 17.41 |
| Quantile (p50) | 0.9290 | 6.59 | 8.68 | 12.18 |
| BiLSTM | 0.9270 | 6.68 | 8.81 | 12.27 |
| Transformer | 0.9289 | 6.61 | 8.69 | 12.07 |
| **Weighted Ensemble** | **0.9299** | **6.56** | **8.63** | **12.06** |

### Pareto-Optimal Solutions

| Solution | Service (%) | Cost ($) | Delivery Eff. (%) | Prod. Smooth. (%) |
|----------|-------------|----------|--------------------|--------------------|
| A (Balanced) | 65.63 | 5,010,951 | 70.71 | 70.20 |
| B (Cost-Service) | 67.46 | 4,902,570 | 65.83 | 65.23 |
| C (High-Service) | 72.33 | 4,231,181 | 60.94 | 60.26 |
| J (Low-Cost) | 67.46 | 4,669,122 | 60.94 | 10.59 |

## Citation

If you use this code in your research, please cite:

```bibtex
@article{zizi2026integrated,
  title={An Integrated Deep Learning and Multi-Objective Pareto Optimization 
         Framework for Retail Supply Chains},
  author={Zizi, Mohammed and Chafi, Anas and El Hammoumi, Mohammed},
}
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Kaggle for the [Store Item Demand Forecasting Challenge](https://www.kaggle.com/competitions/demand-forecasting-kernels-only) dataset
- Laboratory of Industrial Techniques, USMBA, Fez, Morocco

## Contact

- **Mohammed Zizi** — mohammed.zizi3@usmba.ac.ma
- Department of Industrial Engineering, Faculty of Sciences and Techniques, Sidi Mohamed Ben Abdellah University, Fez, Morocco
"# dl-moo-framework" 
"# dl-moo-framework" 
"# dl-moo-framework" 
"# dl-moo-framework" 
