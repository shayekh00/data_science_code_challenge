# 🌨️ Snowfall Prediction Challenge

## 🚀 Problem Overview

The goal of this project is to analyze and model weather station data to predict snowfall events. We were provided with GSOD (Global Summary of the Day) weather data between 2000 and 2005 across a selected set of U.S. stations. The task involves preparing this raw data for classification modeling with appropriate data cleaning, feature engineering, and model training workflows.

---

## 🧠 Solution Structure

This repository is organized for clarity, reproducibility, and scalability.
The data notebooks contain all the decision regarding all the data processing decisions.
Please kindly run the notebooks from start to end in order to run the solution. 
It includes:

### 🧪 Notebooks
- `Data_Exploration.ipynb`  
  Used for in-depth data analysis, visualization, missingness inspection, correlation testing, and hypothesis-driven preprocessing decisions. This notebook **informs** what we do downstream in processing and modeling.
  
- `Coding_Challenge.ipynb`  
  The main pipeline where we **implement** the solution from start to finish: data ingestion, preprocessing (using modular transformers), model training, and evaluation.

---

### 🧱 Modular Pipeline (in `pipeline/`)
All preprocessing steps are abstracted as transformers in the `pipeline/transformers/` directory to ensure code cleanliness, reusability, and maintainability. Examples:
- `log_transform.py`: Log transforms skewed features
- `binary_flag_imputer.py`: Handles missing values in binary indicators
- `knn_imputer.py`: Applies KNN-based imputation
- `cyclical_encoder.py`: Encodes seasonal patterns (e.g., month)
- `outlier.py`: Removes IQR-based outliers per station
- `feature_dropper.py`: Drops low-variance, highly correlated, or irrelevant features

The main runner (`pipeline_runner.py`) composes and executes these transformations in a sequential pipeline.

---

## 🧪 How to Run

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
