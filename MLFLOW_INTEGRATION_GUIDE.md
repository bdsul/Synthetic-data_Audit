# MLflow Integration Guide

This document describes the MLflow integration added to all synthetic data evaluation notebooks.

## MLflow Setup

All notebooks now include:
1. MLflow import and setup at the beginning
2. **Two tracking options:**
   - **Option 1 (Default)**: Local file-based backend (`file:./mlruns`) - No server needed, stores runs locally
   - **Option 2**: MLflow server at `http://localhost:6001` - Requires server setup and authentication
3. Experiment names for each model:
   - `Synthetic_Data_Evaluation_CTGAN`
   - `Synthetic_Data_Evaluation_CopulaGAN`
   - `Synthetic_Data_Evaluation_GaussianCopula`
   - `Synthetic_Data_Evaluation_TVAE`
   - `Synthetic_Data_Evaluation_VanillaGAN`

## Metrics Logged

### Fidelity Metrics
- Diagnostic scores (data validity, data structure)
- Quality scores (column shapes, column pair trends, overall)
- JS Divergence (mean, median, max, min, per-feature)
- MMD score
- Wasserstein distance (mean, median, max, min, per-feature)
- Gower distance metrics (intra-set and cross-set similarities)
- Cosine similarity (average, maximum)

### Utility Metrics
- TSTR (Train on Synthetic, Test on Real): Accuracy, F1 Score, AUC
- TRTR (Train on Real, Test on Real): Accuracy, F1 Score, AUC

### Privacy Metrics
- Average nearest neighbor distance

### Bivariate Analysis Metrics
- Maximum and mean delta correlation
- Maximum and mean delta Wasserstein

### Multivariate Analysis Metrics
- Global MMD (RBF)
- Two-sample classifier accuracy

## Models Saved

All SDV models are saved and logged as artifacts:
- CTGAN model → `ctgan_model/`
- CopulaGAN model → `copulagan_model/`
- Gaussian Copula model → `gaussiancopula_model/`
- TVAE model → `tvae_model/`
- Vanilla GAN → PyTorch models (Generator and Discriminator)

## Synthetic Data Artifacts

Synthetic datasets are saved as CSV files and logged as artifacts.

## Usage

### Option 1: Local File-Based Backend (Default - Recommended)

1. **No server setup needed!** The notebooks use local file storage by default
2. Run the notebooks - metrics will be stored in `./mlruns/` directory
3. View results by starting MLflow UI:
   ```bash
   mlflow ui
   ```
   Then open: `http://localhost:5000` (default port)

### Option 2: MLflow Server

If you want to use a remote MLflow server:

1. **If server requires authentication**, you have two options:
   - Set environment variables:
     ```bash
     export MLFLOW_TRACKING_USERNAME=your_username
     export MLFLOW_TRACKING_PASSWORD=your_password
     ```
   - Or uncomment and modify the tracking URI in notebooks:
     ```python
     mlflow.set_tracking_uri("http://username:password@localhost:6001")
     ```

2. Start MLflow server: `mlflow ui --port 6001`
3. Run the notebooks
4. View results at: `http://localhost:6001`

## Notes

- **Default setup uses local file storage** - no authentication needed
- Each notebook starts an MLflow run at the beginning of model training
- The run ends after all metrics are logged
- All metrics are logged incrementally as they are computed
- To switch to server mode, uncomment Option 2 in the notebook and comment out Option 1

