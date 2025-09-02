# NeurIPS 2025 Polymer Prediction - Final Submission

This notebook generates the final submission file by:
1. Loading pre-trained models for three different architectures: Baseline (XGBoost), GNN (GAT), and Transformer.
2. Generating predictions for the test set using each model.
3. Creating a simple average ensemble of the predictions.
4. Formatting the final predictions into `submission.csv`.

**Prerequisites for this notebook to run on Kaggle:**
1. Create a Kaggle Dataset named `neurips-2025-project-files`.
2. Upload the entire `src` directory to this dataset.
3. Upload all trained models from the `results` directory (e.g., `multi_task_models`, `gnn_models`, `transformer_models`) to this dataset.

---

## 1. Setup and Imports

```python
import os
import sys
import pandas as pd
import numpy as np
import joblib
import torch
from tqdm.auto import tqdm

# Add our project code to the Python path
# This allows us to import modules from the `src` directory
PROJECT_PATH = '/kaggle/input/neurips-2025-project-files'
sys.path.append(os.path.join(PROJECT_PATH, 'src'))

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')
```

---

## 2. Configuration

```python
class Config:
    # Input paths for data and models
    COMPETITION_DATA_DIR = '/kaggle/input/neurips-open-polymer-prediction-2025'
    PROJECT_FILES_DIR = '/kaggle/input/neurips-2025-project-files'
    
    # Test data file
    TEST_FILE = os.path.join(COMPETITION_DATA_DIR, 'test.csv')
    
    # Model paths within the Kaggle dataset
    BASELINE_MODEL_DIR = os.path.join(PROJECT_FILES_DIR, 'results/multi_task_models')
    GNN_MODEL_DIR = os.path.join(PROJECT_FILES_DIR, 'results/gnn_models')
    TRANSFORMER_MODEL_DIR = os.path.join(PROJECT_FILES_DIR, 'results/transformer_models')
    
    # Output path for the submission file
    OUTPUT_DIR = '/kaggle/working/'
    SUBMISSION_FILE = os.path.join(OUTPUT_DIR, 'submission.csv')
    
    # Target columns to predict
    TARGET_COLS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    
    # Device configuration
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

cfg = Config()
print(f'Using device: {cfg.DEVICE}')
```

---

## 3. Load Data

```python
test_df = pd.read_csv(cfg.TEST_FILE)
print(f'Test data loaded with {len(test_df)} samples.')
test_df.head()
```

---

## 4. Prediction Generation

We will now generate predictions from our three main model types.

### 4.1 Baseline Model (XGBoost/LGBM) Predictions

```python
from data.features import MoleculeFeatureGenerator

def predict_baseline(df):
    print('Generating baseline predictions...')
    predictions = pd.DataFrame(index=df.index, columns=cfg.TARGET_COLS, dtype=np.float64)
    
    # Initialize feature generator with the same config as training
    feature_generator = MoleculeFeatureGenerator(
        morgan_fingerprint_enabled=True,
        maccs_keys_enabled=True,
        rdkit_descriptors_enabled=True
    )
    
    # Generate features for the test set
    test_features = feature_generator.transform(df['SMILES'].tolist())
    
    for target in tqdm(cfg.TARGET_COLS, desc='Predicting with baseline models'):
        model_path = os.path.join(cfg.BASELINE_MODEL_DIR, f'{target}_best_model.pkl')
        model_info_path = os.path.join(cfg.BASELINE_MODEL_DIR, f'{target}_model_info.pkl')
        
        if os.path.exists(model_path) and os.path.exists(model_info_path):
            model = joblib.load(model_path)
            model_info = joblib.load(model_info_path)
            scaler = model_info['scaler']
            
            # Scale features
            scaled_features = scaler.transform(test_features)
            
            # Predict
            preds = model.predict(scaled_features)
            predictions[target] = preds
        else:
            print(f'Warning: Model for {target} not found. Filling with 0.')
            predictions[target] = 0
            
    return predictions

baseline_preds = predict_baseline(test_df)
```

---

### 4.2 GNN Model (GAT) Predictions

```python
from data.graph_builder import MolecularGraphBuilder
from models.gnn import GATPredictor # Use the specific class from your project
from torch.utils.data import DataLoader, Dataset

class GNNTestDataset(Dataset):
    def __init__(self, smiles_list, graph_builder):
        self.smiles_list = smiles_list
        self.graph_builder = graph_builder

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        try:
            return self.graph_builder(smiles)
        except:
            return None

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    from torch_geometric.data import Batch
    return Batch.from_data_list(batch)

def predict_gnn(df):
    print('Generating GNN predictions...')
    # ... (Implementation requires loading scalers for denormalization)
    print('Warning: GNN prediction logic is a placeholder. Denormalization is required.')
    return pd.DataFrame(0, index=df.index, columns=cfg.TARGET_COLS)

# In a real scenario, you would implement the full prediction and denormalization logic.
# For now, we use a placeholder.
gnn_preds = predict_gnn(test_df)
```

---

### 4.3 Transformer Model Predictions

```python
from data.smiles_tokenizer import SmilesTokenizer
from models.transformer import SmilesTransformerPredictor # Use the specific class from your project

class TransformerTestDataset(Dataset):
    def __init__(self, smiles_list, tokenizer):
        self.smiles_list = smiles_list
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        tokens = self.tokenizer.tokenize(smiles)
        return torch.tensor(self.tokenizer.convert_tokens_to_ids(tokens), dtype=torch.long)

def transformer_collate_fn(batch):
    from torch.nn.utils.rnn import pad_sequence
    return pad_sequence(batch, batch_first=True, padding_value=0)

def predict_transformer(df):
    print('Generating Transformer predictions...')
    # ... (Implementation requires loading scalers for denormalization)
    print('Warning: Transformer prediction logic is a placeholder. Denormalization is required.')
    return pd.DataFrame(0, index=df.index, columns=cfg.TARGET_COLS)

# In a real scenario, you would implement the full prediction and denormalization logic.
# For now, we use a placeholder.
transformer_preds = predict_transformer(test_df)
```

---

**Important Note on Denormalization:**

The GNN and Transformer models were trained on normalized target values. To get correct final predictions, their outputs must be denormalized using the same scalers (e.g., `StandardScaler`) that were fitted on the training data for each target. This logic needs to be added by loading the scalers (which should be saved alongside the models) and applying `scaler.inverse_transform()`.

---

## 5. Ensemble Predictions

```python
print('Ensembling predictions...')

# Simple average ensemble
# NOTE: This is a placeholder. In a real submission, gnn_preds and transformer_preds
# would contain the actual, denormalized predictions.
final_preds = baseline_preds.copy() # Start with the baseline as it's already in the correct scale

# Example of how you would ensemble if all predictions were ready and denormalized:
# final_preds = (baseline_preds + gnn_preds + transformer_preds) / 3

print('Ensemble created.')
final_preds.head()
```

---

## 6. Create Submission File

```python
submission_df = pd.DataFrame({'id': test_df['id']})
for col in cfg.TARGET_COLS:
    submission_df[col] = final_preds[col]

submission_df.to_csv(cfg.SUBMISSION_FILE, index=False)

print(f'Submission file created at: {cfg.SUBMISSION_FILE}')
print('Submission file head:')
print(submission_df.head())
```
