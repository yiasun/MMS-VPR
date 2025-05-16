# README

## Dependencies

This project requires Python 3.7 or higher and depends on the following third-party libraries:

* **Pillow**              — image loading and preprocessing (`from PIL import Image`)
* **numpy**               — numerical computations (`import numpy as np`)
* **pandas**              — data manipulation (`import pandas as pd`)
* **scikit-learn**        — machine-learning utilities, algorithms and metrics

  * Data splitting & preprocessing: `train_test_split`, `LabelEncoder`
  * Dimensionality reduction: `PCA`
  * Nearest neighbors: `NearestNeighbors`, `KNeighborsClassifier`
  * Probabilistic classifier: `GaussianNB`
  * Neural classifier: `MLPClassifier`
  * Ensemble classifier: `RandomForestClassifier`
  * Support Vector Machine: `SVC`
  * Evaluation metrics: `accuracy_score`, `classification_report`, `confusion_matrix`, `precision_score`, `recall_score`, `f1_score`
* **torch**               — core deep learning framework

  * Neural network modules: `torch.nn`, `torch.nn.functional`
  * Optimizers: `torch.optim`
  * Data handling: `TensorDataset`, `Dataset`, `DataLoader`
* **torch-geometric**     — graph data structures and heterogeneous GNN layers

  * Data containers: `Data`, `HeteroData`
  * Convolutional layers: `GCNConv`, `GATConv`, `HGTConv`
* **transformers**        — Hugging Face’s BERT tokenizer & model (`BertTokenizer`, `BertModel`)
* **matplotlib**          — plotting (`import matplotlib.pyplot as plt`)
* **seaborn**             — statistical data visualization (`import seaborn as sns`)

### Recommended minimum versions

```
Pillow>=8.0.0
numpy>=1.19.0
pandas>=1.1.0
scikit-learn>=0.24.0
torch>=1.8.0
torch-geometric>=2.0.0
transformers>=4.6.0
matplotlib>=3.3.0
seaborn>=0.11.0
```

### Installation

Install all dependencies in one go with pip:

```bash
pip install \
  pillow numpy pandas scikit-learn \
  torch torch-geometric transformers \
  matplotlib seaborn
```

If you prefer to pin exact versions, create a `requirements.txt` containing the lines above and run:

```bash
pip install -r requirements.txt
```
