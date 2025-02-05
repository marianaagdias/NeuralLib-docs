## Development Configuration (config.py)

### Assumed Folder Structure
When using config.py, the project expects a specific directory structure relative to the root directory (dev):
```
dev/
│── NeuralLib/              # Library source code (contains `config.py`)
│── data/                   # Directory where datasets are stored
│── results/                # Stores model training logs, evaluation results, and checkpoints
│── hugging_prodmodels/     # Stores Hugging Face models locally before uploading
```

### **Defined Paths**

- `DATA_BASE_DIR`: Points to `dev/data/`, where datasets are stored.
- `RESULTS_BASE_DIR`: Points to `dev/results/`, where model training outputs are saved.
- `HUGGING_MODELS_BASE_DIR`: Points to `dev/hugging_prodmodels/`, where production models are stored before uploading.

If these directories do not exist, they are automatically created at runtime.