# **Training and Deployment Workflow in NeuralLib**

This section describes how to **train, fine-tune, post-process, and deploy models** using NeuralLib.

---

## **1. Training and Retraining Models**

### **Available Architectures**

To see a list of available architectures, use:

```python
from NeuralLib.architectures import get_valid_architectures
print(get_valid_architectures())
```

Make sure to validate architecture names before training:

```python
from NeuralLib.architectures import validate_architecture_name
validate_architecture_name("GRUseq2seq")
```

---

### **Training a Model from Scratch**

To train a model from scratch, instantiate it and call the `train_from_scratch` method:

```python
from NeuralLib.architectures import GRUseq2seq

model = GRUseq2seq(model_name="test", n_features=10, hid_dim=64, n_layers=2, dropout=0.1, learning_rate=0.001)

model.train_from_scratch(
    path_x="data/train_x.npy",
    path_y="data/train_y.npy",
    batch_size=32,
    epochs=50
)

```

Alternatively, you can use a **higher-level function** to automate training:

```python
python
CopiarEditar
from NeuralLib.training import train_architecture_from_scratch

train_architecture_from_scratch(
    architecture_name="GRUseq2seq",
    architecture_params={"n_features": 10, "hid_dim": 64, "n_layers": 2, "dropout": 0.1, "learning_rate": 0.001},
    train_params={"path_x": "data/train_x.npy", "path_y": "data/train_y.npy", "batch_size": 32, "epochs": 50}
)

```

---

### **Retraining an Existing Model**

If you have a checkpoint from a previous training session, you can **continue training**:

```python
python
CopiarEditar
from NeuralLib.training import retrain_architecture

retrain_architecture(
    architecture_name="GRUseq2seq",
    train_params={"batch_size": 32, "epochs": 20},
    checkpoints_directory="checkpoints/GRUseq2seq_run1"
)

```

You can also **load and retrain models from Hugging Face**:

```python
python
CopiarEditar
retrain_architecture(
    architecture_name="GRUseq2seq",
    train_params={"batch_size": 32, "epochs": 20},
    hugging_face_model="marianaagdias/GRU_ecg_model"
)

```

---

### **Hyperparameter Optimization**

Grid search can be used to **find the best hyperparameters** automatically:

```python
python
CopiarEditar
from NeuralLib.training import run_grid_search

best_model = run_grid_search(
    architecture_name="GRUseq2seq",
    architecture_params_options={
        "hid_dim": [32, 64, 128],
        "dropout": [0.1, 0.2, 0.3],
        "learning_rate": [0.001, 0.0005]
    },
    train_params={"batch_size": 32, "epochs": 50}
)

```

💡 This function iterates over all parameter combinations and selects the **best performing model**.

---

## **2. Post-Processing Model Outputs**

Once a model has been trained, you may need to **post-process** its output. This is particularly useful for peak detection or classification thresholds.

### **Binary Peak Detection**

```python
python
CopiarEditar
from NeuralLib.processing import post_process_peaks_binary

predictions = model.predict(input_signal)

filtered_peaks = post_process_peaks_binary(
    predictions,
    threshold=0.5,
    filter_peaks=True
)

```

This function:
✔ Applies a **sigmoid activation**

✔ Thresholds the output to **detect peaks**

✔ (Optional) **Filters out closely spaced peaks**

---

## **3. Deploying Models to Hugging Face**

Once a model is trained, you can **upload it to Hugging Face** to make it publicly available.

### **Generating a Model Card**

To ensure proper documentation, you must create a structured `README.md`:

```python
python
CopiarEditar
from NeuralLib.upload import create_readme

create_readme(
    hparams_file="model/hparams.yaml",
    training_info_file="model/training_info.json",
    output_file="model/README.md",
    collection="NeuralLib Collection",
    description="A deep learning model for ECG signal classification.",
    model_name="GRUseq2seq"
)

```

This automatically formats:

- Hyperparameters
- Training results
- Example usage

---

### **Uploading the Model**

To push the model to **Hugging Face Model Hub**:

```python
python
CopiarEditar
from NeuralLib.upload import upload_production_model

upload_production_model(
    local_dir="models/my_model",
    repo_name="marianaagdias/my_ecg_model",
    token="your-huggingface-token",
    model_name="GRUseq2seq",
    description="This model detects arrhythmias in ECG signals."
)

```

✔ If the repository doesn’t exist, it will be created automatically.

✔ The model is now **publicly available** and can be used by others.

---

## **Final Notes**

- **Training** (`train_architecture_from_scratch`, `retrain_architecture`, `run_grid_search`)
- **Post-Processing** (`post_process_peaks_binary`)
- **Deployment** (`create_readme`, `upload_production_model`)

This section ensures that **developers and researchers** can:
✔ Train and fine-tune models

✔ Apply post-processing

✔ Deploy models to **Hugging Face**