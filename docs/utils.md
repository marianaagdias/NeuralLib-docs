# utils

## **Overview**

The `utils` module provides essential utility functions and a dataset class (`DatasetSequence`) to support training and inference workflows in `NeuralLib`. It includes:

- **Dataset Management**: `DatasetSequence` for handling biosignal datasets.
- **Device Configuration**: Functions to set up GPUs and reproducibility.
- **Model Handling**: Functions for saving models and predictions.

Check [Dataset Requirements](dataset_requirements.md) for using DatasetSequence class to train your models.

---

## **1. Dataset Management**

### **`DatasetSequence` Class**

The `DatasetSequence` class is a PyTorch Dataset wrapper that loads preprocessed biosignal data stored as `.npy` files. It ensures proper formatting, normalization, and data consistency for training and inference.

### **Initialization Parameters**

```python
DatasetSequence(path_x, path_y, part='train', all_samples=False, samples=None, seq2one=False, min_max_norm_sig=False, window_size=None, overlap=None)
```

- **`path_x`** *(str)* → Path to the directory containing input `.npy` files.
- **`path_y`** *(str)* → Path to the directory containing output `.npy` files.
- **`part`** *(str, default='train')* → Dataset partition to use (`'train'`, `'val'`, or `'test'`).
- **`all_samples`** *(bool, default=False)* → If `True`, loads all available samples; otherwise, a subset is used.
- **`samples`** *(int, optional)* → Number of samples to load (must be set if `all_samples=False`).
- **`seq2one`** *(bool, default=False)* → Defines whether the dataset is sequence-to-one (`True`) or sequence-to-sequence (`False`).
- **`min_max_norm_sig`** *(bool, default=False)* → Whether to apply **Min-Max Normalization** to each signal.
- **`window_size`** *(int, optional)* → Not yet implemented (for potential sliding window functionality).
- **`overlap`** *(float, optional)* → Not yet implemented (for potential window overlap functionality).

### **Key Features**

✔️ **Loads `.npy` files** efficiently.

✔️ **Auto-adjusts input shape** (`seq_len, num_features`).

✔️ **Ensures output consistency** for `seq2seq` and `seq2one` tasks.

✔️ **Optional min-max normalization** of input/output.

---

## **2. Device and Reproducibility Functions**

### **`configure_seed(seed)`**

```python
configure_seed(42)
```

Sets the random seed for NumPy, PyTorch, and CUDA to ensure reproducibility.

### **`configure_device(gpu_id=None)`**

```python
device = configure_device()
```

Automatically configures GPU (if available) or CPU, printing the selected device.

### **`list_gpus()`**

Lists all available GPUs.

---

## **3. Model and Prediction Utilities**

### **`save_model_results(model, results_dir, model_name, best_val_loss)`**

```python
save_model_results(model, "results", "GRU_model", 0.25)
```

Saves the model hyperparameters and best validation loss in a `results.json` file.

### **`save_predictions(predictions, batch_idx, dir)`**

```python
save_predictions(predictions, 3, "predictions_dir")
```

Saves predictions as `.npy` files.

### **`save_predictions_with_filename(predictions, input_filename, dir)`**

```python
save_predictions_with_filename(predictions, "sample_101.npy", "output_dir")
```

Saves predictions using the input filename for easier tracking.

---

## **4. Data Formatting and Collation**

### **`collate_fn(batch)`**

```python
train_dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
```

Custom collation function for variable-length sequences. It:
✔ **Sorts sequences** by length for efficient batch processing.

✔ **Pads sequences** to the longest sequence in the batch.

---

## 5. **LossPlotCallback**

`LossPlotCallback` is a custom PyTorch Lightning callback designed to track and visualize **training and validation loss** over epochs. At the end of training, it generates a loss plot and saves it to a specified path.

```python
from NeuralLib.utils import LossPlotCallback
trainer = pl.Trainer(callbacks=[LossPlotCallback("results/loss_plot.png")])
```

### **Key Features**

- **Automatically tracks `train_loss` and `val_loss`** at the end of each epoch.
- **Saves a loss plot (`.png`)** to the specified path at the end of training.
- **Compatible with PyTorch Lightning's callback system**.

### **Methods**

- `on_train_epoch_end(trainer, pl_module)`: Logs training loss at the end of each epoch.
- `on_validation_epoch_end(trainer, pl_module)`: Logs validation loss at the end of each epoch.
- `on_train_end(trainer, pl_module)`: Generates and saves the loss plot.

---

[Dataset Organization and Requirements](dataset_requirements.md)