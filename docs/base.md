
## Class: `Architecture`
`Architecture` class provides a framework for training, retraining, testing, and managing neural network models tailored for biosignal processing. It integrates with PyTorch Lightning to streamline training workflows, support checkpoint management, and enable logging via TensorBoard. It also includes utilities for loading pre-trained models from local directories or Hugging Face.

---

### **Initialization**

### `__init__(self, architecture_name)`

- Defines a base neural network model with `architecture_name`.
- Initializes training history and checkpoint directory.

---

### **Checkpoint and Training Information Management**

### `create_checkpoints_directory(self, retraining)`

- Creates a unique directory for storing model checkpoints.
- Uses architecture parameters and timestamp for directory naming.
- The directory follows the format: `{RESULTS_BASE_DIR}/{model_name}/checkpoints/{architecture_name}_{hid_dim}hid_{n_layers}l_lr{learning_rate}_drop{dropout}_{timestamp}`.

If retraining, the directory name includes `_retraining` to differentiate from the initial training run.

### `save_training_information(self, trainer, optimizer, train_dataset_name, trained_for, val_loss, total_training_time, gpu_model, retraining, prev_training_history=None)`

- Stores information about the current training process, including dataset, optimizer, validation loss, and training time.
- If retraining, appends previous training history.

### `save_training_information_to_file(self, directory)`

- Saves the training information dictionary to `training_info.json` in the given directory.

---

### **Model Training and Retraining**

### `train_from_scratch(self, path_x, path_y, patience, batch_size, epochs, gpu_id=None, all_samples=False, samples=None, dataset_name=None, trained_for=None, classification=False, enable_tensorboard=False)`

- Trains a model from scratch using PyTorch Lightning.
- Initializes model, dataset, dataloaders, and training parameters.
- Saves best checkpoint and logs training progress using TensorBoard.
- Saves final weights in `.pth` format.

Detailed breakdown of the function:

1. **Setup & Configuration:**

The random seed is set, the device (CPU/GPU) is configured using `configure_device(gpu_id)`. Checks if TensorBoard is available.

1. **Model Initialization & Checkpoints:**

A directory for storing model checkpoints is created in 
`<DEV_BASE_DIR>/results/<model_name>/checkpoints/<architecture_name_hparams_datetime>`

1. **Dataset & DataLoader Preparation:**

Training and validation datasets are instantiated from `DatasetSequence`, loading data from `path_x` (features) and `path_y` (labels). PyTorch `DataLoader` objects are created. The `collate_fn` function is used to handle dynamic sequence lengths.

1. **Defining Callbacks for Training:**
    - **Checkpoint Callback:** Saves the best model based on validation loss (`val_loss`).
    - **Early Stopping Callback:** Stops training early if validation loss doesn't improve for `patience` epochs.
    - **Loss Plot Callback:** Saves a loss curve to visualize training progress.
2. **Trainer Initialization & Logging:**
    - If TensorBoard is enabled, a `TensorBoardLogger` is set up for tracking metrics and hyperparameters (hparams.yaml)
    - The PyTorch Lightning `Trainer` is instantiated, specifying: maximum epochs (`epochs`), device, callbacks (Checkpointing, Early Stopping, Loss Plot), logging
3. **Training Execution:**

The model is trained using `trainer.fit(model, train_dataloader, val_dataloader)`.

1. **Post-Training Processing & Model Saving:**
    - The **best (lowest) validation loss** is extracted from the checkpoint callback.
    - Training metadata (trainer state, optimizer, dataset, GPU info, loss, etc.) is saved (using `model.save_training_information()`) and written to `training_info.json` inside the checkpoint directory.
    - The **final model weights** (corresponding to the lowest validation loss) are saved in `model_weights.pth` inside the checkpoint directory.

### `retrain(self, path_x, path_y, patience, batch_size, epochs, gpu_id=None, all_samples=False, samples=None, dataset_name=None, trained_for=None, classification=False, enable_tensorboard=False, checkpoints_directory=None, hugging_face_model=None)`

- Loads model weights from a previous checkpoint or Hugging Face repository.
- Performs training while preserving previous history. The main difference to `train_from_scratch` is
    - retraining of the whole model (can be either to use additional/different data, to train for more epochs,..). If the purpose is to perform TL, use the `TLModel` class from `model_hub` package.
- Saves updated training information and final model weights.

---

### **Model Testing and Evaluation**

### `test_on_test_set(self, path_x, path_y, all_samples=True, samples=None, checkpoints_dir=None, gpu_id=None, save_predictions=False, post_process_fn=None)`

- Evaluates the trained model on a test dataset.
- Loads model weights from the checkpoint directory.
- Returns predictions and average test loss.
- Optionally saves predictions and applies post-processing functions.

### `test_on_single_signal(self, X, checkpoints_dir=None, gpu_id=None, post_process_fn=None)`

- Tests the model on a single input signal.
- Loads model weights from the checkpoint directory.
- Returns the predicted output, optionally applying post-processing.

---

### **(Auxiliary) Functions and Methods**

### 1. `get_weights_and_info_from_checkpoints(prev_checkpoints_dir)`

- Extracts model weights (`.pth`) and training history (`training_info.json`) from a given checkpoint directory.
- If `.pth` is unavailable, converts `.ckpt` to `.pth`.
- Returns the path to the weights file and training history.

### 2. `get_weights_and_info_from_hugging(hugging_face_model, local_dir=None)`

- Downloads model weights and training history from a Hugging Face repository.
- Returns the local path to the weights and training history file.

### 3. `get_hparams_from_checkpoints(checkpoints_directory)`

- Retrieves hyperparameters from the latest version of `hparams.yaml` in a checkpoint directory.
- Ensures the existence of relevant log directories.

**Example (from tutorial 2 - grid search):**

```python
from NeuralLib.architectures import get_hparams_from_checkpoints, GRUseq2seq
architecture_params = get_hparams_from_checkpoints(best_dir)
# Initialize the model using the loaded parameters
model = GRUseq2seq(**architecture_params)
```

### 4. `get_hparams_from_hugging(hugging_face_model)`

- Downloads hyperparameters (`hparams.yaml`) from a Hugging Face repository.
- Returns hyperparameters as a dictionary.

### 5. `validate_training_context(retraining, checkpoints_directory, hugging_face_model)`

- Ensures correct conditions for training (`retraing=False`) and retraining (`retraing=True`).
- Raises errors if checkpoints are missing when retraining or provided when training from scratch.

---