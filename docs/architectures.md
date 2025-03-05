# architectures


## Overview

The `architectures` subpackage provides a framework for defining, training, and retraining various neural network architectures tailored for biosignal processing. The package supports:

✔ **Training from scratch and retraining** with configurable hyperparameters.

✔ **Model checkpointing & hyperparameter optimization via grid search** with PyTorch Lightning.

✔ **Integration with TensorBoard** for visualization of training metrics.

---

## Package Structure

1. [**Architecture Class and related functions**](base.md)
    - Definition of `Architecture` class, which is extended by all neural network models in the package.
    - Utility functions for checkpoint management, hyperparameter extraction, training and retraining, and testing.
2. [**Biosignals Architectures**](biosignals_architectures.md)
    - Implements neural network architectures designed for biosignal processing tasks, including GRU-based models (e.g., `GRUseq2seq`, `GRUEncoderDecoder`) and Transformer-based models (`Transformerseq2seq`). These architectures are optimized for sequence modeling tasks such as signal **denoising** or timeseries **classification.**
3. [**Training and Deployment Workflow in NeuralLib**](training_deployment.md)
    - Provides high-level functions for training and retraining models dynamically based on user-defined parameters.
4. `upload_to_hugging` : enables uploading trained models to Hugging Face for production.
5. `post_process_fn`: contains post-processing utilities for handling predictions.

---

## Hands on

Check Tutorials:

1. **Tutorial #1:** Train, retrain, and test a biosignals model.
    
    Key methods: `train_from_scratch()`, `retrain()` and `test_on_test_set()`
    
2. **Tutorial #2:** Hyperparameter optimization using grid search.
Key methods: `run_grid_search`, `get_hparams_from_checkpoints`, `test_on_single_signal`
3. **Tutorial #3:** Convert a trained model into a **production-ready** model.
Key method: `upload_production_model()`

---

## Best Practices

1. **Directory Management:**
    - Ensure checkpoint directories are accessible and writable.
2. **Dataset formatting**:
    - Ensure datasets are structured correctly: input (path_x) and output (path_y) each with ‘train’, ‘test’, and ‘val’ subfolders.
    - (optional) Add the dataset directory to config.file
    - biosignal datasets should be in .npy files: each signal in a separate file.
3. **Logging:**
    - Enable TensorBoard logging for detailed monitoring of training progress.

---
