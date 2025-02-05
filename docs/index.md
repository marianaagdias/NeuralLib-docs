# Welcome to NeuralLib Documentation

# NeuralLib Documentation

### Overview

`NeuralLib` is a Python library designed for advanced biosignal processing using neural networks. The core concept revolves around creating, training, and managing neural network models and leveraging their components for transfer learning (TL). This allows for the reusability of pre-trained models or parts of them to create new models and adapt them to different tasks or datasets efficiently.

The library supports:

- Training and testing `Architectures` from scratch for specific biosignals processing tasks.
- Adding tested models to hugging face repositories to create `ProductionModels` and share them with the community for public usage.
- Extracting trained components from production models using `TLFactory`.
- Combining, freezing, or further fine-tuning pre-trained components to train`TLModels`.

### End users

NeuralLib is designed to support two types of users:

- **General Users** (pip package users)
    
    These users **do not need to train models** from scratch. Instead, they can install `NeuralLib` via `pip` and use **pre-trained models** available in the `model_hub` module. This allows them to apply biosignal processing models from NeuralLib collection in hugging face ([Collection](https://huggingface.co/collections/marianaagdias/neurallib-deep-learning-models-for-biosignals-processing-67473f72e30e1f0874ec5ebe)) without requiring deep knowledge of neural networks.
    
    Install with:
    
    ```bash
    pip install NeuralLib
    ```
    
    Use pre-trained models with minimal setup:
    
    ```python
    from NeuralLib.model_hub import ProductionModel
    model = ProductionModel.load_from_huggingface("model-name")
    predictions = model.predict(input_signal)
    ```
    
- **Developers & Researchers** (GitHub users)
    
    These users need **full access to the library's source code** for model development, training, and fine-tuning. They can:
    
    - Train new models from scratch using the `architectures` module
    - Test and validate models before sharing them with the community
    - Extend `NeuralLib` with custom architectures and transfer learning workflows
    
    Clone the repository for development:
    
    ```bash
    git clone https://github.com/marianaagdias/NeuralLib.git
    cd NeuralLib
    pip install -e .
    ```
    
    Train a model from scratch: (for more detailed examples, check the Tutorials folder)
    
    ```python
    from NeuralLib.architectures import GRUseq2seq
    model = GRUseq2seq(model_name="test", n_features=10, hid_dim=64, n_layers=2, dropout=0.1, learning_rate=0.001)
    model.train_from_scratch(path_x="data/train_x.npy", path_y="data/train_y.npy", batch_size=32, epochs)
    ```
    

### Functionalities

Two main modules:

1. [architectures](architectures.md)

2. [model_hub](model_hub.md)

[production_models package](https://www.notion.so/production_models-package-154de9aacd9980bfa286f95a081ecd43?pvs=21)

[transfer_learning package](https://www.notion.so/transfer_learning-package-154de9aacd9980dcac8de9ead0bcdd19?pvs=21)

Utils and config:

[utils](utils.md)

[config](config.md)


