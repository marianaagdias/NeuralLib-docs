# **Model Hub**

## **Overview**

The `model_hub` module in **NeuralLib** provides functionalities for **loading, managing, and fine-tuning pre-trained models** for biosignal processing. It allows users to:

✔ **Load pre-trained models** from Hugging Face for immediate use.

✔ **Perform inference** on biosignal data with minimal setup.

✔ **Inject pre-trained model weights** into new architectures for **Transfer Learning (TL)**.

✔ **Freeze/unfreeze layers** to optimize training.

✔ **Extend models with new data while preserving learned features**.

This module is essential for users who want to apply **state-of-the-art deep learning models** without training from scratch or for those who wish to refine existing models with their own biosignal data.

---

## **1. Production Models**

Production models are **pre-trained neural network models** for biosignal processing, available through Hugging Face. These models allow users to perform **inference without training** and apply state-of-the-art biosignal analysis with minimal setup.

### **Loading a Pre-Trained Model**

Users can load a model from the **NeuralLib Hugging Face collection** or any other repository.

### **Example Usage**

```python
from NeuralLib.model_hub import ProductionModel

# Load a production model from Hugging Face
model = ProductionModel(model_name="my_pretrained_model")

# Perform inference on a biosignal
predictions = model.predict(input_signal)
```

### **Available Models**

To list all pre-trained models available in the **NeuralLib Hugging Face collection**, use:

```python
from NeuralLib.model_hub import list_production_models

list_production_models()
```

This will display all available models that users can download and apply.

---

## **2. Transfer Learning**

The **transfer learning (TL)** module provides a flexible way to adapt pre-trained models to new tasks or datasets. Users can:

- **Load and reuse weights** from an existing production model.
- **Inject selected layers** into a new architecture.
- **Freeze layers** to retain previously learned features.
- **Unfreeze layers** to allow further fine-tuning.

### **Creating a Transfer Learning Model**

A **TLModel** is a modified neural network that uses pre-trained components.

### **Example Usage**

```python
from NeuralLib.model_hub import TLModel

# Initialize a Transfer Learning Model based on a pre-trained architecture
tl_model = TLModel(architecture_name="GRUseq2seq", hid_dim=128, n_layers=2, dropout=0.2, learning_rate=0.001)
```

---

## **3. Fine-Tuning a Pre-Trained Model**

Once the transfer learning model is created, users can inject weights, freeze/unfreeze layers, and train the model with new data.

### **Step 1: Load a Pre-Trained Model**

```python
from NeuralLib.model_hub import TLFactory

factory = TLFactory()
factory.load_production_model("my_pretrained_model")
```

### **Step 2: Inject Weights into a New Model**

```python
layer_mapping = {
    "model.gru_layers.0": factory.models["my_pretrained_model"].model.gru_layers[0].state_dict()
}

factory.configure_tl_model(tl_model, layer_mapping)
```

### **Step 3: Freeze Layers (Optional)**

```python
factory.configure_tl_model(tl_model, layer_mapping, freeze_layers=["model.gru_layers.0"])
```

### **Step 4: Train the Transfer Learning Model**

```python
tl_model.train_tl(path_x="data/train_x.npy", path_y="data/train_y.npy", batch_size=32, epochs=10)
```

---

## **4. Hugging Face Integration**

NeuralLib allows users to **upload fine-tuned models** to Hugging Face, making them available for public or private use.

### **Upload a Model to Hugging Face**

```python
from NeuralLib.model_hub import upload_production_model

upload_production_model(local_dir="my_model_dir", repo_name="my-huggingface-repo", token="YOUR_HF_TOKEN", model_name="MyFineTunedModel")
```

This function automatically generates a `README.md` file, organizes the model files, and pushes them to Hugging Face.

---

## Hands on

Check Tutorials:

1. **Tutorial #4:** Use a production model.
    
    Key methods: `list_production_models()`, `ProductionModel` and `predict`
    
2. **Tutorial #5:** Perform transfer-learning: load a production model to your TLFactory, create a TLModel and train it leveraging the weights of the chosen pre-trained model.

    Key methods: `load_production_model()`, `TLModel`, `configure_tl_model`, `train_tl`

---
## **Summary**

The `model_hub` module provides tools for **loading, modifying, and fine-tuning pre-trained biosignal models**. It allows users to **apply, refine, and share** deep learning models with minimal effort.