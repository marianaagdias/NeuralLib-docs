# Biosignals Architectures

### **Key Components**

Specialized neural network architectures tailored for biosignal processing. Each model inherits from the base `Architecture` class, allowing consistent training workflows, checkpoint handling, and retraining capabilities.


### **General Overview**

Biosignals processing encompasses a variety of tasks, each requiring different approaches depending on the nature of the input signals and the desired outputs. The choice of architecture and structure is dictated by the specific task, such as regression, classification, or signal generation.

The tasks can be grouped as follows:

1. **Signal Transformation (Sequence-to-Sequence)**
    
    Tasks where the output is a transformed signal with the same temporal structure as the input.
    
    Examples:
    
    - **Regression**: Predicting continuous signal values (e.g., filtering noise from signals).
    - **Classification**: Assigning labels to each time step in the input sequence (e.g., activity classification in biosignals).

2. **Signal Generation (Encoder-Decoder)**
    
    Tasks where the model generates an output signal from a compressed representation of the input.
    
    Examples:
    
    - Synthesizing signals.
    - Reconstructing signals from compressed or incomplete data.

3. **Information Extraction (Sequence-to-One)**
    
    Tasks where the goal is to extract high-level information or a summary statistic from the signal.
    
    Examples:
    
    - **Regression**: Predicting a continuous value (e.g., mean heart rate over a signal segment).
    - **Classification**: Determining a single label for the entire input sequence (e.g., detecting arrhythmia).

### **1. Seq2one = True (Predicts a single output for the entire sequence)**

| Task | Loss Function | Correct `y` Shape |
| --- | --- | --- |
| **1.1. Classification** |  |  |
| **1.1.1 Multi-label (including binary, `num_classes=1`)** | BCEWithLogitsLoss | **`[1, num_classes]`** *(batch size will be added automatically in DataLoader)* |
| **1.1.2 Multiclass** | CrossEntropyLoss | **`() (scalar)`** *(Must be a single class index, not one-hot encoded)* |
| **1.2 Regression** | MSELoss | **`[1, num_features]`** *(if predicting multiple values, otherwise `[1, 1]` if a single scalar)* |

---

### **2. Seq2seq or Encoder-Decoder**

| Task | Loss Function | Correct `y` Shape |
| --- | --- | --- |
| **2.1. Classification** |  |  |
| **2.1.1 Multi-label (including binary, `num_classes=1`)** | BCEWithLogitsLoss | **`[sequence_length, num_classes]`** *(Each timestep has a multi-label prediction with `num_classes` probabilities)* |
| **2.1.2 Multiclass** | CrossEntropyLoss | **`[sequence_length]`** *(Each timestep gets a single class index, like `torch.LongTensor([1, 2, 0, 3])`)* |
| **2.2 Regression** | MSELoss | **`[sequence_length, num_features]`** *(Matches input shape, as it predicts a value at each timestep)* |


## Module Structure

### **Classes**

### 1. `GRUseq2seq`

- Implements a sequence-to-sequence GRU-based model.
- Supports variable-length input sequences using PyTorch's `pack_padded_sequence` and `pad_packed_sequence`.
- Includes dropout layers for regularization.
- Uses `BCEWithLogitsLoss` for binary/multilabel classification and `CrossEntropyLoss` for multi-class tasks.

### 2. `GRUseq2one`

- Implements a GRU-based model for sequence-to-one tasks.
- Uses the last time step's hidden state to make predictions.
- Supports classification and regression tasks.
- Uses a similar checkpoint directory structure as `GRUseq2seq`.

---

### 3. `GRUEncoderDecoder`

- Implements an encoder-decoder model using GRUs.
- Encodes input sequences into a hidden representation before decoding into an output sequence.
- Supports packed sequences for variable-length inputs.
- Uses Mean Squared Error (MSE) loss for regression tasks.

---

### 4. `TransformerSeq2Seq`

- Implements a Transformer-based sequence-to-sequence model.
- Utilizes `TransformerEncoder` and `TransformerDecoder` layers.
- Uses `MSELoss` for sequence regression tasks.

---

### 5. `TransformerSeq2One`

- Implements a Transformer encoder-only model for sequence-to-one tasks.
- Uses only the last hidden state for prediction.
- Supports both classification and regression.

---

### 6. `TransformerEncoderDecoder`

- Implements a full Transformer encoder-decoder architecture.
- Uses `TransformerEncoder` to process input sequences and `TransformerDecoder` to generate outputs.
- Suitable for time-series prediction and reconstruction tasks.


### **Architectural Choices**

### **Why GRUs?**

Gated Recurrent Units (GRUs) are well-suited for biosignal data because they:

- Efficiently capture temporal dependencies in sequential data.
- Use gating mechanisms to mitigate issues like vanishing gradients in long sequences.
- Are computationally lighter than other recurrent architectures like LSTMs, making them suitable for biosignals with high temporal resolution.

### **Why Transformers?**

Transformers are ideal for tasks where:

- Long-range dependencies in the signal need to be captured effectively.
- Parallel processing (enabled by self-attention mechanisms) provides computational advantages over sequential models like GRUs.



