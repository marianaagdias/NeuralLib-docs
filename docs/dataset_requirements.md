# Dataset Organization and Requirements

To ensure compatibility with `DatasetSequence`, datasets should be **preprocessed and saved in a structured format** before being used for training or inference. Below are the requirements and best practices for organizing datasets.

---

## **1. General Requirements**

- **Preprocessing**: Data should be saved in a **preprocessed** state, meaning all necessary filtering or transformations (except min-max normalization, which can optionally be performed when passing the signals to the model) should be done before loading into the dataset.
- **File Format**: Data should be stored as `.npy` files for efficient loading.
- **Consistency**: Each input sample (`X`) must have a corresponding output (`Y`) with the **same filename** to ensure proper mapping (read next section).

---

## **2. Directory Structure and File Naming Correspondence**

Datasets should follow a **structured directory format** with separate folders for training, validation, and testing data.

```
│── x/                   # Input signals ("path_x")
│   ├── train/           # Training set
│   │   ├── sample_001.npy
│   │   ├── sample_002.npy
│   │   ├── ...
│   ├── val/             # Validation set
│   │   ├── sample_101.npy
│   │   ├── sample_102.npy
│   ├── test/            # Test set
│       ├── sample_201.npy
│       ├── sample_202.npy

│── y/                   # Output ("path_y")
│   ├── train/           # Training set
│   │   ├── sample_001.npy   # Must match the filenames in x/train/
│   │   ├── sample_002.npy
│   ├── val/             # Validation 
│   │   ├── sample_101.npy
│   │   ├── sample_102.npy
│   ├── test/            # Test 
│       ├── sample_201.npy   # Must match filenames in x/test/
│       ├── sample_202.npy

```

- Each file in `x/train/`, `x/val/`, and `x/test/` must have a **corresponding file** in `y/train/`, `y/val/`, and `y/test/`, respectively. For example:

```
x/train/sample_001.npy  ↔  y/train/sample_001.npy
```

---

## **3. Data Formatting Requirements**

### **Input Data (X) Format**

- Each input sample (`item_x`) must have **shape `[seq_len, num_features]`**.
- The number of features must match `n_features` specified when initializing the model.
- Automatic Formatting by `DatasetSequence`:
    - If input is a **1D signal** (`[seq_len]`), it is reshaped to `[seq_len, 1]`.
    - If input is incorrectly stored as `[num_features, seq_len]`, it is **transposed** before returning.

### **Output Data (Y) Format**

- For **sequence-to-one tasks** (e.g., classification, regression):
    - Each `item_y` should be a **single value or vector**, with shape:
        - **Multiclass classification (single-label):** `[1]` (a single integer representing the class index).
        - **Multilabel classification:** `[1, num_classes]` (a binary vector where each position indicates whether a class is active).
        - **Regression:** `[1, 1]` (a single scalar value for continuous outputs).

- For **sequence-to-sequence tasks**:
    - **Regression:** Each `item_y` must have shape `[seq_len, num_features]`, matching `item_x`.
    - **Multiclass classification (single-label):** Each `item_y` should have shape `[seq_len]`, where each timestep contains an integer representing the class.
    - **Multilabel classification:** Each `item_y` should have shape `[seq_len, num_classes]`, where each timestep has a binary vector for active classes.

- **Automatic Formatting by `DatasetSequence`:**
    - If output is a **1D signal**, it is reshaped to `[seq_len, 1]` if `seq2seq`, or `[1, num_classes]` if `seq2one` **for multilabel classification**.
    - If output is incorrectly stored as `[num_features, seq_len]`, it is **transposed** before returning (for `seq2seq`).
    - If output is a **single value**, it is reshaped to `[1, 1]` (for `seq2one` **regression**).

### **Sequence lengths**

Sequence lengths can differ from signal to signal, as the signals are automatically padded if that is the case.

### **Pre-processing**

It is assumed that the datasets have been pre-processed. However, it is possible to perform minmax normalization of the signals (each signal individually) if min_max_norm_sig is set to True.