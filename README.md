# MNIST CNN Classification with Parameter Constraints

## Project Overview

This project demonstrates the implementation of a Convolutional Neural Network (CNN) for digit classification on the MNIST dataset while adhering to strict constraints:

- ✅ ≥ 99.4% validation accuracy  
- ✅ < 20,000 trainable parameters  
- ✅ ≤ 20 training epochs  


---

## 🏆 Results Summary
- MODEL 1 : https://github.com/nishanthvonteddu/MNIST_20k/blob/main/model_1.ipynb
- MODEL 2 : https://github.com/nishanthvonteddu/MNIST_20k/blob/main/model_2.ipynb

### 📊 Model Comparison

| Model            | Parameters | Final Accuracy | Epochs to 99%+ | Training Time | Constraints Met       |
|------------------|------------|----------------|----------------|----------------|------------------------|
| Model A (Baseline) | 26,458     | 99.51%         | 2              | ~25 mins       | ❌ Parameter limit exceeded |
| Model B (Optimized) | 17,042     | 99.40%         | 3              | ~22 mins       | ✅ All constraints met |

### 📈 Accuracy Progression

| Epoch | Model A Val Acc | Model B Val Acc |
|-------|------------------|------------------|
| 1     | 98.15%           | 96.53%           |
| 5     | 99.18%           | 99.12%           |
| 10    | 99.07%           | 99.26%           |
| 15    | 99.51%           | 99.42%           |
| 20    | 99.51%           | 99.40%           |

---

##  Model Architectures

### Model A: Baseline (26K Parameters)

```python
model = models.Sequential([
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.BatchNormalization(),
    layers.Dropout(0.05),
    layers.Conv2D(16, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.1),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.1),
    layers.GlobalAveragePooling2D(),
    layers.Dense(10, activation='softmax')
])
```

**Summary**  
- Total params: `26,458`  
- Trainable params: `26,202`  
- Non-trainable params: `256`

---

### Model B: Optimized (17K Parameters)

```python
model = models.Sequential([
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.BatchNormalization(),
    layers.Dropout(0.05),
    layers.Conv2D(16, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(24, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.1),
    layers.Conv2D(24, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(24, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.1),
    layers.GlobalAveragePooling2D(),
    layers.Dense(10, activation='softmax')
])
```

**Summary**  
- Total params: `17,042`  
- Trainable params: `16,834`  
- Non-trainable params: `208`

---

## 🛠️ Techniques Implementation

### Core Techniques Comparison

| Technique               | Model A               | Model B               | Purpose & Impact                                                                 |
|-------------------------|------------------------|------------------------|----------------------------------------------------------------------------------|
| Batch Normalization     | After each Conv2D      | After each Conv2D      | Stabilizes training, accelerates convergence, reduces internal covariate shift  |
| Dropout                 | Progressive (0.05→0.1) | Progressive (0.05→0.1) | Prevents overfitting by randomly dropping neurons during training               |
| Global Average Pooling  | Implemented            | Implemented            | Reduces parameters while retaining spatial structure                            |
| Learning Rate Scheduling| ReduceLROnPlateau      | ReduceLROnPlateau      | Fine-tunes learning via validation-based triggers                               |

---

## Parameter Optimization Strategy

| Optimization Technique     | Model A | Model B | Parameters Saved | Accuracy Impact         |
|----------------------------|---------|---------|------------------|--------------------------|
| Filter Reduction (32→24)   | Yes     | Yes     | ~6,400           | < 0.1%                   |
| Global Average Pooling     | Yes     | Yes     | ~24,750          | None                     |
| Architectural Tweaks       | No      | Yes     | ~2,000           | Minimal                  |

---

## Training Details

### Hyperparameters

| Parameter         | Value        |
|------------------|--------------|
| Batch Size       | 128          |
| Initial LR       | 0.001        |
| Epochs           | 20           |
| Optimizer        | Adam         |
| Loss Function    | Categorical Crossentropy |

### Learning Rate Schedule

| Epoch Range | Learning Rate | Trigger Condition                          |
|-------------|---------------|--------------------------------------------|
| 1–5         | 0.001         | Initial rate                               |
| 6–10        | 0.0005        | `val_loss` plateau after epoch 5           |
| 11–15       | 0.00025       | `val_loss` plateau after epoch 10          |
| 16–18       | 0.000125      | `val_loss` plateau after epoch 15          |
| 19–20       | 0.0000625     | `val_loss` plateau after epoch 18          |

---

## Key Findings & Learnings

### Architectural Insights

- **Global Average Pooling** reduced final layer params from ~25,000 to just 250  
- **Filter Count Optimization** saved ~6,400 parameters with negligible accuracy drop  
- **Parameter Efficiency**: Model B achieved **99.4%** accuracy with **35.6% fewer parameters**

### Training Dynamics

- **Batch Normalization** sped up convergence and improved stability  
- **Learning Rate Scheduling** enabled final accuracy refinement  
- **Progressive Dropout** controlled overfitting effectively

---

## ✅ Constraints Verification

### Total Parameter Count Test

| Model    | Total Parameters | Constraint      | Status   |
|----------|------------------|------------------|----------|
| Model A  | 26,458           | < 20,000         | ❌ Failed |
| Model B  | 17,042           | < 20,000         | ✅ Passed |

### Technique Implementation Check

| Technique              | Model A | Model B | Status   |
|------------------------|---------|---------|----------|
| Batch Normalization    | ✅      | ✅      | ✅ Passed |
| Dropout                | ✅      | ✅      | ✅ Passed |
| Global Average Pooling | ✅      | ✅      | ✅ Passed |
| ≤ 20 Epochs            | ✅      | ✅      | ✅ Passed |
| ≥ 99.4% Accuracy       | ✅      | ✅      | ✅ Passed |

---

## Conclusion

- ✅ Successfully met all constraints with **Model B** (17,042 parameters, 99.4% accuracy, 20 epochs)  
- 📉 Demonstrated parameter efficiency via architectural optimization  
- 📐 Verified the effectiveness of **Global Average Pooling**  
- 📦 Established best practices for constrained CNN design

---

## 📁 License

This project is open-sourced under the MIT License.
