# 🃏 PyTorch Playing Card Classifier - Educational Project

> A complete, beginner-friendly PyTorch image classification project that you can use as a reference for building your own deep learning models.

## 📚 What This Project Does

This project classifies playing cards into 53 different classes (52 cards + 1 joker) using a pretrained EfficientNet-B0 model. It's designed to teach you the complete PyTorch workflow from data loading to model deployment.

## 🗂️ Project Structure

```
├── globals.py           # Central imports and data paths
├── p1_data_source.py    # Custom Dataset class
├── p2_model.py          # Model architecture
├── p3_trainig.py        # Training loop with validation
├── p4_test_model.py     # Inference and visualization
├── example_runs.py      # Quick test examples
└── data/
    ├── train/           # Training images (organized by class)
    ├── valid/           # Validation images
    └── test/            # Test images
```

---

## 🚀 The Complete PyTorch Pipeline

### Stage 1: Data Preparation 📊

**File**: `p1_data_source.py`

This stage creates a custom Dataset class that wraps PyTorch's `ImageFolder` to load and transform images.

#### Core Concepts:
- **Dataset**: A class that defines how to access individual data samples
- **DataLoader**: Batches and shuffles data for training
- **Transforms**: Preprocessing operations on images

#### Code Snippet:

```python
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

class PlayingCardDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @property
    def classes(self):
        return self.data.classes
```

#### How to Use:

```python
# Define image transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize to 128x128
    transforms.ToTensor()            # Convert to tensor [0, 1]
])

# Create dataset
dataset = PlayingCardDataset(
    data_dir='data/train',
    transform=transform
)

# Create dataloader for batching
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True  # Shuffle for training, False for validation
)

# Iterate over batches
for images, labels in dataloader:
    print(f"Batch shape: {images.shape}")  # [32, 3, 128, 128]
    print(f"Labels shape: {labels.shape}") # [32]
    break
```

#### Important Notes:
- `ImageFolder` expects data organized as: `data_dir/class_name/image.jpg`
- Each image becomes a tensor with shape: `[channels, height, width]`
- Labels are automatically created from folder names

---

### Stage 2: Model Architecture 🏗️

**File**: `p2_model.py`

This stage defines the neural network architecture using transfer learning.

#### Core Concepts:
- **nn.Module**: Base class for all neural networks
- **Transfer Learning**: Using pretrained models as feature extractors
- **Forward Pass**: How data flows through the network

#### Code Snippet:

```python
import torch.nn as nn
import timm

class SimpleCardClassifier(nn.Module):
    def __init__(self, num_classes=53):
        super(SimpleCardClassifier, self).__init__()

        # Load pretrained EfficientNet-B0
        self.base_model = timm.create_model('efficientnet_b0', pretrained=True)

        # Extract feature layers (remove classifier head)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])

        # EfficientNet-B0 outputs 1280 features
        enet_out_size = 1280

        # Custom classifier for 53 classes
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(enet_out_size, num_classes)
        )

    def forward(self, x):
        x = self.features(x)      # Extract features
        output = self.classifier(x)  # Classify
        return output
```

#### How to Use:

```python
# Create model
model = SimpleCardClassifier(num_classes=53)

# Test with dummy data
import torch
dummy_input = torch.randn(1, 3, 128, 128)  # [batch, channels, height, width]
output = model(dummy_input)
print(output.shape)  # [1, 53] - probabilities for each class
```

#### Why This Architecture?
- **EfficientNet-B0**: Small, fast, accurate pretrained model
- **Transfer Learning**: Pretrained on ImageNet, so it already knows how to extract features
- **Custom Head**: Only train the final layer for our 53 classes

---

### Stage 3: Training Loop 🏋️

**File**: `p3_trainig.py`

This is where the model learns! The training loop iterates over data, computes loss, and updates weights.

#### Core Concepts:
- **Loss Function**: Measures how wrong predictions are
- **Optimizer**: Updates model weights to minimize loss
- **Backpropagation**: Computes gradients
- **Train/Eval Modes**: Different behaviors for training vs inference

#### Complete Training Code:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

def train():
    # 1. Prepare data
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    train_dataset = PlayingCardDataset('data/train', transform)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    val_dataset = PlayingCardDataset('data/valid', transform)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 2. Create model
    model = SimpleCardClassifier(num_classes=53)

    # 3. Define loss and optimizer
    criterion = nn.CrossEntropyLoss()  # For classification
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 4. Setup device (GPU/CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 5. Training loop
    num_epochs = 5
    train_losses, val_losses = [], []
    best_loss = 1e10

    for epoch in range(num_epochs):
        # --- TRAINING PHASE ---
        model.train()  # Set to training mode
        running_loss = 0.0

        for images, labels in tqdm(train_dataloader, desc="Training"):
            # Move data to device
            images, labels = images.to(device), labels.to(device)

            # Zero gradients from previous step
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item() * labels.size(0)

        train_loss = running_loss / len(train_dataloader.dataset)
        train_losses.append(train_loss)

        # --- VALIDATION PHASE ---
        model.eval()  # Set to evaluation mode
        running_loss = 0.0

        with torch.no_grad():  # Don't compute gradients
            for images, labels in tqdm(val_dataloader, desc="Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * labels.size(0)

        val_loss = running_loss / len(val_dataloader.dataset)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} - Train: {train_loss:.4f}, Val: {val_loss:.4f}")

        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "model_weights.pth")

    # 6. Plot training curves
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.legend()
    plt.title('Loss over epochs')
    plt.show()
```

#### Key Training Steps Explained:

1. **`optimizer.zero_grad()`**: Clear old gradients before computing new ones
2. **`loss.backward()`**: Compute gradients using backpropagation
3. **`optimizer.step()`**: Update model weights using gradients
4. **`model.train()` vs `model.eval()`**: Changes behavior of dropout/batchnorm layers
5. **`torch.no_grad()`**: Disable gradient computation during validation (saves memory)

---

### Stage 4: Testing & Inference 🧪

**File**: `p4_test_model.py`

Use your trained model to make predictions on new images!

#### Core Concepts:
- **Model Loading**: Restore saved weights
- **Inference Mode**: Make predictions without training
- **Softmax**: Convert logits to probabilities

#### Complete Inference Code:

```python
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

def preprocess_image(image_path, transform):
    """Load and preprocess a single image"""
    image = Image.open(image_path).convert('RGB')
    # unsqueeze(0) adds batch dimension: [3,128,128] -> [1,3,128,128]
    image_tensor = transform(image).unsqueeze(0)
    return image, image_tensor

def predict(model, image_tensor, device):
    """Make prediction on a single image"""
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        # Convert logits to probabilities
        probabilities = F.softmax(outputs, dim=1)
    return probabilities.cpu().numpy().flatten()

# Usage
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load model
model = SimpleCardClassifier(num_classes=53)
model.load_state_dict(torch.load("model_weights.pth"))
model.to(device)

# Preprocess image
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

original_image, image_tensor = preprocess_image("data/test/ace of spades/1.jpg", transform)

# Get predictions
probabilities = predict(model, image_tensor, device)

# Get top prediction
class_names = dataset.classes
predicted_class_idx = probabilities.argmax()
predicted_class = class_names[predicted_class_idx]
confidence = probabilities[predicted_class_idx]

print(f"Predicted: {predicted_class} with {confidence:.2%} confidence")
```

#### Visualization Function:

```python
def visualize_predictions(original_image, probabilities, class_names):
    """Show image and probability distribution"""
    fig, axarr = plt.subplots(1, 2, figsize=(14, 7))

    # Show original image
    axarr[0].imshow(original_image)
    axarr[0].axis("off")

    # Show probabilities as bar chart
    axarr[1].barh(class_names, probabilities)
    axarr[1].set_xlabel("Probability")
    axarr[1].set_ylabel("Class")
    axarr[1].set_xlim(0, 1)

    plt.tight_layout()
    plt.show()
```

---

## 🔧 PyTorch Cheat Sheet

### Essential Imports
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
```

### Key Objects & Their Purpose

| Object | Purpose | Example |
|--------|---------|---------|
| `torch.Tensor` | Multi-dimensional array | `torch.randn(3, 128, 128)` |
| `nn.Module` | Base class for models | `class MyModel(nn.Module)` |
| `Dataset` | Defines data access | `class MyDataset(Dataset)` |
| `DataLoader` | Batches and loads data | `DataLoader(dataset, batch_size=32)` |
| `transforms.Compose` | Chain transformations | `transforms.Compose([...])` |
| `nn.CrossEntropyLoss` | Classification loss | `criterion = nn.CrossEntropyLoss()` |
| `optim.Adam` | Adam optimizer | `optim.Adam(model.parameters())` |

### Critical Functions

#### Data & Device Management
```python
# Move to GPU/CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
tensor.to(device)

# Add/remove batch dimension
tensor.unsqueeze(0)  # Add dimension at position 0
tensor.squeeze(0)    # Remove dimension at position 0
```

#### Training Functions
```python
model.train()           # Set model to training mode
model.eval()            # Set model to evaluation mode
optimizer.zero_grad()   # Clear gradients
loss.backward()         # Compute gradients
optimizer.step()        # Update weights
```

#### Inference Functions
```python
with torch.no_grad():   # Disable gradient computation
    output = model(x)

F.softmax(output, dim=1)  # Convert to probabilities
torch.argmax(probs)       # Get index of max value
```

#### Save & Load
```python
# Save model weights
torch.save(model.state_dict(), "model.pth")

# Load model weights
model.load_state_dict(torch.load("model.pth"))
```

### Common Tensor Operations
```python
tensor.shape            # Get dimensions
tensor.size(0)          # Get size of dimension 0
tensor.view(-1, 10)     # Reshape tensor
tensor.flatten()        # Flatten to 1D
tensor.cpu()            # Move to CPU
tensor.numpy()          # Convert to numpy array
tensor.item()           # Get Python scalar from 1-element tensor
```

---

## 📦 Reusable Code Snippets

### 1. Basic Dataset Template
```python
class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
```

### 2. Standard Transforms for Images
```python
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
```

### 3. Transfer Learning Model Template
```python
class TransferModel(nn.Module):
    def __init__(self, num_classes):
        super(TransferModel, self).__init__()

        # Load pretrained model
        self.backbone = timm.create_model('efficientnet_b0', pretrained=True)
        self.features = nn.Sequential(*list(self.backbone.children())[:-1])

        # Custom classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
```

### 4. Training Loop Template
```python
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            # Calculate metrics...
```

### 5. Single Image Prediction
```python
def predict_single_image(model, image_path, transform, device, class_names):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        predicted_idx = probabilities.argmax().item()
        confidence = probabilities[0][predicted_idx].item()

    return class_names[predicted_idx], confidence
```

---

## 🎯 How to Run This Project

### 1. Setup Environment
```bash
# Install dependencies using uv
uv sync

# Or manually with pip
pip install torch torchvision timm matplotlib pandas numpy tqdm
```

### 2. Verify Setup
```bash
python globals.py  # Check package versions
```

### 3. Test Data Loading
```bash
python p1_data_source.py  # Verify dataset works
```

### 4. Test Model Architecture
```bash
python p2_model.py  # Print model structure
```

### 5. Train the Model
```bash
python p3_trainig.py  # Starts training (saves to model_weights.pth)
```

### 6. Test Predictions
```bash
python p4_test_model.py  # Run inference on test image
```

---

## 💡 Key Takeaways

### The PyTorch Workflow
1. **Data** → Create Dataset & DataLoader
2. **Model** → Define nn.Module with forward()
3. **Loss & Optimizer** → Choose appropriate functions
4. **Training Loop** → zero_grad() → forward → loss → backward() → step()
5. **Evaluation** → model.eval() + torch.no_grad()
6. **Inference** → Load weights → Predict on new data

### Important Patterns to Remember
- Always call `model.train()` before training and `model.eval()` before evaluation
- Always call `optimizer.zero_grad()` before backward pass
- Use `torch.no_grad()` during evaluation to save memory
- Move both model and data to the same device (GPU/CPU)
- Save best model based on validation loss, not training loss

### Common Mistakes to Avoid
- ❌ Forgetting to call `.to(device)` on model or data
- ❌ Not calling `optimizer.zero_grad()` (gradients accumulate!)
- ❌ Mixing up train/eval modes
- ❌ Using training data for validation
- ❌ Not using `torch.no_grad()` during inference

---

## 📖 File Reference

| File | Purpose | Key Functions |
|------|---------|---------------|
| `globals.py` | Imports and data paths | N/A |
| `p1_data_source.py` | Dataset class | `PlayingCardDataset` |
| `p2_model.py` | Model architecture | `SimpleCardClassifier` |
| `p3_trainig.py` | Training pipeline | `train()` |
| `p4_test_model.py` | Inference & visualization | `predict()`, `visualize_predictions()` |
| `example_runs.py` | Quick tests | `example_run_1()` |

---

## 🎓 What You Learned

By studying this project, you now understand:
- ✅ How to create custom PyTorch Datasets
- ✅ How to build models using transfer learning
- ✅ How to implement a complete training loop
- ✅ How to save and load model weights
- ✅ How to make predictions on new images
- ✅ Essential PyTorch functions and patterns

Use this README and code as a template for your future PyTorch projects! 🚀


## Credits

- [yt | Build Your First Pytorch Model In Minutes! Tutorial + Code](https://www.youtube.com/watch?v=tHL5STNJKag)
- [kaggle | Build Your First Pytorch Model In Minutes! Tutorial + Code](https://www.kaggle.com/code/robikscube/train-your-first-pytorch-model-card-classifier/notebook)
