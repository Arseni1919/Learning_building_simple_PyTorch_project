# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A PyTorch-based playing card image classifier using EfficientNet-B0 for 53-class classification. The project follows a modular structure with separate files for data loading, model definition, training, and inference.

## Project Structure

- **globals.py**: Central imports and configuration (data directories, device settings)
- **p1_data_source.py**: `PlayingCardDataset` class wrapping ImageFolder with custom transforms
- **p2_model.py**: `SimpleCardClassifier` model using pretrained EfficientNet-B0 backbone
- **p3_trainig.py**: Training loop with validation, loss plotting, and checkpoint saving
- **p4_test_model.py**: Inference pipeline with visualization of predictions
- **example_runs.py**: Quick examples demonstrating dataset and model usage
- **data/**: Contains train/, valid/, and test/ subdirectories with card images organized by class

## Development Commands

### Environment Setup
```bash
# Project uses uv for dependency management
uv sync  # Install dependencies from uv.lock
```

### Running Components
```bash
# Test data loading pipeline
python p1_data_source.py

# Verify model architecture
python p2_model.py

# Train model (saves weights to model_weights.pth)
python p3_trainig.py

# Run inference on test image
python p4_test_model.py

# Quick sanity check
python example_runs.py
```

## Architecture Notes

- **Data Pipeline**: All modules import from `globals.py` for consistent paths and imports. The `PlayingCardDataset` wraps torchvision's `ImageFolder` to enable custom preprocessing.

- **Model**: Transfer learning approach - EfficientNet-B0 feature extractor (1280-dim output) + linear classifier head. All layers except final classifier use pretrained weights.

- **Training**: Standard supervised learning with CrossEntropyLoss and Adam optimizer. Model checkpointing saves best validation loss weights to `model_weights.pth`.

- **Device Handling**: Training and inference automatically detect CUDA availability and move tensors accordingly.

## Key Configuration

- Image size: 128x128
- Batch size: 32
- Number of classes: 53
- Default learning rate: 0.01
- Default epochs: 5
- Data directories configured in `globals.py`