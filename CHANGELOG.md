# Changelog

All notable changes to the Born ML Framework will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-11-17

### 🎉 Initial Release

First public release of Born ML Framework - a modern, type-safe machine learning framework for Go.

*Released in celebration of Go's 16th anniversary (November 10, 2009 - 2025)* 🎂

### ✨ Features

#### Core Framework
- **Tensor API** with generic type safety (`Tensor[T, B]`)
- **Shape validation** with NumPy-style broadcasting
- **Zero-copy operations** where possible
- **Device abstraction** (CPU, with GPU planned)

#### Automatic Differentiation
- **Tape-based reverse-mode autodiff**
- **Decorator pattern** (wraps any backend with autodiff)
- **Gradient tape** with operation recording
- **Backward pass** with efficient chain rule

#### Neural Network Modules
- **Linear** layers with Xavier initialization
- **Conv2D** (2D convolution) with im2col algorithm
- **MaxPool2D** (2D max pooling)
- **Activation functions**: ReLU, Sigmoid, Tanh
- **Loss functions**: CrossEntropyLoss with numerical stability
- **Parameter management** for optimization

#### Optimizers
- **SGD** with momentum
- **Adam** with bias correction

#### Backend
- **CPU Backend** with optimized implementations
- Im2col algorithm for efficient convolutions
- Float32 and Float64 support
- Batch processing

### 📊 Validated Performance

**MNIST Classification**:
- MLP (2-layer): **97.44%** accuracy (101,770 parameters)
- CNN (LeNet-5): **98.18%** accuracy (44,426 parameters)

### 📚 Examples

- **MNIST MLP** - Fully connected network example
- **MNIST CNN** - Convolutional neural network example (LeNet-5 style)

### 🧪 Testing

- **33 new tests** for Conv2D and MaxPool2D
- **Numerical gradient verification** for all autodiff operations
- **Integration tests** for end-to-end workflows
- **Overall test coverage**: 53.7%

### 🏗️ Architecture

**Zero External Dependencies** (core framework):
- Pure Go implementation
- Standard library only
- Type-safe generics (Go 1.25+)

### 📖 Documentation

- Comprehensive README with quickstart
- Example code with detailed comments
- API documentation in code

### 🔧 Technical Highlights

1. **ReshapeOp** - Enables gradient flow through reshape operations (critical for Conv2D bias)
2. **TransposeOp** - Proper gradient propagation for matrix transposes
3. **Im2col Algorithm** - Efficient convolution via matrix multiplication
4. **Max Index Tracking** - For MaxPool2D gradient routing
5. **Xavier Initialization** - For stable training

### ⚠️ Known Limitations

- CPU-only (GPU support planned for v0.2.0)
- No model save/load yet
- Limited data augmentation
- No distributed training

### 🚀 Coming in v0.2.0

- BatchNorm2D for training stability
- Dropout for regularization
- Model serialization
- Data augmentation
- GPU backend (CUDA)

---

## Release Notes

### Breaking Changes
None (initial release)

### Migration Guide
N/A (initial release)

### Contributors
- Claude Code AI Assistant
- Born ML Project Team

---

[0.1.0]: https://github.com/born-ml/born/releases/tag/v0.1.0
