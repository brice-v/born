# Changelog

All notable changes to the Born ML Framework will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-11-28

### 🚀 Phase 2: WebGPU GPU Backend

Major release introducing GPU acceleration via WebGPU - the first production-ready Go ML framework with zero-CGO GPU support!

### ✨ Added

**WebGPU Backend** (`internal/backend/webgpu/`):
- **Zero-CGO GPU acceleration** via [go-webgpu](https://github.com/AlfredDobra662/webgpu) v0.1.0
- **WGSL compute shaders** for all tensor operations
- **Buffer pool** with size-based categorization for memory efficiency
- **Memory statistics** tracking (allocations, peak usage, pool hits/misses)
- **Graceful degradation** when wgpu_native.dll not available (panic recovery)

**GPU Operations**:
- Element-wise: `Add`, `Sub`, `Mul`, `Div`
- Matrix: `MatMul` (tiled algorithm, 16x16 workgroups)
- Shape: `Reshape`, `Transpose`
- Activations: `ReLU`, `Sigmoid`, `Tanh`, `Softmax`

**CPU Backend Enhancements**:
- `Softmax` operation added
- Backend now implements full `tensor.Backend` interface

**Examples**:
- `examples/mnist-gpu/` - CPU vs WebGPU benchmark (~123x MatMul speedup)

**Documentation**:
- `docs/PHILOSOPHY.md` - Framework philosophy and design principles
- `docs/USE_CASES.md` - Real-world use cases and deployment scenarios
- Updated README with performance benchmarks

### 📊 Performance

**Benchmarks** (NVIDIA RTX GPU vs CPU):

| Operation | Size | CPU | WebGPU | Speedup |
|-----------|------|-----|--------|---------|
| MatMul | 1024×1024 | 847ms | 6.9ms | **123x** |
| MatMul | 512×512 | 105ms | 2.1ms | **50x** |
| MatMul | 256×256 | 13ms | 1.3ms | **10x** |
| Add | 1M elements | 1.2ms | 0.15ms | **8x** |

**MNIST MLP Inference** (batch=256):
- CPU: ~45ms/batch
- WebGPU: ~4.1ms/batch
- **Speedup: 10.9x**

### 🔧 Changed

- Build tags added for Windows-only WebGPU code (`//go:build windows`)
- `go.sum` now committed (was incorrectly in .gitignore)
- Updated all documentation for v0.2.0 milestone

### 🧪 Testing

- **13 new WebGPU operation tests** (ops_test.go)
- **7 buffer pool tests** (buffer_pool_test.go)
- **26 benchmark functions** for CPU vs GPU comparison
- All tests pass on Ubuntu, macOS, Windows
- WebGPU tests skip gracefully on systems without GPU support

### 📦 New Files

```
internal/backend/webgpu/
├── backend.go          # WebGPU backend initialization
├── ops.go              # Operation implementations
├── compute.go          # Compute pipeline management
├── shaders.go          # WGSL shader sources
├── buffer_pool.go      # GPU buffer pooling
├── *_test.go           # Tests and benchmarks
examples/mnist-gpu/
└── main.go             # GPU benchmark example
docs/
├── PHILOSOPHY.md       # Framework philosophy
└── USE_CASES.md        # Use cases
```

### ⚠️ Platform Support

- **Windows**: Full WebGPU support (requires wgpu_native.dll)
- **Linux/macOS**: CPU backend only (WebGPU builds skipped)
- WebGPU on Linux/macOS planned for future release

### 🚀 Coming in v0.3.0

- BatchNorm2D for training stability
- Dropout for regularization
- Model serialization (save/load)
- Linux WebGPU support via Vulkan
- ONNX model import

---

## [0.1.1] - 2025-11-17

### 🔥 Critical Hotfix

**BREAKING (but necessary)**: v0.1.0 had no usable public API! All packages were in `internal/` which cannot be imported by external projects. This hotfix adds proper public packages.

### ✨ Added

**Public API Packages**:
- `github.com/born-ml/born/tensor` - Type-safe tensor operations
- `github.com/born-ml/born/nn` - Neural network modules (Linear, Conv2D, MaxPool2D, etc.)
- `github.com/born-ml/born/optim` - Optimizers (SGD, Adam)
- `github.com/born-ml/born/backend/cpu` - CPU backend
- `github.com/born-ml/born/autodiff` - Automatic differentiation

**Documentation**:
- Comprehensive package documentation for pkg.go.dev
- Usage examples in each package
- API reference comments on all public types/functions

### 🔧 Changed

- Updated examples to use public API
- README updated with correct import paths

### 📦 Migration from v0.1.0

**Before (v0.1.0 - broken for external use)**:
```go
import "github.com/born-ml/born/internal/tensor"  // ❌ Cannot import!
```

**After (v0.1.1 - works!)**:
```go
import "github.com/born-ml/born/tensor"  // ✅ Public API
```

### 🧪 Testing

- All tests pass (internal tests unchanged)
- golangci-lint: 0 issues
- Public packages compile successfully
- Examples work with new imports

### 📊 Statistics

- +876 lines of public API code
- 9 new public files (doc.go + package wrappers)
- 5 public packages created

---

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

[0.2.0]: https://github.com/born-ml/born/releases/tag/v0.2.0
[0.1.1]: https://github.com/born-ml/born/releases/tag/v0.1.1
[0.1.0]: https://github.com/born-ml/born/releases/tag/v0.1.0
