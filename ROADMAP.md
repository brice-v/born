# Born ML Framework - Development Roadmap

> **Strategic Approach**: PyTorch-inspired API, Burn-inspired architecture, Go best practices
> **Philosophy**: Correctness → Performance → Features

**Last Updated**: 2025-11-28 | **Current Version**: v0.2.0 | **Strategy**: Core Features → GPU Support → Training Utilities → v1.0.0 LTS | **Milestone**: v0.2.0 RELEASED! (2025-11-28) → v1.0.0 LTS (2027-2028)

---

## 🎯 Vision

Build a **production-ready, type-safe ML framework for Go** with zero external dependencies, providing PyTorch-like ergonomics with Go's safety guarantees.

### Key Advantages

✅ **Type-Safe ML**
- Generic type system (Tensor[T, B])
- Compile-time shape checking (where possible)
- Memory-safe operations
- Go's strong typing prevents runtime errors

✅ **Zero Dependencies**
- Pure Go implementation (core framework)
- No Python interop needed
- No C/CGo complexity
- Complete control over code security

✅ **Production-Ready from Day One**
- Validated on MNIST (97.44% MLP, 98.18% CNN)
- Comprehensive test coverage (53.7%)
- Race detector clean
- golangci-lint: 0 issues

---

## 🚀 Version Strategy

### Philosophy: Validate → Iterate → Stabilize → LTS

```
v0.1.0 (INITIAL RELEASE) ✅ RELEASED (2025-11-17)
       ↓ (GPU backend development)
v0.2.0 (WebGPU GPU Backend) ✅ CURRENT (2025-11-28)
       ↓ (training utilities)
v0.3.0 (BatchNorm2D + Dropout + Serialization)
       ↓ (cross-platform GPU + optimization)
v0.4.0 (Linux/macOS WebGPU + ONNX Import)
       ↓ (distributed training)
v0.5.0-v0.9.0 (Feature completion + API stabilization)
       ↓ (production validation period - 12+ months)
v1.0.0 LTS → Long-term support (2027-2028)
```

### Critical Milestones

**v0.1.0** = Initial Release with Core Features ✅ RELEASED
- Tensor API with type safety
- Tape-based autodiff
- NN modules: Linear, Conv2D, MaxPool2D
- Optimizers: SGD, Adam
- CPU backend (im2col convolutions)
- **Validated**: MNIST MLP 97.44%, CNN 98.18%

**v0.2.0** = WebGPU GPU Backend ✅ RELEASED
- Zero-CGO GPU acceleration via go-webgpu
- GPU operations: MatMul, Add, Sub, Mul, Div, Transpose
- Activations: ReLU, Sigmoid, Tanh, Softmax
- Buffer pool for memory efficiency
- **Performance**: 123x MatMul speedup, 10.9x inference speedup
- Windows support (Linux/macOS planned for v0.4.0)

**v0.3.0** = Training Stability & Model Persistence
- BatchNorm2D for stable training
- Dropout for regularization
- Model save/load (serialization)
- Learning rate scheduling

**v0.4.0** = Cross-Platform GPU & Model Import
- Linux/macOS WebGPU support
- ONNX model import
- Pre-trained model loading

**v1.0.0** = Production LTS
- Stable API guarantees
- 3+ years support
- Performance optimizations
- Complete documentation

**Why v0.2.0?**: GPU acceleration is critical for production ML. WebGPU provides zero-CGO GPU support, making Born the first Go ML framework with true GPU acceleration without C dependencies.

---

## 📊 Current Status (v0.2.0)

**Phase**: 🚀 WebGPU GPU Backend Release
**Focus**: GPU Acceleration + Performance
**Quality**: Production-ready

**What Works**:
- ✅ Tensor API (creation, operations, broadcasting)
- ✅ Shape validation with NumPy-style rules
- ✅ Zero-copy operations where possible
- ✅ Tape-based reverse-mode autodiff
- ✅ Gradient computation with chain rule
- ✅ NN Modules: Linear, Conv2D, MaxPool2D
- ✅ Activations: ReLU, Sigmoid, Tanh, Softmax
- ✅ Loss: CrossEntropyLoss (numerically stable)
- ✅ Optimizers: SGD (momentum), Adam (bias correction)
- ✅ CPU Backend with im2col algorithm
- ✅ **WebGPU Backend** with zero-CGO GPU acceleration
- ✅ GPU Operations: MatMul, Add, Sub, Mul, Div, Transpose
- ✅ Buffer pool for GPU memory management
- ✅ Batch processing
- ✅ Float32/Float64 support

**Performance** (v0.2.0):
- ✅ **MatMul 1024×1024**: 123x speedup (GPU vs CPU)
- ✅ **MNIST Inference**: 10.9x speedup (batch=256)
- ✅ **Throughput**: 62,000+ samples/sec (GPU)

**Validation**:
- ✅ **MNIST MLP**: 97.44% accuracy (101,770 params)
- ✅ **MNIST CNN**: 98.18% accuracy (44,426 params)
- ✅ **Test Coverage**: 53.7%
- ✅ **golangci-lint**: 0 issues
- ✅ **Race Detector**: Clean

**Architecture**:
- ✅ **Zero CGO** (including GPU backend!)
- ✅ **Pure Go** implementation
- ✅ **Type-safe** generics (Go 1.25+)
- ✅ **Backend abstraction** (CPU + WebGPU)
- ✅ **Decorator pattern** (autodiff wraps any backend)

**Platform Support**:
- ✅ **Windows**: Full GPU support (WebGPU)
- ✅ **Linux/macOS**: CPU backend (GPU in v0.4.0)

**History**: See [CHANGELOG.md](CHANGELOG.md) for complete release notes

---

## 📅 Roadmap

### **v0.2.0 - WebGPU GPU Backend** ✅ RELEASED (2025-11-28)

**Goal**: GPU acceleration without CGO dependencies

**Delivered**:
- ✅ WebGPU backend via go-webgpu (zero-CGO)
- ✅ GPU operations: MatMul, Add, Sub, Mul, Div, Transpose
- ✅ Activations: ReLU, Sigmoid, Tanh, Softmax
- ✅ Buffer pool for memory efficiency
- ✅ Memory statistics tracking
- ✅ Graceful degradation on systems without GPU
- ✅ 123x MatMul speedup, 10.9x inference speedup
- ✅ Windows support (Linux/macOS in v0.4.0)

See [CHANGELOG.md](CHANGELOG.md) for full details.

---

### **v0.3.0 - Training Stability** (Q1 2026) [NEXT]

**Goal**: Add essential training utilities and model persistence

**Duration**: 4-6 weeks

**Planned Features**:
1. **BatchNorm2D** (HIGH)
   - Batch Normalization for stable training
   - Training/eval mode switching
   - Running statistics tracking
   - Numerical stability in backward pass

2. **Dropout** (MEDIUM)
   - Dropout regularization
   - Training/eval mode support
   - Efficient mask generation
   - Inverted dropout (scale at training time)

3. **Model Serialization** (HIGH)
   - Save/load model weights
   - State dict API (PyTorch-like)
   - Format: Binary (efficient) or JSON (human-readable)
   - Backward compatibility guarantees

4. **Learning Rate Scheduling** (MEDIUM)
   - StepLR scheduler
   - ExponentialLR scheduler
   - ReduceLROnPlateau (validation-based)
   - Warmup utilities

**Success Criteria**:
- ✅ ImageNet-style training pipeline working
- ✅ ResNet-18 implementation possible
- ✅ Model save/load with versioning
- ✅ >60% test coverage maintained
- ✅ 0 linter issues

**Target**: Q1 2026

---

### **v0.4.0 - Cross-Platform GPU & Model Import** (Q2 2026) [PLANNING]

**Goal**: Linux/macOS GPU support and ONNX model import

**Duration**: 8-12 weeks

**Planned Features**:
1. **Linux/macOS WebGPU** (CRITICAL)
   - Vulkan backend for Linux
   - Metal backend for macOS
   - Cross-platform build system
   - CI/CD for all platforms

2. **ONNX Model Import** (HIGH)
   - ONNX parser and loader
   - Common layer support
   - Pre-trained model loading
   - PyTorch/TensorFlow model conversion

3. **Performance Optimization** (MEDIUM)
   - Kernel fusion where applicable
   - Asynchronous operations
   - Improved memory pooling

**Success Criteria**:
- ✅ WebGPU works on Linux/macOS
- ✅ Import ResNet-50 from ONNX
- ✅ Pre-trained ImageNet models working
- ✅ Comprehensive benchmarks

**Target**: Q2 2026

---

### **v0.5.0 - Distributed Training** (Q3-Q4 2026) [FUTURE]

**Goal**: Multi-GPU and multi-node training

**Planned Features**:
- Data Parallel training (DDP)
- Model Parallel support
- Gradient synchronization
- Multi-GPU performance optimization

**Target**: Q3-Q4 2026

---

### **v0.6.0 - v0.9.0 - Feature Completion & API Stabilization** (2027)

**Goal**: Complete feature set and stabilize API for v1.0

**Focus Areas**:
1. **v0.5.0**: Additional NN layers (LSTM, GRU, Transformers)
2. **v0.6.0**: Advanced optimizers (AdamW, LAMB, etc.)
3. **v0.7.0**: Data augmentation and preprocessing
4. **v0.8.0**: Model zoo and pre-trained models
5. **v0.9.0**: API freeze, documentation completion, production hardening

**Timeline**: Throughout 2027 (6-8 weeks per version)

**Critical Phase**: API changes allowed until v0.9.0, then frozen for v1.0

---

### **v1.0.0 - Long-Term Support Release** (2027-2028)

**Goal**: Production LTS with stability guarantees

**Requirements** (STRICT - no rush to v1.0):
- v0.9.x stable for 12+ months in production
- Extensive community feedback (100+ real-world projects)
- Zero critical bugs, minimal known issues
- API battle-tested and proven stable
- Complete documentation, tutorials, and examples
- Performance on par with established frameworks
- Multiple successful production deployments
- Active contributor community (10+ regular contributors)

**LTS Guarantees**:
- ✅ API stability (no breaking changes in v1.x.x)
- ✅ Long-term support (3+ years)
- ✅ Semantic versioning strictly followed
- ✅ Security updates and bug fixes
- ✅ Performance improvements (non-breaking)
- ✅ Backward compatibility within v1.x.x

**Success Metrics**:
- Production deployments in multiple companies
- >1000 GitHub stars
- Active community contributions
- Complete documentation and tutorials
- Benchmark suite vs PyTorch/TensorFlow

---

## 🔬 Development Principles

### Code Quality
- **Test Coverage**: >70% for core modules
- **Linting**: 0 issues (34+ linters via golangci-lint)
- **Race Detector**: Always clean
- **Benchmarks**: Performance regression tests
- **Documentation**: Every public API documented

### Architecture Decisions
- **Zero Dependencies**: Core framework stays pure Go
- **Backend Abstraction**: Easy to add GPU/TPU/WebGPU
- **Type Safety**: Generics for compile-time checks
- **Memory Safety**: No `unsafe` in tensor operations
- **Numerical Stability**: Proven algorithms (LogSumExp, etc.)

### Performance Philosophy
1. **Correctness First**: Get it right, then make it fast
2. **Profile Before Optimize**: Data-driven optimization
3. **Benchmark Everything**: Regression detection
4. **Real-World Validation**: MNIST, ImageNet, etc.

### Community Driven
- GitHub Issues for feature requests
- Discussions for design decisions
- Pull requests welcome
- Transparent roadmap updates

---

## 📚 Resources

**Inspiration**:
- PyTorch: https://pytorch.org/ (API design)
- Burn (Rust): https://github.com/tracel-ai/burn (architecture)
- Gorgonia: https://github.com/gorgonia/gorgonia (Go ML)

**Documentation**:
- README.md - Quick start
- CONTRIBUTING.md - How to contribute
- docs/guides/ - User guides
- CHANGELOG.md - Release history

**Development**:
- Go 1.25+ required
- golangci-lint for quality
- CUDA Toolkit for GPU (v0.3.0+)

---

## 📞 Support

**Bug Reports**:
- GitHub Issues: https://github.com/born-ml/born/issues
- Security: See SECURITY.md

**Questions**:
- GitHub Discussions: https://github.com/born-ml/born/discussions
- Stack Overflow: Tag `born-ml`

**Contributing**:
- See CONTRIBUTING.md
- Good first issues labeled
- Code reviews welcome

---

## 🚦 Feature Request Process

1. **Open GitHub Issue** with feature proposal
2. **Community Discussion** (upvotes, comments)
3. **Roadmap Review** (quarterly)
4. **Prioritization** based on:
   - Community demand (upvotes)
   - Implementation complexity
   - Alignment with vision
   - Maintainability

5. **Assignment** to milestone
6. **Implementation** with tests + docs
7. **Release** with migration guide if needed

---

## 🔒 Stability Guarantees

### v0.x.x (Current)
- ⚠️ API may change between minor versions
- ⚠️ Deprecation warnings provided
- ✅ Migration guides for breaking changes
- ✅ Semantic versioning followed

### v1.x.x (Future LTS)
- ✅ API stability guaranteed
- ✅ No breaking changes in minor/patch
- ✅ 3+ years support
- ✅ Security updates
- ✅ Performance improvements (non-breaking)

---

## 🎯 Success Metrics

**Technical**:
- [ ] >70% test coverage
- [ ] <1ms tensor creation (CPU)
- [ ] <100ms MNIST training/epoch (GPU)
- [ ] Zero memory leaks (validated)
- [ ] 0 golangci-lint issues

**Community**:
- [ ] >1000 GitHub stars
- [ ] >10 contributors
- [ ] >5 production deployments
- [ ] >100 weekly downloads (pkg.go.dev)

**Documentation**:
- [ ] Complete API documentation
- [ ] 10+ tutorials
- [ ] 5+ example projects
- [ ] Migration guides for all versions

---

*Version 2.0 (2025-11-28)*
*Current: v0.2.0 (WebGPU GPU Backend) | Phase: GPU Acceleration | Next: v0.3.0 (Training Stability) | Target: v1.0.0 LTS (2027-2028)*
