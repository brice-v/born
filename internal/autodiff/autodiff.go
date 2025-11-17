// Package autodiff implements automatic differentiation using the decorator pattern.
//
// AutodiffBackend wraps any Backend implementation (CPU, GPU, etc.) and adds
// gradient tracking capabilities through a GradientTape.
//
// Architecture:
//   - Decorator pattern: AutodiffBackend[B] wraps any Backend implementation
//   - GradientTape: Records operations during forward pass
//   - Operation interface: Each op (Add, Mul, MatMul) implements backward pass
//   - Reverse-mode AD: Computes gradients efficiently using chain rule
//
// Usage:
//
//	// Wrap any backend with autodiff
//	cpuBackend := cpu.New()
//	autodiffBackend := autodiff.New(cpuBackend)
//
//	// Use with tensors
//	x := tensor.FromSlice([]float32{2.0}, tensor.Shape{1}, autodiffBackend)
//	y := x.Mul(x) // y = x²
//
//	// Compute gradients
//	y.Backward()
//	fmt.Println(x.Grad()) // dy/dx = 2x = 4.0
package autodiff

import (
	"math"

	"github.com/born-ml/born/internal/autodiff/ops"
	"github.com/born-ml/born/internal/tensor"
)

// AutodiffBackend wraps a Backend and adds automatic differentiation.
// It implements the tensor.Backend interface and records operations in a GradientTape.
//
// Type parameter B must satisfy the tensor.Backend interface.
type AutodiffBackend[B tensor.Backend] struct {
	inner B             // Wrapped backend (CPU, GPU, etc.)
	tape  *GradientTape // Records operations for backpropagation
}

// New creates a new AutodiffBackend wrapping the given backend.
func New[B tensor.Backend](backend B) *AutodiffBackend[B] {
	return &AutodiffBackend[B]{
		inner: backend,
		tape:  NewGradientTape(),
	}
}

// Tape returns the gradient tape for manual control.
// Useful for:
//   - Starting/stopping recording
//   - Clearing tape between iterations
//   - Inspecting recorded operations
func (b *AutodiffBackend[B]) Tape() *GradientTape {
	return b.tape
}

// Inner returns the wrapped backend for direct access.
func (b *AutodiffBackend[B]) Inner() B {
	return b.inner
}

// Name returns the backend name.
func (b *AutodiffBackend[B]) Name() string {
	return "Autodiff(" + b.inner.Name() + ")"
}

// Device returns the compute device.
func (b *AutodiffBackend[B]) Device() tensor.Device {
	return b.inner.Device()
}

// Add performs element-wise addition and records the operation.
func (b *AutodiffBackend[B]) Add(a, c *tensor.RawTensor) *tensor.RawTensor {
	// CRITICAL: Prevent inplace modification that would corrupt autodiff graph.
	// Temporarily increase refCount so IsUnique() returns false.
	// This forces CPU backend to allocate new result instead of inplace modification.
	defer a.ForceNonUnique()()
	defer c.ForceNonUnique()()

	// Forward pass using wrapped backend
	result := b.inner.Add(a, c)

	// Record operation if tape is recording
	if b.tape.IsRecording() {
		op := ops.NewAddOp(a, c, result)
		b.tape.Record(op)
	}

	return result
}

// Sub performs element-wise subtraction and records the operation.
func (b *AutodiffBackend[B]) Sub(a, c *tensor.RawTensor) *tensor.RawTensor {
	defer a.ForceNonUnique()()
	defer c.ForceNonUnique()()

	result := b.inner.Sub(a, c)

	if b.tape.IsRecording() {
		op := ops.NewSubOp(a, c, result)
		b.tape.Record(op)
	}

	return result
}

// Mul performs element-wise multiplication and records the operation.
func (b *AutodiffBackend[B]) Mul(a, c *tensor.RawTensor) *tensor.RawTensor {
	defer a.ForceNonUnique()()
	defer c.ForceNonUnique()()

	result := b.inner.Mul(a, c)

	if b.tape.IsRecording() {
		op := ops.NewMulOp(a, c, result)
		b.tape.Record(op)
	}

	return result
}

// Div performs element-wise division and records the operation.
func (b *AutodiffBackend[B]) Div(a, c *tensor.RawTensor) *tensor.RawTensor {
	defer a.ForceNonUnique()()
	defer c.ForceNonUnique()()

	result := b.inner.Div(a, c)

	if b.tape.IsRecording() {
		op := ops.NewDivOp(a, c, result)
		b.tape.Record(op)
	}

	return result
}

// MatMul performs matrix multiplication and records the operation.
func (b *AutodiffBackend[B]) MatMul(a, c *tensor.RawTensor) *tensor.RawTensor {
	defer a.ForceNonUnique()()
	defer c.ForceNonUnique()()

	result := b.inner.MatMul(a, c)

	if b.tape.IsRecording() {
		op := ops.NewMatMulOp(a, c, result)
		b.tape.Record(op)
	}

	return result
}

// Reshape reshapes a tensor and records the operation.
//
// CRITICAL: Like Transpose, Reshape must be recorded on tape!
// Without recording, gradients won't flow back to reshaped parameters.
//
// Example: Conv2D bias
//   - bias parameter: [out_channels]
//   - reshaped for broadcasting: [1, out_channels, 1, 1]
//   - Without ReshapeOp: gradient computed for reshaped tensor only
//   - With ReshapeOp: gradient propagates back to original bias parameter
func (b *AutodiffBackend[B]) Reshape(t *tensor.RawTensor, newShape tensor.Shape) *tensor.RawTensor {
	defer t.ForceNonUnique()()

	// Forward pass using wrapped backend
	result := b.inner.Reshape(t, newShape)

	// Record operation if tape is recording
	if b.tape.IsRecording() {
		op := ops.NewReshapeOp(t, result)
		b.tape.Record(op)
	}

	return result
}

// Transpose transposes a tensor and records the operation.
//
// CRITICAL: Even though conceptually transpose is a "view", the underlying
// backend may create a new tensor (e.g., CPU backend copies data).
// We MUST record this operation so gradients flow back correctly.
//
// For example, in Linear layer:
//
//	w = weight parameter
//	wT = w.Transpose()  // Creates NEW tensor!
//	output = input @ wT  // MatMul records operation with wT
//
// Without recording Transpose:
//   - Backward computes grad for wT (new tensor)
//   - Optimizer looks for grad of w (original parameter)
//   - NO GRADIENT FOUND! Parameters don't update!
//
// With TransposeOp:
//   - Backward computes grad for wT
//   - TransposeOp.Backward propagates grad back to w
//   - Optimizer finds grad for w ✓
func (b *AutodiffBackend[B]) Transpose(t *tensor.RawTensor, axes ...int) *tensor.RawTensor {
	defer t.ForceNonUnique()()

	// Handle default axes (reverse all dimensions)
	ndim := len(t.Shape())
	if len(axes) == 0 {
		axes = make([]int, ndim)
		for i := range axes {
			axes[i] = ndim - 1 - i
		}
	}

	// Forward pass using wrapped backend
	result := b.inner.Transpose(t, axes...)

	// Record operation if tape is recording
	if b.tape.IsRecording() {
		op := ops.NewTransposeOp(t, result, axes)
		b.tape.Record(op)
	}

	return result
}

// Conv2D performs 2D convolution and records the operation.
//
// CRITICAL: Conv2D must be recorded on tape for gradient flow!
// Just like Transpose, Conv2D creates new tensors and without recording,
// gradients won't flow back to the kernel/input parameters.
func (b *AutodiffBackend[B]) Conv2D(input, kernel *tensor.RawTensor, stride, padding int) *tensor.RawTensor {
	defer input.ForceNonUnique()()
	defer kernel.ForceNonUnique()()

	// Forward pass using wrapped backend
	result := b.inner.Conv2D(input, kernel, stride, padding)

	// Record operation if tape is recording
	if b.tape.IsRecording() {
		op := ops.NewConv2DOp(input, kernel, result, stride, padding)
		b.tape.Record(op)
	}

	return result
}

// MaxPool2D performs 2D max pooling and records the operation.
//
// CRITICAL: MaxPool2D must be recorded on tape for gradient flow!
// During backward pass, gradients only flow to positions that had max values.
// MaxPool2DOp stores max indices during forward pass for correct gradient routing.
func (b *AutodiffBackend[B]) MaxPool2D(input *tensor.RawTensor, kernelSize, stride int) *tensor.RawTensor {
	defer input.ForceNonUnique()()

	// Forward pass using wrapped backend
	result := b.inner.MaxPool2D(input, kernelSize, stride)

	// Record operation if tape is recording
	if b.tape.IsRecording() {
		op := ops.NewMaxPool2DOp(input, result, kernelSize, stride)
		b.tape.Record(op)
	}

	return result
}

// ReLU applies ReLU activation and records the operation.
func (b *AutodiffBackend[B]) ReLU(x *tensor.RawTensor) *tensor.RawTensor {
	// Forward pass: max(0, x)
	result, err := tensor.NewRaw(x.Shape(), x.DType(), b.Device())
	if err != nil {
		panic(err)
	}

	// Apply ReLU based on dtype
	switch x.DType() {
	case tensor.Float32:
		xData := x.AsFloat32()
		resData := result.AsFloat32()
		for i, val := range xData {
			if val > 0 {
				resData[i] = val
			} else {
				resData[i] = 0
			}
		}

	case tensor.Float64:
		xData := x.AsFloat64()
		resData := result.AsFloat64()
		for i, val := range xData {
			if val > 0 {
				resData[i] = val
			} else {
				resData[i] = 0
			}
		}
	}

	// Record operation if tape is recording
	if b.tape.IsRecording() {
		op := ops.NewReLUOp(x, result)
		b.tape.Record(op)
	}

	return result
}

// Sigmoid applies sigmoid activation: σ(x) = 1 / (1 + exp(-x)).
func (b *AutodiffBackend[B]) Sigmoid(x *tensor.RawTensor) *tensor.RawTensor {
	// Forward pass: σ(x) = 1 / (1 + exp(-x))
	result, err := tensor.NewRaw(x.Shape(), x.DType(), b.Device())
	if err != nil {
		panic(err)
	}

	// Apply Sigmoid based on dtype
	switch x.DType() {
	case tensor.Float32:
		xData := x.AsFloat32()
		resData := result.AsFloat32()
		for i, val := range xData {
			// σ(x) = 1 / (1 + exp(-x))
			resData[i] = float32(1.0 / (1.0 + math.Exp(float64(-val))))
		}

	case tensor.Float64:
		xData := x.AsFloat64()
		resData := result.AsFloat64()
		for i, val := range xData {
			// σ(x) = 1 / (1 + exp(-x))
			resData[i] = 1.0 / (1.0 + math.Exp(-val))
		}
	}

	// Record operation if tape is recording
	if b.tape.IsRecording() {
		op := ops.NewSigmoidOp(x, result)
		b.tape.Record(op)
	}

	return result
}

// Tanh applies hyperbolic tangent activation.
func (b *AutodiffBackend[B]) Tanh(x *tensor.RawTensor) *tensor.RawTensor {
	// Forward pass: tanh(x)
	result, err := tensor.NewRaw(x.Shape(), x.DType(), b.Device())
	if err != nil {
		panic(err)
	}

	// Apply Tanh based on dtype
	switch x.DType() {
	case tensor.Float32:
		xData := x.AsFloat32()
		resData := result.AsFloat32()
		for i, val := range xData {
			resData[i] = float32(math.Tanh(float64(val)))
		}

	case tensor.Float64:
		xData := x.AsFloat64()
		resData := result.AsFloat64()
		for i, val := range xData {
			resData[i] = math.Tanh(val)
		}
	}

	// Record operation if tape is recording
	if b.tape.IsRecording() {
		op := ops.NewTanhOp(x, result)
		b.tape.Record(op)
	}

	return result
}

// Log computes element-wise natural logarithm.
//
// Forward:
//
//	output = log(input)
//
// Backward:
//
//	∂L/∂input = ∂L/∂output * (1 / input)
//
// Note: Input values must be positive. For numerical stability with values
// close to zero, consider using LogWithEpsilon operation instead.
func (b *AutodiffBackend[B]) Log(x *tensor.RawTensor) *tensor.RawTensor {
	defer x.ForceNonUnique()()

	// Forward pass: log(x)
	result, err := tensor.NewRaw(x.Shape(), x.DType(), b.Device())
	if err != nil {
		panic(err)
	}

	// Apply Log based on dtype
	switch x.DType() {
	case tensor.Float32:
		xData := x.AsFloat32()
		resData := result.AsFloat32()
		for i, val := range xData {
			resData[i] = float32(math.Log(float64(val)))
		}

	case tensor.Float64:
		xData := x.AsFloat64()
		resData := result.AsFloat64()
		for i, val := range xData {
			resData[i] = math.Log(val)
		}

	default:
		panic("Log: only supports float32 and float64")
	}

	// Record operation if tape is recording
	if b.tape.IsRecording() {
		op := ops.NewLogOp(x, result)
		b.tape.Record(op)
	}

	return result
}

// Softmax applies softmax activation along the last dimension.
//
// Forward (for each row):
//
//	softmax(x)_i = exp(x_i - max(x)) / Σ_j exp(x_j - max(x))
//
// The max-shifting ensures numerical stability (prevents overflow).
//
// Backward:
//
//	The Jacobian of softmax is complex, but the gradient simplifies to:
//	∂L/∂x_j = softmax_j * (∂L/∂softmax_j - Σ_i (∂L/∂softmax_i * softmax_i))
//
// Currently supports 2D tensors [batch_size, num_classes].
func (b *AutodiffBackend[B]) Softmax(x *tensor.RawTensor) *tensor.RawTensor {
	defer x.ForceNonUnique()()

	// Use helper function from ops package for forward pass
	result := ops.Softmax(x, b.Device())

	// Record operation if tape is recording
	if b.tape.IsRecording() {
		op := ops.NewSoftmaxOp(x, result)
		b.tape.Record(op)
	}

	return result
}

// CrossEntropy computes cross-entropy loss for classification.
//
// Forward:
//
//	Loss = mean(-log_softmax(logits)[targets])
//
// Uses the log-sum-exp trick for numerical stability.
//
// Backward:
//
//	∂L/∂logits = (softmax(logits) - y_one_hot) / batch_size
//
// Parameters:
//   - logits: Model predictions [batch_size, num_classes]
//   - targets: Ground truth class indices [batch_size]
//
// Returns:
//   - Scalar loss value (mean over batch)
func (b *AutodiffBackend[B]) CrossEntropy(logits, targets *tensor.RawTensor) *tensor.RawTensor {
	defer logits.ForceNonUnique()()
	// Note: targets doesn't need ForceNonUnique() as it's not differentiated

	// Forward pass using helper function
	result := ops.CrossEntropyForward(logits, targets, b.Device())

	// Record operation if tape is recording
	if b.tape.IsRecording() {
		op := ops.NewCrossEntropyOp(logits, targets, result)
		b.tape.Record(op)
	}

	return result
}
