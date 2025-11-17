package tensor

// Backend defines the interface that all compute backends must implement.
// Backends handle the actual computation for tensor operations.
//
// Implementations:
//   - CPU: Pure Go with SIMD optimizations (TASK-003)
//   - CUDA: NVIDIA GPU via driver API (Phase 2)
//   - Vulkan: Cross-platform GPU compute (Phase 2)
//   - Metal: Apple GPU (Phase 2)
//   - WebGPU: Browser/native GPU (Phase 2)
type Backend interface {
	// Element-wise binary operations
	Add(a, b *RawTensor) *RawTensor
	Sub(a, b *RawTensor) *RawTensor
	Mul(a, b *RawTensor) *RawTensor
	Div(a, b *RawTensor) *RawTensor

	// Matrix operations
	MatMul(a, b *RawTensor) *RawTensor

	// Convolutional operations
	Conv2D(input, kernel *RawTensor, stride, padding int) *RawTensor
	MaxPool2D(input *RawTensor, kernelSize, stride int) *RawTensor

	// Shape operations
	Reshape(t *RawTensor, newShape Shape) *RawTensor
	Transpose(t *RawTensor, axes ...int) *RawTensor

	// Metadata
	Name() string
	Device() Device
}
