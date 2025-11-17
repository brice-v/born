// Package tensor provides the core tensor types and operations for Born ML framework.
package tensor

import "fmt"

// Verify that MockBackend implements Backend.
var _ Backend = (*MockBackend)(nil)

// MockBackend is a simple backend for testing.
// It implements all operations naively for correctness verification.
type MockBackend struct{}

// NewMockBackend creates a new MockBackend.
func NewMockBackend() *MockBackend {
	return &MockBackend{}
}

// Name returns the backend name.
func (m *MockBackend) Name() string {
	return "mock"
}

// Device returns the device type.
func (m *MockBackend) Device() Device {
	return CPU
}

// Add performs element-wise addition with broadcasting.
func (m *MockBackend) Add(a, b *RawTensor) *RawTensor {
	return m.elementWise(a, b, func(x, y float64) float64 { return x + y })
}

// Sub performs element-wise subtraction with broadcasting.
func (m *MockBackend) Sub(a, b *RawTensor) *RawTensor {
	return m.elementWise(a, b, func(x, y float64) float64 { return x - y })
}

// Mul performs element-wise multiplication with broadcasting.
func (m *MockBackend) Mul(a, b *RawTensor) *RawTensor {
	return m.elementWise(a, b, func(x, y float64) float64 { return x * y })
}

// Div performs element-wise division with broadcasting.
func (m *MockBackend) Div(a, b *RawTensor) *RawTensor {
	return m.elementWise(a, b, func(x, y float64) float64 { return x / y })
}

// elementWise performs element-wise operations with broadcasting.
func (m *MockBackend) elementWise(a, b *RawTensor, op func(float64, float64) float64) *RawTensor {
	// Broadcast shapes
	outShape, _, err := BroadcastShapes(a.Shape(), b.Shape())
	if err != nil {
		panic(err)
	}

	// Create output tensor
	result, err := NewRaw(outShape, a.DType(), m.Device())
	if err != nil {
		panic(err)
	}

	// Perform operation (naive implementation)
	numElements := outShape.NumElements()

	// Convert to float64 for generic processing
	aData := m.toFloat64Slice(a)
	bData := m.toFloat64Slice(b)
	resultData := m.toFloat64Slice(result)

	for i := 0; i < numElements; i++ {
		aIdx := m.broadcastIndex(i, outShape, a.Shape())
		bIdx := m.broadcastIndex(i, outShape, b.Shape())

		resultData[i] = op(aData[aIdx], bData[bIdx])
	}

	m.fromFloat64Slice(resultData, result)
	return result
}

// MatMul performs matrix multiplication.
func (m *MockBackend) MatMul(a, b *RawTensor) *RawTensor {
	aShape := a.Shape()
	bShape := b.Shape()

	// Only support 2D for now
	if len(aShape) != 2 || len(bShape) != 2 {
		panic("MatMul only supports 2D tensors in mock backend")
	}

	if aShape[1] != bShape[0] {
		panic(fmt.Sprintf("incompatible shapes for MatMul: %v @ %v", aShape, bShape))
	}

	M, K := aShape[0], aShape[1]
	N := bShape[1]

	outShape := Shape{M, N}
	result, err := NewRaw(outShape, a.DType(), m.Device())
	if err != nil {
		panic(err)
	}

	aData := m.toFloat64Slice(a)
	bData := m.toFloat64Slice(b)
	resultData := m.toFloat64Slice(result)

	// Naive matrix multiplication
	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			sum := 0.0
			for k := 0; k < K; k++ {
				sum += aData[i*K+k] * bData[k*N+j]
			}
			resultData[i*N+j] = sum
		}
	}

	m.fromFloat64Slice(resultData, result)
	return result
}

// Conv2D performs 2D convolution (naive implementation for testing).
func (m *MockBackend) Conv2D(input, kernel *RawTensor, stride, padding int) *RawTensor {
	inputShape := input.Shape()
	kernelShape := kernel.Shape()

	if len(inputShape) != 4 || len(kernelShape) != 4 {
		panic("Conv2D requires 4D tensors [N,C,H,W]")
	}

	N := inputShape[0]
	CIn := inputShape[1]
	H := inputShape[2]
	W := inputShape[3]
	COut := kernelShape[0]
	KH := kernelShape[2]
	KW := kernelShape[3]

	if CIn != kernelShape[1] {
		panic(fmt.Sprintf("Conv2D: input channels %d != kernel channels %d", CIn, kernelShape[1]))
	}

	HOut := (H+2*padding-KH)/stride + 1
	WOut := (W+2*padding-KW)/stride + 1

	output, err := NewRaw(Shape{N, COut, HOut, WOut}, input.DType(), m.Device())
	if err != nil {
		panic(err)
	}

	inputData := m.toFloat64Slice(input)
	kernelData := m.toFloat64Slice(kernel)
	outputData := m.toFloat64Slice(output)

	// Naive convolution (direct implementation)
	for n := 0; n < N; n++ {
		for cOut := 0; cOut < COut; cOut++ {
			for outH := 0; outH < HOut; outH++ {
				for outW := 0; outW < WOut; outW++ {
					sum := 0.0

					// Convolve over input patch
					for cIn := 0; cIn < CIn; cIn++ {
						for kh := 0; kh < KH; kh++ {
							for kw := 0; kw < KW; kw++ {
								h := outH*stride - padding + kh
								w := outW*stride - padding + kw

								// Check bounds (zero padding)
								if h >= 0 && h < H && w >= 0 && w < W {
									inputIdx := n*CIn*H*W + cIn*H*W + h*W + w
									kernelIdx := cOut*CIn*KH*KW + cIn*KH*KW + kh*KW + kw
									sum += inputData[inputIdx] * kernelData[kernelIdx]
								}
							}
						}
					}

					outputIdx := n*COut*HOut*WOut + cOut*HOut*WOut + outH*WOut + outW
					outputData[outputIdx] = sum
				}
			}
		}
	}

	m.fromFloat64Slice(outputData, output)
	return output
}

// MaxPool2D performs 2D max pooling (naive implementation for testing).
func (m *MockBackend) MaxPool2D(input *RawTensor, kernelSize, stride int) *RawTensor {
	inputShape := input.Shape()
	if len(inputShape) != 4 {
		panic(fmt.Sprintf("MaxPool2D: expected 4D input [N,C,H,W], got %dD", len(inputShape)))
	}

	N := inputShape[0]
	C := inputShape[1]
	H := inputShape[2]
	W := inputShape[3]

	// Compute output dimensions
	HOut := (H-kernelSize)/stride + 1
	WOut := (W-kernelSize)/stride + 1

	output, err := NewRaw(Shape{N, C, HOut, WOut}, input.DType(), m.Device())
	if err != nil {
		panic(err)
	}

	inputData := m.toFloat64Slice(input)
	outputData := m.toFloat64Slice(output)

	// Naive max pooling
	for n := 0; n < N; n++ {
		for c := 0; c < C; c++ {
			for outH := 0; outH < HOut; outH++ {
				for outW := 0; outW < WOut; outW++ {
					hStart := outH * stride
					wStart := outW * stride

					// Find max in pooling window
					maxVal := -1e308 // Negative infinity
					for kh := 0; kh < kernelSize; kh++ {
						for kw := 0; kw < kernelSize; kw++ {
							h := hStart + kh
							w := wStart + kw
							inputIdx := n*C*H*W + c*H*W + h*W + w
							if inputData[inputIdx] > maxVal {
								maxVal = inputData[inputIdx]
							}
						}
					}

					outputIdx := n*C*HOut*WOut + c*HOut*WOut + outH*WOut + outW
					outputData[outputIdx] = maxVal
				}
			}
		}
	}

	m.fromFloat64Slice(outputData, output)
	return output
}

// Reshape changes tensor shape.
func (m *MockBackend) Reshape(t *RawTensor, newShape Shape) *RawTensor {
	if err := newShape.Validate(); err != nil {
		panic(err)
	}

	if t.NumElements() != newShape.NumElements() {
		panic(fmt.Sprintf("cannot reshape tensor with %d elements to shape %v (%d elements)",
			t.NumElements(), newShape, newShape.NumElements()))
	}

	result, err := NewRaw(newShape, t.DType(), m.Device())
	if err != nil {
		panic(err)
	}

	// Copy data
	copy(result.Data(), t.Data())
	return result
}

// Transpose transposes tensor dimensions.
func (m *MockBackend) Transpose(t *RawTensor, axes ...int) *RawTensor {
	shape := t.Shape()

	// Default: reverse all dimensions
	if len(axes) == 0 {
		axes = make([]int, len(shape))
		for i := range axes {
			axes[i] = len(shape) - 1 - i
		}
	}

	// Validate axes
	if len(axes) != len(shape) {
		panic(fmt.Sprintf("axes length %d doesn't match tensor dimensions %d", len(axes), len(shape)))
	}

	// Compute new shape
	newShape := make(Shape, len(shape))
	for i, axis := range axes {
		if axis < 0 || axis >= len(shape) {
			panic(fmt.Sprintf("axis %d out of bounds for tensor with %d dimensions", axis, len(shape)))
		}
		newShape[i] = shape[axis]
	}

	result, err := NewRaw(newShape, t.DType(), m.Device())
	if err != nil {
		panic(err)
	}

	// Transpose data (naive implementation)
	tData := m.toFloat64Slice(t)
	resultData := m.toFloat64Slice(result)

	oldStrides := shape.ComputeStrides()
	newStrides := newShape.ComputeStrides()

	for i := 0; i < t.NumElements(); i++ {
		// Convert flat index to multi-dimensional indices
		indices := make([]int, len(shape))
		temp := i
		for j := 0; j < len(shape); j++ {
			indices[j] = temp / oldStrides[j]
			temp %= oldStrides[j]
		}

		// Permute indices
		permuted := make([]int, len(indices))
		for j, axis := range axes {
			permuted[j] = indices[axis]
		}

		// Convert permuted indices to flat index
		newIdx := 0
		for j, idx := range permuted {
			newIdx += idx * newStrides[j]
		}

		resultData[newIdx] = tData[i]
	}

	m.fromFloat64Slice(resultData, result)
	return result
}

// Helper functions

func (m *MockBackend) toFloat64Slice(t *RawTensor) []float64 {
	switch t.DType() {
	case Float32:
		src := t.AsFloat32()
		dst := make([]float64, len(src))
		for i, v := range src {
			dst[i] = float64(v)
		}
		return dst
	case Float64:
		return t.AsFloat64()
	case Int32:
		src := t.AsInt32()
		dst := make([]float64, len(src))
		for i, v := range src {
			dst[i] = float64(v)
		}
		return dst
	case Int64:
		src := t.AsInt64()
		dst := make([]float64, len(src))
		for i, v := range src {
			dst[i] = float64(v)
		}
		return dst
	default:
		panic(fmt.Sprintf("unsupported dtype: %s", t.DType()))
	}
}

func (m *MockBackend) fromFloat64Slice(src []float64, t *RawTensor) {
	switch t.DType() {
	case Float32:
		dst := t.AsFloat32()
		for i, v := range src {
			dst[i] = float32(v)
		}
	case Float64:
		copy(t.AsFloat64(), src)
	case Int32:
		dst := t.AsInt32()
		for i, v := range src {
			dst[i] = int32(v)
		}
	case Int64:
		dst := t.AsInt64()
		for i, v := range src {
			dst[i] = int64(v)
		}
	}
}

func (m *MockBackend) broadcastIndex(flatIdx int, outShape, inShape Shape) int {
	// Convert flat index to multi-dimensional indices in output shape
	outStrides := outShape.ComputeStrides()
	indices := make([]int, len(outShape))

	temp := flatIdx
	for i := 0; i < len(outShape); i++ {
		indices[i] = temp / outStrides[i]
		temp %= outStrides[i]
	}

	// Map to input shape (accounting for broadcasting)
	inStrides := inShape.ComputeStrides()
	inIdx := 0

	offset := len(outShape) - len(inShape)
	for i := 0; i < len(inShape); i++ {
		outDimIdx := indices[offset+i]
		inDim := inShape[i]

		// If input dimension is 1, always use index 0 (broadcasting)
		if inDim == 1 {
			outDimIdx = 0
		}

		inIdx += outDimIdx * inStrides[i]
	}

	return inIdx
}
