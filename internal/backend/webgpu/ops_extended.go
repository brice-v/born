//go:build windows

package webgpu

import (
	"github.com/born-ml/born/internal/tensor"
)

// Extended operations - GPU implementations using WGSL shaders.

// Scalar operations - use runScalarOp for GPU execution.

// MulScalar multiplies tensor elements by a scalar on GPU.
func (b *Backend) MulScalar(x *tensor.RawTensor, scalar any) *tensor.RawTensor {
	s := toFloat32(scalar)
	result, err := b.runScalarOp(x, s, "scalarMul", scalarMulShader)
	if err != nil {
		panic("webgpu: MulScalar: " + err.Error())
	}
	return result
}

// AddScalar adds a scalar to tensor elements on GPU.
func (b *Backend) AddScalar(x *tensor.RawTensor, scalar any) *tensor.RawTensor {
	s := toFloat32(scalar)
	result, err := b.runScalarOp(x, s, "scalarAdd", scalarAddShader)
	if err != nil {
		panic("webgpu: AddScalar: " + err.Error())
	}
	return result
}

// SubScalar subtracts a scalar from tensor elements on GPU.
func (b *Backend) SubScalar(x *tensor.RawTensor, scalar any) *tensor.RawTensor {
	s := toFloat32(scalar)
	result, err := b.runScalarOp(x, -s, "scalarAdd", scalarAddShader) // x - s = x + (-s)
	if err != nil {
		panic("webgpu: SubScalar: " + err.Error())
	}
	return result
}

// DivScalar divides tensor elements by a scalar on GPU.
func (b *Backend) DivScalar(x *tensor.RawTensor, scalar any) *tensor.RawTensor {
	s := toFloat32(scalar)
	if s == 0 {
		panic("webgpu: DivScalar: division by zero")
	}
	result, err := b.runScalarOp(x, 1.0/s, "scalarMul", scalarMulShader) // x / s = x * (1/s)
	if err != nil {
		panic("webgpu: DivScalar: " + err.Error())
	}
	return result
}

// toFloat32 converts any numeric type to float32.
func toFloat32(v any) float32 {
	switch val := v.(type) {
	case float32:
		return val
	case float64:
		return float32(val)
	case int:
		return float32(val)
	case int32:
		return float32(val)
	case int64:
		return float32(val)
	default:
		panic("webgpu: unsupported scalar type")
	}
}

// Math operations.

// Log computes natural logarithm element-wise on GPU.
func (b *Backend) Log(x *tensor.RawTensor) *tensor.RawTensor {
	result, err := b.runUnaryOp(x, "log", logShader)
	if err != nil {
		panic("webgpu: Log: " + err.Error())
	}
	return result
}

// Activation functions.

// Softmax applies softmax along the specified dimension.
// Supports N-dimensional tensors with dim=-1 (last dimension).
func (b *Backend) Softmax(x *tensor.RawTensor, dim int) *tensor.RawTensor {
	shape := x.Shape()
	ndim := len(shape)

	// Normalize negative dim
	if dim < 0 {
		dim = ndim + dim
	}

	// Only support softmax on last dimension
	if dim != ndim-1 {
		panic("webgpu: Softmax currently only supports dim=-1 (last dimension)")
	}

	// For 2D tensors, use GPU softmax directly
	if ndim == 2 {
		result, err := b.runSoftmax(x)
		if err != nil {
			panic("webgpu: Softmax: " + err.Error())
		}
		return result
	}

	// For N-D tensors: flatten → 2D softmax → reshape back
	// [d0, d1, ..., d_{n-2}, d_{n-1}] → [d0*d1*...*d_{n-2}, d_{n-1}]
	lastDim := shape[ndim-1]
	batchSize := 1
	for i := 0; i < ndim-1; i++ {
		batchSize *= shape[i]
	}

	// Reshape to 2D
	flat := b.Reshape(x, tensor.Shape{batchSize, lastDim})

	// Apply 2D softmax
	result2D, err := b.runSoftmax(flat)
	if err != nil {
		panic("webgpu: Softmax: " + err.Error())
	}

	// Reshape back to original shape
	return b.Reshape(result2D, shape)
}

// Comparison operations.

// Greater performs element-wise greater-than comparison on GPU.
// Always returns float32 tensor (0.0 for false, 1.0 for true).
func (b *Backend) Greater(a, other *tensor.RawTensor) *tensor.RawTensor {
	result, err := b.runComparisonOp(a, other, "greater", greaterShader)
	if err != nil {
		panic("webgpu: Greater: " + err.Error())
	}
	return result
}

// Lower performs element-wise less-than comparison on GPU.
// Always returns float32 tensor (0.0 for false, 1.0 for true).
func (b *Backend) Lower(a, other *tensor.RawTensor) *tensor.RawTensor {
	result, err := b.runComparisonOp(a, other, "lower", lowerShader)
	if err != nil {
		panic("webgpu: Lower: " + err.Error())
	}
	return result
}

// GreaterEqual performs element-wise greater-or-equal comparison on GPU.
// Always returns float32 tensor (0.0 for false, 1.0 for true).
func (b *Backend) GreaterEqual(a, other *tensor.RawTensor) *tensor.RawTensor {
	result, err := b.runComparisonOp(a, other, "greaterEqual", greaterEqualShader)
	if err != nil {
		panic("webgpu: GreaterEqual: " + err.Error())
	}
	return result
}

// LowerEqual performs element-wise less-or-equal comparison on GPU.
// Always returns float32 tensor (0.0 for false, 1.0 for true).
func (b *Backend) LowerEqual(a, other *tensor.RawTensor) *tensor.RawTensor {
	result, err := b.runComparisonOp(a, other, "lowerEqual", lowerEqualShader)
	if err != nil {
		panic("webgpu: LowerEqual: " + err.Error())
	}
	return result
}

// Equal performs element-wise equality comparison on GPU.
// Always returns float32 tensor (0.0 for false, 1.0 for true).
func (b *Backend) Equal(a, other *tensor.RawTensor) *tensor.RawTensor {
	result, err := b.runComparisonOp(a, other, "equal", equalShader)
	if err != nil {
		panic("webgpu: Equal: " + err.Error())
	}
	return result
}

// NotEqual performs element-wise inequality comparison on GPU.
// Always returns float32 tensor (0.0 for false, 1.0 for true).
func (b *Backend) NotEqual(a, other *tensor.RawTensor) *tensor.RawTensor {
	result, err := b.runComparisonOp(a, other, "notEqual", notEqualShader)
	if err != nil {
		panic("webgpu: NotEqual: " + err.Error())
	}
	return result
}

// Boolean operations.

// Or performs element-wise logical OR on GPU.
// Supports mixed dtypes by casting to float32 (for boolean tensors from different sources).
func (b *Backend) Or(a, other *tensor.RawTensor) *tensor.RawTensor {
	// Cast to float32 if dtypes differ (common for boolean results from different tensor types)
	aFloat := a
	otherFloat := other
	if a.DType() != tensor.Float32 {
		aFloat = b.Cast(a, tensor.Float32)
	}
	if other.DType() != tensor.Float32 {
		otherFloat = b.Cast(other, tensor.Float32)
	}

	result, err := b.runBinaryOp(aFloat, otherFloat, "or", orShader)
	if err != nil {
		panic("webgpu: Or: " + err.Error())
	}
	return result
}

// And performs element-wise logical AND on GPU.
// Supports mixed dtypes by casting to float32 (for boolean tensors from different sources).
func (b *Backend) And(a, other *tensor.RawTensor) *tensor.RawTensor {
	// Cast to float32 if dtypes differ (common for boolean results from different tensor types)
	aFloat := a
	otherFloat := other
	if a.DType() != tensor.Float32 {
		aFloat = b.Cast(a, tensor.Float32)
	}
	if other.DType() != tensor.Float32 {
		otherFloat = b.Cast(other, tensor.Float32)
	}

	result, err := b.runBinaryOp(aFloat, otherFloat, "and", andShader)
	if err != nil {
		panic("webgpu: And: " + err.Error())
	}
	return result
}

// Not performs element-wise logical NOT on GPU.
func (b *Backend) Not(x *tensor.RawTensor) *tensor.RawTensor {
	result, err := b.runUnaryOp(x, "not", notShader)
	if err != nil {
		panic("webgpu: Not: " + err.Error())
	}
	return result
}

// Reduction operations.

// Sum computes the sum of all elements on GPU.
func (b *Backend) Sum(x *tensor.RawTensor) *tensor.RawTensor {
	result, err := b.runSum(x)
	if err != nil {
		panic("webgpu: Sum: " + err.Error())
	}
	return result
}

// Argmax returns indices of maximum values along dimension on GPU.
func (b *Backend) Argmax(x *tensor.RawTensor, dim int) *tensor.RawTensor {
	result, err := b.runArgmax(x, dim)
	if err != nil {
		panic("webgpu: Argmax: " + err.Error())
	}
	return result
}

// Shape operations.

// Expand broadcasts tensor to new shape.
func (b *Backend) Expand(x *tensor.RawTensor, newShape tensor.Shape) *tensor.RawTensor {
	// Expand is a broadcasting operation - implemented on CPU for simplicity
	shape := x.Shape()

	// Validate shapes are compatible for broadcasting
	if len(newShape) < len(shape) {
		panic("webgpu: Expand: new shape must have at least as many dimensions")
	}

	// Create result tensor
	result, err := tensor.NewRaw(newShape, x.DType(), tensor.WebGPU)
	if err != nil {
		panic("webgpu: Expand: " + err.Error())
	}

	switch x.DType() {
	case tensor.Float32:
		expandFloat32(x.AsFloat32(), result.AsFloat32(), shape, newShape)
	case tensor.Int32:
		expandInt32(x.AsInt32(), result.AsInt32(), shape, newShape)
	default:
		panic("webgpu: Expand: unsupported dtype " + x.DType().String())
	}

	return result
}

// expandFloat32 broadcasts data from source shape to target shape.
func expandFloat32(src, dst []float32, srcShape, dstShape tensor.Shape) {
	expandGeneric(src, dst, srcShape, dstShape)
}

// expandInt32 broadcasts int32 data from source shape to target shape.
func expandInt32(src, dst []int32, srcShape, dstShape tensor.Shape) {
	expandGeneric(src, dst, srcShape, dstShape)
}

// expandGeneric broadcasts data from source shape to target shape (generic version).
func expandGeneric[T any](src, dst []T, srcShape, dstShape tensor.Shape) {
	// Calculate strides
	srcStrides := srcShape.ComputeStrides()
	dstStrides := dstShape.ComputeStrides()

	// Pad source shape to match destination dimensions
	dimDiff := len(dstShape) - len(srcShape)
	paddedSrcShape := make(tensor.Shape, len(dstShape))
	paddedSrcStrides := make([]int, len(dstShape))
	for i := 0; i < dimDiff; i++ {
		paddedSrcShape[i] = 1
		paddedSrcStrides[i] = 0
	}
	for i := 0; i < len(srcShape); i++ {
		paddedSrcShape[dimDiff+i] = srcShape[i]
		paddedSrcStrides[dimDiff+i] = srcStrides[i]
	}

	numElements := dstShape.NumElements()
	for i := 0; i < numElements; i++ {
		// Calculate destination coordinates
		temp := i
		srcIdx := 0
		for d := 0; d < len(dstShape); d++ {
			coord := temp / dstStrides[d]
			temp %= dstStrides[d]

			// Map to source index (with broadcasting)
			srcCoord := coord
			if paddedSrcShape[d] == 1 {
				srcCoord = 0
			}
			srcIdx += srcCoord * paddedSrcStrides[d]
		}
		dst[i] = src[srcIdx]
	}
}

// Type conversion.

// Cast converts tensor to different data type.
// Supports float32 and int32 as target types.
func (b *Backend) Cast(x *tensor.RawTensor, dtype tensor.DataType) *tensor.RawTensor {
	if x.DType() == dtype {
		// No conversion needed - just copy
		result, err := tensor.NewRaw(x.Shape(), dtype, tensor.WebGPU)
		if err != nil {
			panic("webgpu: Cast: " + err.Error())
		}
		copy(result.Data(), x.Data())
		return result
	}

	result, err := tensor.NewRaw(x.Shape(), dtype, tensor.WebGPU)
	if err != nil {
		panic("webgpu: Cast: " + err.Error())
	}

	// Route by target dtype
	switch dtype {
	case tensor.Float32:
		b.castToFloat32(x, result)
	case tensor.Int32:
		b.castToInt32(x, result)
	default:
		panic("webgpu: Cast: unsupported target type " + dtype.String())
	}

	return result
}

// castToFloat32 converts any supported dtype to float32.
func (b *Backend) castToFloat32(x, result *tensor.RawTensor) {
	dst := result.AsFloat32()
	switch x.DType() {
	case tensor.Float64:
		src := x.AsFloat64()
		for i, v := range src {
			dst[i] = float32(v)
		}
	case tensor.Int32:
		src := x.AsInt32()
		for i, v := range src {
			dst[i] = float32(v)
		}
	case tensor.Int64:
		src := x.AsInt64()
		for i, v := range src {
			dst[i] = float32(v)
		}
	default:
		panic("webgpu: Cast: unsupported source type for float32: " + x.DType().String())
	}
}

// castToInt32 converts any supported dtype to int32.
func (b *Backend) castToInt32(x, result *tensor.RawTensor) {
	dst := result.AsInt32()
	switch x.DType() {
	case tensor.Float32:
		src := x.AsFloat32()
		for i, v := range src {
			dst[i] = int32(v)
		}
	case tensor.Float64:
		src := x.AsFloat64()
		for i, v := range src {
			dst[i] = int32(v)
		}
	case tensor.Int64:
		src := x.AsInt64()
		for i, v := range src {
			dst[i] = int32(v) //nolint:gosec // ML tensors typically use small values; overflow is acceptable
		}
	default:
		panic("webgpu: Cast: unsupported source type for int32: " + x.DType().String())
	}
}
