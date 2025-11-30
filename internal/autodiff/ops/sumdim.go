package ops

import "github.com/born-ml/born/internal/tensor"

// SumDimOp represents a reduction sum operation along a dimension: output = sum(x, dim).
//
// Forward:
//
//	y = sum(x, dim, keepDim)
//
// Backward:
//
//	grad_x = broadcast(grad_y, x.shape)
//
// If keepDim=false, we need to unsqueeze grad_y first to match broadcasting requirements.
type SumDimOp struct {
	inputs  []*tensor.RawTensor // [x]
	output  *tensor.RawTensor   // sum(x, dim)
	dim     int                 // dimension to reduce
	keepDim bool                // whether to keep dimension
}

// NewSumDimOp creates a new SumDimOp.
func NewSumDimOp(x, output *tensor.RawTensor, dim int, keepDim bool) *SumDimOp {
	return &SumDimOp{
		inputs:  []*tensor.RawTensor{x},
		output:  output,
		dim:     dim,
		keepDim: keepDim,
	}
}

// Backward computes input gradients for sum reduction.
//
// The gradient flows by broadcasting grad_output to match input shape.
// Since sum just accumulates values, each input element contributes 1.0 to the output,
// so the gradient is simply broadcast back.
func (op *SumDimOp) Backward(outputGrad *tensor.RawTensor, backend tensor.Backend) []*tensor.RawTensor {
	x := op.inputs[0]
	grad := outputGrad

	// If keepDim=false, we need to unsqueeze the gradient first
	if !op.keepDim {
		grad = unsqueezeDim(grad, op.dim, x.Shape())
	}

	// Broadcast gradient to input shape
	gradX := broadcastTo(grad, x.Shape(), backend)

	return []*tensor.RawTensor{gradX}
}

// Inputs returns the input tensors [x].
func (op *SumDimOp) Inputs() []*tensor.RawTensor {
	return op.inputs
}

// Output returns the output tensor sum(x, dim).
func (op *SumDimOp) Output() *tensor.RawTensor {
	return op.output
}

// unsqueezeDim adds a dimension of size 1 at the specified position.
func unsqueezeDim(t *tensor.RawTensor, dim int, targetShape tensor.Shape) *tensor.RawTensor {
	// Normalize negative dim
	ndim := len(targetShape)
	if dim < 0 {
		dim = ndim + dim
	}

	// Create new shape with dimension inserted
	newShape := make(tensor.Shape, 0, len(t.Shape())+1)
	for i := 0; i < ndim; i++ {
		if i == dim {
			newShape = append(newShape, 1)
		} else if len(newShape) < len(t.Shape())+1 {
			// Map from original shape
			origIdx := i
			if i > dim {
				origIdx = i - 1
			}
			if origIdx < len(t.Shape()) {
				newShape = append(newShape, t.Shape()[origIdx])
			}
		}
	}

	// Handle case where we need to append the dimension
	if len(newShape) < len(t.Shape())+1 {
		newShape = append(newShape, t.Shape()[len(newShape):]...)
	}

	// Reshape tensor
	result, err := tensor.NewRaw(newShape, t.DType(), t.Device())
	if err != nil {
		panic("unsqueezeDim: failed to create result tensor")
	}

	// Copy data (shape is different but data is same)
	switch t.DType() {
	case tensor.Float32:
		copy(result.AsFloat32(), t.AsFloat32())
	case tensor.Float64:
		copy(result.AsFloat64(), t.AsFloat64())
	}

	return result
}

// broadcastTo broadcasts a tensor to match target shape.
func broadcastTo(t *tensor.RawTensor, targetShape tensor.Shape, _ tensor.Backend) *tensor.RawTensor {
	// If shapes already match, return clone
	if t.Shape().Equal(targetShape) {
		return t.Clone()
	}

	// Create result tensor
	result, err := tensor.NewRaw(targetShape, t.DType(), t.Device())
	if err != nil {
		panic("broadcastTo: failed to create result tensor")
	}

	// Broadcast data
	switch t.DType() {
	case tensor.Float32:
		broadcastFloat32(t.AsFloat32(), result.AsFloat32(), t.Shape(), targetShape)
	case tensor.Float64:
		broadcastFloat64(t.AsFloat64(), result.AsFloat64(), t.Shape(), targetShape)
	}

	return result
}

// broadcastFloat32 broadcasts float32 data to target shape.
func broadcastFloat32(src, dst []float32, srcShape, dstShape tensor.Shape) {
	srcStrides := srcShape.ComputeStrides()
	dstStrides := dstShape.ComputeStrides()
	numElements := dstShape.NumElements()

	for i := 0; i < numElements; i++ {
		// Compute destination coordinates
		srcIdx := 0
		temp := i
		for d := 0; d < len(dstShape); d++ {
			coord := temp / dstStrides[d]
			temp %= dstStrides[d]

			// Map to source dimension
			srcDim := d - (len(dstShape) - len(srcShape))
			if srcDim >= 0 && srcDim < len(srcShape) {
				// If source dimension is 1, always use coordinate 0
				if srcShape[srcDim] == 1 {
					coord = 0
				}
				srcIdx += coord * srcStrides[srcDim]
			}
		}

		dst[i] = src[srcIdx]
	}
}

// broadcastFloat64 broadcasts float64 data to target shape.
func broadcastFloat64(src, dst []float64, srcShape, dstShape tensor.Shape) {
	srcStrides := srcShape.ComputeStrides()
	dstStrides := dstShape.ComputeStrides()
	numElements := dstShape.NumElements()

	for i := 0; i < numElements; i++ {
		// Compute destination coordinates
		srcIdx := 0
		temp := i
		for d := 0; d < len(dstShape); d++ {
			coord := temp / dstStrides[d]
			temp %= dstStrides[d]

			// Map to source dimension
			srcDim := d - (len(dstShape) - len(srcShape))
			if srcDim >= 0 && srcDim < len(srcShape) {
				// If source dimension is 1, always use coordinate 0
				if srcShape[srcDim] == 1 {
					coord = 0
				}
				srcIdx += coord * srcStrides[srcDim]
			}
		}

		dst[i] = src[srcIdx]
	}
}
