package ops

import "github.com/born-ml/born/internal/tensor"

// MeanDimOp represents a reduction mean operation along a dimension: output = mean(x, dim).
//
// Forward:
//
//	y = mean(x, dim, keepDim) = sum(x, dim, keepDim) / size[dim]
//
// Backward:
//
//	grad_x = broadcast(grad_y, x.shape) / size[dim]
//
// If keepDim=false, we need to unsqueeze grad_y first to match broadcasting requirements.
type MeanDimOp struct {
	inputs  []*tensor.RawTensor // [x]
	output  *tensor.RawTensor   // mean(x, dim)
	dim     int                 // dimension to reduce
	keepDim bool                // whether to keep dimension
	dimSize int                 // size of reduced dimension (for backward pass)
}

// NewMeanDimOp creates a new MeanDimOp.
func NewMeanDimOp(x, output *tensor.RawTensor, dim int, keepDim bool) *MeanDimOp {
	// Normalize negative dim
	actualDim := dim
	if actualDim < 0 {
		actualDim = len(x.Shape()) + actualDim
	}

	return &MeanDimOp{
		inputs:  []*tensor.RawTensor{x},
		output:  output,
		dim:     dim,
		keepDim: keepDim,
		dimSize: x.Shape()[actualDim],
	}
}

// Backward computes input gradients for mean reduction.
//
// The gradient flows by broadcasting grad_output to match input shape,
// then dividing by the size of the reduced dimension.
func (op *MeanDimOp) Backward(outputGrad *tensor.RawTensor, backend tensor.Backend) []*tensor.RawTensor {
	x := op.inputs[0]
	grad := outputGrad

	// If keepDim=false, we need to unsqueeze the gradient first
	if !op.keepDim {
		grad = unsqueezeDim(grad, op.dim, x.Shape())
	}

	// Broadcast gradient to input shape
	gradX := broadcastTo(grad, x.Shape(), backend)

	// Divide by the size of the reduced dimension
	divisor := float64(op.dimSize)
	switch gradX.DType() {
	case tensor.Float32:
		data := gradX.AsFloat32()
		divisorF32 := float32(divisor)
		for i := range data {
			data[i] /= divisorF32
		}
	case tensor.Float64:
		data := gradX.AsFloat64()
		for i := range data {
			data[i] /= divisor
		}
	}

	return []*tensor.RawTensor{gradX}
}

// Inputs returns the input tensors [x].
func (op *MeanDimOp) Inputs() []*tensor.RawTensor {
	return op.inputs
}

// Output returns the output tensor mean(x, dim).
func (op *MeanDimOp) Output() *tensor.RawTensor {
	return op.output
}
