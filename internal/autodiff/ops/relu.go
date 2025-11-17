package ops

import (
	"fmt"

	"github.com/born-ml/born/internal/tensor"
)

// ReLUOp represents a ReLU (Rectified Linear Unit) activation: output = max(0, x).
//
// Backward pass:
//   - d(ReLU(x))/dx = 1 if x > 0, else 0
//
// The gradient is computed by creating a mask where input > 0, then
// multiplying the output gradient by this mask.
type ReLUOp struct {
	input  *tensor.RawTensor // x
	output *tensor.RawTensor // max(0, x)
}

// NewReLUOp creates a new ReLUOp.
func NewReLUOp(input, output *tensor.RawTensor) *ReLUOp {
	return &ReLUOp{
		input:  input,
		output: output,
	}
}

// Backward computes input gradient for ReLU.
func (op *ReLUOp) Backward(outputGrad *tensor.RawTensor, backend tensor.Backend) []*tensor.RawTensor {
	// Create mask: 1 where input > 0, 0 otherwise
	mask := createReLUMask(op.input, backend)

	// grad_input = outputGrad * mask
	gradInput := backend.Mul(outputGrad, mask)

	return []*tensor.RawTensor{gradInput}
}

// Inputs returns the input tensor [x].
func (op *ReLUOp) Inputs() []*tensor.RawTensor {
	return []*tensor.RawTensor{op.input}
}

// Output returns the output tensor max(0, x).
func (op *ReLUOp) Output() *tensor.RawTensor {
	return op.output
}

// createReLUMask creates a binary mask where input > 0.
func createReLUMask(input *tensor.RawTensor, backend tensor.Backend) *tensor.RawTensor {
	// Create a tensor of same shape as input
	mask, err := tensor.NewRaw(input.Shape(), input.DType(), backend.Device())
	if err != nil {
		panic(fmt.Sprintf("relu: failed to create mask: %v", err))
	}

	// Set mask values based on input dtype
	switch input.DType() {
	case tensor.Float32:
		inputData := input.AsFloat32()
		maskData := mask.AsFloat32()
		for i, val := range inputData {
			if val > 0 {
				maskData[i] = 1.0
			} else {
				maskData[i] = 0.0
			}
		}

	case tensor.Float64:
		inputData := input.AsFloat64()
		maskData := mask.AsFloat64()
		for i, val := range inputData {
			if val > 0 {
				maskData[i] = 1.0
			} else {
				maskData[i] = 0.0
			}
		}

	default:
		panic(fmt.Sprintf("relu: unsupported dtype %s (only float32/float64 supported)", input.DType()))
	}

	return mask
}
