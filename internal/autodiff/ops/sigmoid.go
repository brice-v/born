package ops

import (
	"github.com/born-ml/born/internal/tensor"
)

// SigmoidOp represents the sigmoid activation operation: σ(x) = 1 / (1 + exp(-x)).
type SigmoidOp struct {
	input  *tensor.RawTensor
	output *tensor.RawTensor
}

// NewSigmoidOp creates a new sigmoid operation.
func NewSigmoidOp(input, output *tensor.RawTensor) *SigmoidOp {
	return &SigmoidOp{
		input:  input,
		output: output,
	}
}

// Inputs returns the input tensors.
func (op *SigmoidOp) Inputs() []*tensor.RawTensor {
	return []*tensor.RawTensor{op.input}
}

// Output returns the output tensor.
func (op *SigmoidOp) Output() *tensor.RawTensor {
	return op.output
}

// Backward computes the gradient for sigmoid.
//
// For σ(x) = 1 / (1 + exp(-x)):
// dσ/dx = σ(x) * (1 - σ(x))
//
// Since we have the output σ(x) already computed, we can use it:
// grad_input = grad_output * output * (1 - output).
func (op *SigmoidOp) Backward(outputGrad *tensor.RawTensor, backend tensor.Backend) []*tensor.RawTensor {
	output := op.output

	// Create tensor filled with ones
	ones, err := tensor.NewRaw(output.Shape(), output.DType(), backend.Device())
	if err != nil {
		panic(err)
	}

	// Fill with ones
	switch output.DType() {
	case tensor.Float32:
		data := ones.AsFloat32()
		for i := range data {
			data[i] = 1.0
		}
	case tensor.Float64:
		data := ones.AsFloat64()
		for i := range data {
			data[i] = 1.0
		}
	}

	// 1 - σ(x)
	oneMinusSigmoid := backend.Sub(ones, output)

	// σ(x) * (1 - σ(x))
	sigmoidDerivative := backend.Mul(output, oneMinusSigmoid)

	// grad_input = grad_output * σ'(x)
	inputGrad := backend.Mul(outputGrad, sigmoidDerivative)

	return []*tensor.RawTensor{inputGrad}
}
