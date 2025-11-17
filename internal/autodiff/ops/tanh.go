package ops

import (
	"github.com/born-ml/born/internal/tensor"
)

// TanhOp represents the hyperbolic tangent activation: tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x)).
type TanhOp struct {
	input  *tensor.RawTensor
	output *tensor.RawTensor
}

// NewTanhOp creates a new tanh operation.
func NewTanhOp(input, output *tensor.RawTensor) *TanhOp {
	return &TanhOp{
		input:  input,
		output: output,
	}
}

// Inputs returns the input tensors.
func (op *TanhOp) Inputs() []*tensor.RawTensor {
	return []*tensor.RawTensor{op.input}
}

// Output returns the output tensor.
func (op *TanhOp) Output() *tensor.RawTensor {
	return op.output
}

// Backward computes the gradient for tanh.
//
// For tanh(x):
// d(tanh(x))/dx = 1 - tanh²(x)
//
// Since we have the output tanh(x) already computed:
// grad_input = grad_output * (1 - output²).
func (op *TanhOp) Backward(outputGrad *tensor.RawTensor, backend tensor.Backend) []*tensor.RawTensor {
	output := op.output

	// output²
	outputSquared := backend.Mul(output, output)

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

	// 1 - tanh²(x)
	tanhDerivative := backend.Sub(ones, outputSquared)

	// grad_input = grad_output * (1 - tanh²(x))
	inputGrad := backend.Mul(outputGrad, tanhDerivative)

	return []*tensor.RawTensor{inputGrad}
}
