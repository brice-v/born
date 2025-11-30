package ops

import (
	"math"

	"github.com/born-ml/born/internal/tensor"
)

// SiLUOp represents the SiLU (Swish) activation operation: y = x * sigmoid(x).
//
// Also known as Swish activation, widely used in modern transformers
// (LLaMA, Mistral, GPT-Neo).
type SiLUOp struct {
	input  *tensor.RawTensor
	output *tensor.RawTensor
}

// NewSiLUOp creates a new SiLU operation.
func NewSiLUOp(input, output *tensor.RawTensor) *SiLUOp {
	return &SiLUOp{
		input:  input,
		output: output,
	}
}

// Inputs returns the input tensors.
func (op *SiLUOp) Inputs() []*tensor.RawTensor {
	return []*tensor.RawTensor{op.input}
}

// Output returns the output tensor.
func (op *SiLUOp) Output() *tensor.RawTensor {
	return op.output
}

// Backward computes the gradient for SiLU.
//
// For y = x * sigmoid(x):
//
//	dy/dx = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
//	      = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
//
// We compute the gradient directly for numerical accuracy.
func (op *SiLUOp) Backward(outputGrad *tensor.RawTensor, backend tensor.Backend) []*tensor.RawTensor {
	x := op.input

	// Create gradient tensor
	inputGrad, err := tensor.NewRaw(x.Shape(), x.DType(), backend.Device())
	if err != nil {
		panic(err)
	}

	// Compute gradient directly element-wise
	switch x.DType() {
	case tensor.Float32:
		xData := x.AsFloat32()
		gradData := inputGrad.AsFloat32()
		outGradData := outputGrad.AsFloat32()

		for i := range xData {
			// sigmoid(x) = 1 / (1 + exp(-x))
			sig := float32(1.0 / (1.0 + math.Exp(float64(-xData[i]))))

			// dy/dx = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
			derivative := sig * (1.0 + xData[i]*(1.0-sig))

			// grad_input = grad_output * dy/dx
			gradData[i] = outGradData[i] * derivative
		}

	case tensor.Float64:
		xData := x.AsFloat64()
		gradData := inputGrad.AsFloat64()
		outGradData := outputGrad.AsFloat64()

		for i := range xData {
			// sigmoid(x) = 1 / (1 + exp(-x))
			sig := 1.0 / (1.0 + math.Exp(-xData[i]))

			// dy/dx = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
			derivative := sig * (1.0 + xData[i]*(1.0-sig))

			// grad_input = grad_output * dy/dx
			gradData[i] = outGradData[i] * derivative
		}
	}

	return []*tensor.RawTensor{inputGrad}
}
