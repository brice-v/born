package ops

import (
	"math"

	"github.com/born-ml/born/internal/tensor"
)

// LogOp represents element-wise natural logarithm operation.
//
// Forward:
//
//	output = log(input)
//
// Backward:
//
//	∂L/∂input = ∂L/∂output * (1 / input)
//
// The gradient is the reciprocal of the input, scaled by the output gradient.
type LogOp struct {
	input  *tensor.RawTensor
	output *tensor.RawTensor
}

// NewLogOp creates a new log operation.
func NewLogOp(input, output *tensor.RawTensor) *LogOp {
	return &LogOp{
		input:  input,
		output: output,
	}
}

// Inputs returns the input tensors.
func (op *LogOp) Inputs() []*tensor.RawTensor {
	return []*tensor.RawTensor{op.input}
}

// Output returns the output tensor.
func (op *LogOp) Output() *tensor.RawTensor {
	return op.output
}

// Backward computes the gradient with respect to input.
//
// Gradient formula:
//
//	∂L/∂input[i] = ∂L/∂output[i] * (1 / input[i])
//
// Note: This assumes input > 0 (log is only defined for positive values).
// In practice, a small epsilon (e.g., 1e-8) is often added for numerical stability.
func (op *LogOp) Backward(outputGrad *tensor.RawTensor, _ tensor.Backend) []*tensor.RawTensor {
	// Create gradient tensor with same shape as input
	inputGrad, err := tensor.NewRaw(op.input.Shape(), op.input.DType(), op.input.Device())
	if err != nil {
		panic(err)
	}

	// Get data slices based on dtype
	switch op.input.DType() {
	case tensor.Float32:
		inputData := op.input.AsFloat32()
		gradData := inputGrad.AsFloat32()
		outGradData := outputGrad.AsFloat32()

		// Compute gradient: grad_input = grad_output / input
		for i := range inputData {
			gradData[i] = outGradData[i] / inputData[i]
		}

	case tensor.Float64:
		inputData := op.input.AsFloat64()
		gradData := inputGrad.AsFloat64()
		outGradData := outputGrad.AsFloat64()

		for i := range inputData {
			gradData[i] = outGradData[i] / inputData[i]
		}

	default:
		panic("LogOp: backward only supports float32 and float64")
	}

	return []*tensor.RawTensor{inputGrad}
}

// LogWithEpsilonOp represents log with numerical stability epsilon.
//
// Forward:
//
//	output = log(input + epsilon)
//
// This is numerically more stable when input might be very close to zero.
type LogWithEpsilonOp struct {
	input   *tensor.RawTensor
	output  *tensor.RawTensor
	epsilon float64
}

// NewLogWithEpsilonOp creates a log operation with epsilon for stability.
func NewLogWithEpsilonOp(input, output *tensor.RawTensor, epsilon float64) *LogWithEpsilonOp {
	return &LogWithEpsilonOp{
		input:   input,
		output:  output,
		epsilon: epsilon,
	}
}

// Inputs returns the input tensors.
func (op *LogWithEpsilonOp) Inputs() []*tensor.RawTensor {
	return []*tensor.RawTensor{op.input}
}

// Output returns the output tensor.
func (op *LogWithEpsilonOp) Output() *tensor.RawTensor {
	return op.output
}

// Backward computes gradient: ∂L/∂input = ∂L/∂output / (input + epsilon).
func (op *LogWithEpsilonOp) Backward(outputGrad *tensor.RawTensor, _ tensor.Backend) []*tensor.RawTensor {
	inputGrad, err := tensor.NewRaw(op.input.Shape(), op.input.DType(), op.input.Device())
	if err != nil {
		panic(err)
	}

	switch op.input.DType() {
	case tensor.Float32:
		inputData := op.input.AsFloat32()
		gradData := inputGrad.AsFloat32()
		outGradData := outputGrad.AsFloat32()
		eps := float32(op.epsilon)

		for i := range inputData {
			gradData[i] = outGradData[i] / (inputData[i] + eps)
		}

	case tensor.Float64:
		inputData := op.input.AsFloat64()
		gradData := inputGrad.AsFloat64()
		outGradData := outputGrad.AsFloat64()

		for i := range inputData {
			gradData[i] = outGradData[i] / (inputData[i] + op.epsilon)
		}

	default:
		panic("LogWithEpsilonOp: backward only supports float32 and float64")
	}

	return []*tensor.RawTensor{inputGrad}
}

// Exp computes element-wise exponential (helper for softmax).
//
// Forward: output = exp(input)
// Backward: ∂L/∂input = ∂L/∂output * exp(input) = ∂L/∂output * output
//
// Note: This is a helper function, not a full Operation.
// For autodiff support, use ExpOp (to be implemented if needed).
func Exp(input *tensor.RawTensor, device tensor.Device) *tensor.RawTensor {
	output, err := tensor.NewRaw(input.Shape(), input.DType(), device)
	if err != nil {
		panic(err)
	}

	switch input.DType() {
	case tensor.Float32:
		inputData := input.AsFloat32()
		outputData := output.AsFloat32()
		for i, val := range inputData {
			outputData[i] = float32(math.Exp(float64(val)))
		}

	case tensor.Float64:
		inputData := input.AsFloat64()
		outputData := output.AsFloat64()
		for i, val := range inputData {
			outputData[i] = math.Exp(val)
		}

	default:
		panic("Exp: only supports float32 and float64")
	}

	return output
}

// Log computes element-wise natural logarithm (helper function).
//
// Forward: output = log(input)
//
// Note: This is a helper function for use outside autodiff.
// For autodiff support, use backend.Log() which records LogOp.
func Log(input *tensor.RawTensor, device tensor.Device) *tensor.RawTensor {
	output, err := tensor.NewRaw(input.Shape(), input.DType(), device)
	if err != nil {
		panic(err)
	}

	switch input.DType() {
	case tensor.Float32:
		inputData := input.AsFloat32()
		outputData := output.AsFloat32()
		for i, val := range inputData {
			outputData[i] = float32(math.Log(float64(val)))
		}

	case tensor.Float64:
		inputData := input.AsFloat64()
		outputData := output.AsFloat64()
		for i, val := range inputData {
			outputData[i] = math.Log(val)
		}

	default:
		panic("Log: only supports float32 and float64")
	}

	return output
}
