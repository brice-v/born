package ops

import (
	"math"

	"github.com/born-ml/born/internal/tensor"
)

// SoftmaxOp represents the softmax operation along the last dimension.
//
// Forward (for each row):
//
//	softmax(x)_i = exp(x_i - max(x)) / Σ_j exp(x_j - max(x))
//
// The max-shifting ensures numerical stability (prevents overflow).
//
// Backward:
//
//	The Jacobian of softmax is:
//	∂softmax_i/∂x_j = softmax_i * (δ_ij - softmax_j)
//
//	Chain rule gives:
//	∂L/∂x_j = Σ_i (∂L/∂softmax_i) * softmax_i * (δ_ij - softmax_j)
//	        = softmax_j * (∂L/∂softmax_j - Σ_i (∂L/∂softmax_i * softmax_i))
//
// Assumptions:
//   - Input shape: [batch_size, num_classes] (2D)
//   - Softmax applied along last dimension (classes)
type SoftmaxOp struct {
	input  *tensor.RawTensor
	output *tensor.RawTensor // Cached softmax output for backward pass
}

// NewSoftmaxOp creates a new softmax operation.
func NewSoftmaxOp(input, output *tensor.RawTensor) *SoftmaxOp {
	return &SoftmaxOp{
		input:  input,
		output: output,
	}
}

// Inputs returns the input tensors.
func (op *SoftmaxOp) Inputs() []*tensor.RawTensor {
	return []*tensor.RawTensor{op.input}
}

// Output returns the output tensor.
func (op *SoftmaxOp) Output() *tensor.RawTensor {
	return op.output
}

// Backward computes the gradient with respect to input.
//
// Uses the simplified formula for batched softmax:
//
//	∂L/∂x[b,j] = softmax[b,j] * (∂L/∂softmax[b,j] - dot(∂L/∂softmax[b,:], softmax[b,:]))
//
// Where:
//   - b is the batch index
//   - j is the class index
//   - dot is the dot product over the class dimension
func (op *SoftmaxOp) Backward(outputGrad *tensor.RawTensor, _ tensor.Backend) []*tensor.RawTensor {
	shape := op.input.Shape()
	if len(shape) != 2 {
		panic("SoftmaxOp: backward only supports 2D tensors [batch_size, num_classes]")
	}

	batchSize := shape[0]
	numClasses := shape[1]

	// Create gradient tensor
	inputGrad, err := tensor.NewRaw(shape, op.input.DType(), op.input.Device())
	if err != nil {
		panic(err)
	}

	switch op.input.DType() {
	case tensor.Float32:
		softmaxData := op.output.AsFloat32()
		outGradData := outputGrad.AsFloat32()
		inGradData := inputGrad.AsFloat32()

		for b := 0; b < batchSize; b++ {
			// Compute dot product: Σ_i (grad_output[i] * softmax[i])
			dotProduct := float32(0.0)
			for j := 0; j < numClasses; j++ {
				idx := b*numClasses + j
				dotProduct += outGradData[idx] * softmaxData[idx]
			}

			// Compute gradient for each class
			for j := 0; j < numClasses; j++ {
				idx := b*numClasses + j
				// grad_input[j] = softmax[j] * (grad_output[j] - dotProduct)
				inGradData[idx] = softmaxData[idx] * (outGradData[idx] - dotProduct)
			}
		}

	case tensor.Float64:
		softmaxData := op.output.AsFloat64()
		outGradData := outputGrad.AsFloat64()
		inGradData := inputGrad.AsFloat64()

		for b := 0; b < batchSize; b++ {
			dotProduct := 0.0
			for j := 0; j < numClasses; j++ {
				idx := b*numClasses + j
				dotProduct += outGradData[idx] * softmaxData[idx]
			}

			for j := 0; j < numClasses; j++ {
				idx := b*numClasses + j
				inGradData[idx] = softmaxData[idx] * (outGradData[idx] - dotProduct)
			}
		}

	default:
		panic("SoftmaxOp: backward only supports float32 and float64")
	}

	return []*tensor.RawTensor{inputGrad}
}

// LogSoftmaxOp represents the log-softmax operation.
//
// Forward:
//
//	log_softmax(x)_i = x_i - max(x) - log(Σ_j exp(x_j - max(x)))
//
// This is more numerically stable than computing softmax then log.
//
// Backward:
//
//	∂L/∂x_j = ∂L/∂log_softmax_j - softmax_j * Σ_i ∂L/∂log_softmax_i
//
// Note: We need to cache both log_softmax (output) and softmax for backward.
type LogSoftmaxOp struct {
	input       *tensor.RawTensor
	output      *tensor.RawTensor // log_softmax output
	softmaxData []float32         // Cached softmax for backward (float32 only for now)
}

// NewLogSoftmaxOp creates a new log-softmax operation.
//
// Parameters:
//   - input: Input logits
//   - output: Log-softmax output
//   - softmaxData: Pre-computed softmax (needed for backward)
func NewLogSoftmaxOp(input, output *tensor.RawTensor, softmaxData []float32) *LogSoftmaxOp {
	return &LogSoftmaxOp{
		input:       input,
		output:      output,
		softmaxData: softmaxData,
	}
}

// Inputs returns the input tensors.
func (op *LogSoftmaxOp) Inputs() []*tensor.RawTensor {
	return []*tensor.RawTensor{op.input}
}

// Output returns the output tensor.
func (op *LogSoftmaxOp) Output() *tensor.RawTensor {
	return op.output
}

// Backward computes gradient for log-softmax.
//
// Formula:
//
//	∂L/∂x[b,j] = ∂L/∂log_softmax[b,j] - softmax[b,j] * Σ_i ∂L/∂log_softmax[b,i]
func (op *LogSoftmaxOp) Backward(outputGrad *tensor.RawTensor, _ tensor.Backend) []*tensor.RawTensor {
	shape := op.input.Shape()
	if len(shape) != 2 {
		panic("LogSoftmaxOp: backward only supports 2D tensors")
	}

	batchSize := shape[0]
	numClasses := shape[1]

	inputGrad, err := tensor.NewRaw(shape, op.input.DType(), op.input.Device())
	if err != nil {
		panic(err)
	}

	switch op.input.DType() {
	case tensor.Float32:
		outGradData := outputGrad.AsFloat32()
		inGradData := inputGrad.AsFloat32()

		for b := 0; b < batchSize; b++ {
			// Sum gradient over classes: Σ_i ∂L/∂log_softmax[i]
			gradSum := float32(0.0)
			for j := 0; j < numClasses; j++ {
				idx := b*numClasses + j
				gradSum += outGradData[idx]
			}

			// Compute gradient
			for j := 0; j < numClasses; j++ {
				idx := b*numClasses + j
				inGradData[idx] = outGradData[idx] - op.softmaxData[idx]*gradSum
			}
		}

	default:
		panic("LogSoftmaxOp: backward only supports float32 for now")
	}

	return []*tensor.RawTensor{inputGrad}
}

// softmaxFloat32 computes softmax for float32 data.
func softmaxFloat32(inputData, outputData []float32, batchSize, numClasses int) {
	for b := 0; b < batchSize; b++ {
		// Find max for numerical stability
		offset := b * numClasses
		maxVal := inputData[offset]
		for j := 1; j < numClasses; j++ {
			if inputData[offset+j] > maxVal {
				maxVal = inputData[offset+j]
			}
		}

		// Compute exp and sum
		sumExp := float32(0.0)
		for j := 0; j < numClasses; j++ {
			idx := offset + j
			outputData[idx] = float32(math.Exp(float64(inputData[idx] - maxVal)))
			sumExp += outputData[idx]
		}

		// Normalize
		for j := 0; j < numClasses; j++ {
			outputData[offset+j] /= sumExp
		}
	}
}

// softmaxFloat64 computes softmax for float64 data.
func softmaxFloat64(inputData, outputData []float64, batchSize, numClasses int) {
	for b := 0; b < batchSize; b++ {
		offset := b * numClasses
		maxVal := inputData[offset]
		for j := 1; j < numClasses; j++ {
			if inputData[offset+j] > maxVal {
				maxVal = inputData[offset+j]
			}
		}

		sumExp := 0.0
		for j := 0; j < numClasses; j++ {
			idx := offset + j
			outputData[idx] = math.Exp(inputData[idx] - maxVal)
			sumExp += outputData[idx]
		}

		for j := 0; j < numClasses; j++ {
			outputData[offset+j] /= sumExp
		}
	}
}

// Softmax computes softmax along last dimension (helper function).
//
// This is a helper for use outside autodiff.
// For autodiff support, use backend.Softmax() which records SoftmaxOp.
func Softmax(input *tensor.RawTensor, device tensor.Device) *tensor.RawTensor {
	shape := input.Shape()
	if len(shape) != 2 {
		panic("Softmax: only supports 2D tensors [batch_size, num_classes]")
	}

	batchSize := shape[0]
	numClasses := shape[1]

	output, err := tensor.NewRaw(shape, input.DType(), device)
	if err != nil {
		panic(err)
	}

	switch input.DType() {
	case tensor.Float32:
		softmaxFloat32(input.AsFloat32(), output.AsFloat32(), batchSize, numClasses)

	case tensor.Float64:
		softmaxFloat64(input.AsFloat64(), output.AsFloat64(), batchSize, numClasses)

	default:
		panic("Softmax: only supports float32 and float64")
	}

	return output
}
