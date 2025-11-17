package ops

import (
	"math"

	"github.com/born-ml/born/internal/tensor"
)

// CrossEntropyOp represents the cross-entropy loss operation.
//
// Forward:
//
//	Loss = mean(-log_softmax(logits)[targets])
//
// Where log_softmax uses the log-sum-exp trick for numerical stability:
//
//	log_softmax(z) = z - (max(z) + log(Σ exp(z - max(z))))
//
// Backward:
//
//	∂L/∂logits = (softmax(logits) - y_one_hot) / batch_size
//
// This elegant gradient formula is the key reason why softmax + cross-entropy
// are often fused together in modern frameworks (PyTorch, TensorFlow, Burn).
//
// Assumptions:
//   - Logits shape: [batch_size, num_classes] (2D)
//   - Targets shape: [batch_size] (1D, class indices)
//   - Output: scalar loss (mean over batch)
type CrossEntropyOp struct {
	logits  *tensor.RawTensor // Input logits [batch_size, num_classes]
	targets *tensor.RawTensor // Target class indices [batch_size]
	output  *tensor.RawTensor // Scalar loss output
}

// NewCrossEntropyOp creates a new cross-entropy operation.
func NewCrossEntropyOp(logits, targets, output *tensor.RawTensor) *CrossEntropyOp {
	return &CrossEntropyOp{
		logits:  logits,
		targets: targets,
		output:  output,
	}
}

// Inputs returns the input tensors.
func (op *CrossEntropyOp) Inputs() []*tensor.RawTensor {
	return []*tensor.RawTensor{op.logits}
}

// Output returns the output tensor.
func (op *CrossEntropyOp) Output() *tensor.RawTensor {
	return op.output
}

// Backward computes the gradient with respect to logits.
//
// Gradient formula:
//
//	∂L/∂logits[b,i] = (softmax(logits[b])[i] - y_one_hot[b,i]) / batch_size
//
// Where y_one_hot[b,i] = 1 if i == targets[b], else 0.
//
// Note: The gradient is averaged over the batch size because the forward
// pass computes mean loss.
func (op *CrossEntropyOp) Backward(outputGrad *tensor.RawTensor, _ tensor.Backend) []*tensor.RawTensor {
	logitsShape := op.logits.Shape()
	if len(logitsShape) != 2 {
		panic("CrossEntropyOp: backward only supports 2D logits [batch_size, num_classes]")
	}

	batchSize := logitsShape[0]
	numClasses := logitsShape[1]

	// Create gradient tensor for logits
	logitsGrad, err := tensor.NewRaw(logitsShape, op.logits.DType(), op.logits.Device())
	if err != nil {
		panic(err)
	}

	switch op.logits.DType() {
	case tensor.Float32:
		computeCrossEntropyGradFloat32(
			op.logits.AsFloat32(),
			op.targets.AsInt32(),
			outputGrad.AsFloat32(),
			logitsGrad.AsFloat32(),
			batchSize,
			numClasses,
		)

	case tensor.Float64:
		computeCrossEntropyGradFloat64(
			op.logits.AsFloat64(),
			op.targets.AsInt32(),
			outputGrad.AsFloat64(),
			logitsGrad.AsFloat64(),
			batchSize,
			numClasses,
		)

	default:
		panic("CrossEntropyOp: backward only supports float32 and float64")
	}

	return []*tensor.RawTensor{logitsGrad}
}

// computeCrossEntropyGradFloat32 computes gradients for float32 cross-entropy.
func computeCrossEntropyGradFloat32(
	logitsData []float32,
	targetsData []int32,
	outGradData []float32,
	gradData []float32,
	batchSize, numClasses int,
) {
	gradScale := outGradData[0] // Usually 1.0, but we respect upstream gradient

	for b := 0; b < batchSize; b++ {
		// Extract logits for this sample
		sampleLogits := logitsData[b*numClasses : (b+1)*numClasses]

		// Compute softmax probabilities
		probs := computeSoftmaxFloat32(sampleLogits)

		// Gradient = (softmax - y_one_hot) / batch_size
		target := int(targetsData[b])
		for i := 0; i < numClasses; i++ {
			grad := probs[i]
			if i == target {
				grad -= 1.0
			}

			// Scale by upstream gradient and average over batch
			gradData[b*numClasses+i] = gradScale * grad / float32(batchSize)
		}
	}
}

// computeCrossEntropyGradFloat64 computes gradients for float64 cross-entropy.
func computeCrossEntropyGradFloat64(
	logitsData []float64,
	targetsData []int32,
	outGradData []float64,
	gradData []float64,
	batchSize, numClasses int,
) {
	gradScale := outGradData[0]

	for b := 0; b < batchSize; b++ {
		sampleLogits := logitsData[b*numClasses : (b+1)*numClasses]
		probs := computeSoftmaxFloat64(sampleLogits)

		target := int(targetsData[b])
		for i := 0; i < numClasses; i++ {
			grad := probs[i]
			if i == target {
				grad -= 1.0
			}
			gradData[b*numClasses+i] = gradScale * grad / float64(batchSize)
		}
	}
}

// computeSoftmaxFloat32 computes softmax with numerical stability for a single sample.
func computeSoftmaxFloat32(logits []float32) []float32 {
	n := len(logits)
	probs := make([]float32, n)

	// Find max for numerical stability
	maxVal := logits[0]
	for i := 1; i < n; i++ {
		if logits[i] > maxVal {
			maxVal = logits[i]
		}
	}

	// Compute exp(z - max) and sum
	sumExp := float32(0.0)
	for i := 0; i < n; i++ {
		probs[i] = float32(math.Exp(float64(logits[i] - maxVal)))
		sumExp += probs[i]
	}

	// Normalize
	for i := 0; i < n; i++ {
		probs[i] /= sumExp
	}

	return probs
}

// computeSoftmaxFloat64 computes softmax with numerical stability for a single sample.
func computeSoftmaxFloat64(logits []float64) []float64 {
	n := len(logits)
	probs := make([]float64, n)

	// Find max for numerical stability
	maxVal := logits[0]
	for i := 1; i < n; i++ {
		if logits[i] > maxVal {
			maxVal = logits[i]
		}
	}

	// Compute exp(z - max) and sum
	sumExp := 0.0
	for i := 0; i < n; i++ {
		probs[i] = math.Exp(logits[i] - maxVal)
		sumExp += probs[i]
	}

	// Normalize
	for i := 0; i < n; i++ {
		probs[i] /= sumExp
	}

	return probs
}

// CrossEntropyForward computes cross-entropy loss (helper function).
//
// This is a helper for use outside autodiff context.
// For autodiff support, use AutodiffBackend with CrossEntropyOp.
//
// Parameters:
//   - logits: [batch_size, num_classes]
//   - targets: [batch_size] (class indices)
//
// Returns:
//   - Scalar loss tensor (mean over batch)
func CrossEntropyForward(logits, targets *tensor.RawTensor, device tensor.Device) *tensor.RawTensor {
	logitsShape := logits.Shape()
	if len(logitsShape) != 2 {
		panic("CrossEntropyForward: logits must be 2D [batch_size, num_classes]")
	}

	targetsShape := targets.Shape()
	if len(targetsShape) != 1 {
		panic("CrossEntropyForward: targets must be 1D [batch_size]")
	}

	batchSize := logitsShape[0]
	numClasses := logitsShape[1]

	if targetsShape[0] != batchSize {
		panic("CrossEntropyForward: batch size mismatch between logits and targets")
	}

	// Create scalar output
	output, err := tensor.NewRaw(tensor.Shape{1}, logits.DType(), device)
	if err != nil {
		panic(err)
	}

	switch logits.DType() {
	case tensor.Float32:
		logitsData := logits.AsFloat32()
		targetsData := targets.AsInt32()

		totalLoss := float32(0.0)

		for b := 0; b < batchSize; b++ {
			sampleLogits := logitsData[b*numClasses : (b+1)*numClasses]
			logProbs := computeLogSoftmaxFloat32(sampleLogits)

			target := int(targetsData[b])
			if target < 0 || target >= numClasses {
				panic("CrossEntropyForward: target index out of bounds")
			}

			// Negative log-likelihood
			totalLoss += -logProbs[target]
		}

		// Average over batch
		output.AsFloat32()[0] = totalLoss / float32(batchSize)

	case tensor.Float64:
		logitsData := logits.AsFloat64()
		targetsData := targets.AsInt32()

		totalLoss := 0.0

		for b := 0; b < batchSize; b++ {
			sampleLogits := logitsData[b*numClasses : (b+1)*numClasses]
			logProbs := computeLogSoftmaxFloat64(sampleLogits)

			target := int(targetsData[b])
			if target < 0 || target >= numClasses {
				panic("CrossEntropyForward: target index out of bounds")
			}

			totalLoss += -logProbs[target]
		}

		output.AsFloat64()[0] = totalLoss / float64(batchSize)

	default:
		panic("CrossEntropyForward: only supports float32 and float64")
	}

	return output
}

// computeLogSoftmaxFloat32 computes log-softmax with numerical stability.
func computeLogSoftmaxFloat32(logits []float32) []float32 {
	n := len(logits)
	result := make([]float32, n)

	// Find max for numerical stability
	maxVal := logits[0]
	for i := 1; i < n; i++ {
		if logits[i] > maxVal {
			maxVal = logits[i]
		}
	}

	// Compute log-sum-exp: log(Σ exp(z - max))
	sumExp := float32(0.0)
	for i := 0; i < n; i++ {
		sumExp += float32(math.Exp(float64(logits[i] - maxVal)))
	}
	logSumExp := maxVal + float32(math.Log(float64(sumExp)))

	// log_softmax = z - log_sum_exp
	for i := 0; i < n; i++ {
		result[i] = logits[i] - logSumExp
	}

	return result
}

// computeLogSoftmaxFloat64 computes log-softmax with numerical stability.
func computeLogSoftmaxFloat64(logits []float64) []float64 {
	n := len(logits)
	result := make([]float64, n)

	// Find max for numerical stability
	maxVal := logits[0]
	for i := 1; i < n; i++ {
		if logits[i] > maxVal {
			maxVal = logits[i]
		}
	}

	// Compute log-sum-exp
	sumExp := 0.0
	for i := 0; i < n; i++ {
		sumExp += math.Exp(logits[i] - maxVal)
	}
	logSumExp := maxVal + math.Log(sumExp)

	// log_softmax = z - log_sum_exp
	for i := 0; i < n; i++ {
		result[i] = logits[i] - logSumExp
	}

	return result
}
