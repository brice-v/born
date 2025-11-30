package ops

import (
	"github.com/born-ml/born/internal/tensor"
)

// EmbeddingOp represents an embedding lookup operation.
//
// Forward: output[i] = weight[indices[i]]
//
// Backward:
//
//	For each index i, accumulate grad_output[i] to grad_weight[indices[i]]
//	This is a scatter-add operation where gradients for the same index are summed.
//
// Example:
//
//	indices = [0, 1, 0]  // index 0 appears twice
//	grad_output = [[1,2], [3,4], [5,6]]
//	grad_weight[0] = [1,2] + [5,6] = [6,8]  // Accumulated!
//	grad_weight[1] = [3,4]
type EmbeddingOp struct {
	weight  *tensor.RawTensor // Embedding weight [numEmbeddings, embeddingDim]
	indices *tensor.RawTensor // Index tensor (int32)
	output  *tensor.RawTensor // Output embeddings
}

// NewEmbeddingOp creates a new embedding operation.
func NewEmbeddingOp(weight, indices, output *tensor.RawTensor) *EmbeddingOp {
	return &EmbeddingOp{
		weight:  weight,
		indices: indices,
		output:  output,
	}
}

// Inputs returns the input tensors (weight and indices).
// Note: Only weight needs gradient; indices are integer indices.
func (op *EmbeddingOp) Inputs() []*tensor.RawTensor {
	return []*tensor.RawTensor{op.weight}
}

// Output returns the output tensor.
func (op *EmbeddingOp) Output() *tensor.RawTensor {
	return op.output
}

// Backward computes gradients for the embedding weights.
//
// Gradient computation:
//   - For each position i in output, grad_output[i] flows back to weight[indices[i]]
//   - Multiple indices pointing to the same embedding accumulate gradients
//
// Algorithm:
//  1. Create grad_weight tensor (same shape as weight) initialized to zeros
//  2. For each index i:
//     - Read index value: idx = indices[i]
//     - Add grad_output[i] to grad_weight[idx]
//  3. Return grad_weight
func (op *EmbeddingOp) Backward(gradOutput *tensor.RawTensor, backend tensor.Backend) []*tensor.RawTensor {
	weightShape := op.weight.Shape()
	numEmbeddings := weightShape[0]
	embeddingDim := weightShape[1]

	// Create gradient for weight (initialized to zero)
	gradWeight, err := tensor.NewRaw(weightShape, tensor.Float32, backend.Device())
	if err != nil {
		panic(err)
	}

	// Zero-initialize gradient
	gradWeightData := gradWeight.AsFloat32()
	for i := range gradWeightData {
		gradWeightData[i] = 0
	}

	// Scatter-add gradients to weight rows
	indicesData := op.indices.AsInt32()
	gradOutputData := gradOutput.AsFloat32()

	numIndices := op.indices.NumElements()

	for i := 0; i < numIndices; i++ {
		idx := int(indicesData[i])

		// Validate index
		if idx < 0 || idx >= numEmbeddings {
			panic("embedding backward: index out of bounds")
		}

		// Accumulate gradient for this embedding
		gradOutOffset := i * embeddingDim
		gradWeightOffset := idx * embeddingDim

		for j := 0; j < embeddingDim; j++ {
			gradWeightData[gradWeightOffset+j] += gradOutputData[gradOutOffset+j]
		}
	}

	// Return gradient for weight (indices don't need gradient)
	return []*tensor.RawTensor{gradWeight}
}
