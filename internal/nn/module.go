// Package nn implements neural network modules for the Born ML Framework.
//
// This package provides building blocks for constructing neural networks:
//   - Module interface: Base interface for all NN components
//   - Parameter: Trainable parameters with gradient tracking
//   - Linear: Fully connected layer
//   - Activations: ReLU, Sigmoid, Tanh
//   - Loss functions: MSE, CrossEntropy
//   - Sequential: Container for stacking layers
//
// Design inspired by PyTorch's nn.Module but adapted for Go generics.
package nn

import (
	"github.com/born-ml/born/internal/tensor"
)

// Module is the base interface for all neural network components.
//
// Every NN module must implement:
//   - Forward: Compute output from input
//   - Parameters: Return all trainable parameters
//
// Modules can be composed to build complex architectures:
//
//	model := nn.Sequential[Backend](
//	    nn.NewLinear(784, 128, backend),
//	    nn.NewReLU(),
//	    nn.NewLinear(128, 10, backend),
//	)
//
// Type parameter B must satisfy the tensor.Backend interface.
type Module[B tensor.Backend] interface {
	// Forward computes the output of the module given an input tensor.
	//
	// The input tensor should have the appropriate shape for this module.
	// For example, Linear expects [batch_size, in_features].
	//
	// Returns the output tensor with shape determined by the module type.
	Forward(input *tensor.Tensor[float32, B]) *tensor.Tensor[float32, B]

	// Parameters returns all trainable parameters of this module.
	//
	// This includes weights, biases, and any nested module parameters.
	// Returns an empty slice for modules without trainable parameters
	// (e.g., activation functions).
	Parameters() []*Parameter[B]
}
