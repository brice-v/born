package nn

import (
	"github.com/born-ml/born/internal/tensor"
)

// Parameter represents a trainable parameter in a neural network.
//
// Parameters are tensors that require gradient computation during training.
// They typically represent weights and biases of layers.
//
// Example:
//
//	// Create a weight parameter
//	weight := nn.NewParameter("weight", weightTensor)
//
//	// Access the tensor
//	w := weight.Tensor()
//
//	// Get gradient after backward pass
//	grad := weight.Grad()
type Parameter[B tensor.Backend] struct {
	name   string                     // Parameter name (e.g., "weight", "bias")
	tensor *tensor.Tensor[float32, B] // The parameter tensor
	grad   *tensor.Tensor[float32, B] // Gradient tensor (computed during backward pass)
}

// NewParameter creates a new trainable parameter.
//
// The parameter tensor should be initialized before creating the Parameter.
// Gradient will be allocated during the first backward pass.
//
// Parameters:
//   - name: Descriptive name for this parameter (e.g., "linear1.weight")
//   - tensor: The initialized parameter tensor
//
// Returns a new Parameter.
func NewParameter[B tensor.Backend](name string, t *tensor.Tensor[float32, B]) *Parameter[B] {
	return &Parameter[B]{
		name:   name,
		tensor: t,
		grad:   nil, // Gradient allocated on first backward pass
	}
}

// Name returns the parameter name.
func (p *Parameter[B]) Name() string {
	return p.name
}

// Tensor returns the parameter tensor.
func (p *Parameter[B]) Tensor() *tensor.Tensor[float32, B] {
	return p.tensor
}

// Grad returns the gradient tensor.
//
// Returns nil if no gradient has been computed yet (before backward pass).
func (p *Parameter[B]) Grad() *tensor.Tensor[float32, B] {
	return p.grad
}

// SetGrad sets the gradient tensor.
//
// This is typically called by the optimizer or during backward pass.
func (p *Parameter[B]) SetGrad(grad *tensor.Tensor[float32, B]) {
	p.grad = grad
}

// ZeroGrad clears the gradient tensor.
//
// This should be called before each training iteration to avoid
// accumulating gradients from previous iterations.
func (p *Parameter[B]) ZeroGrad() {
	p.grad = nil
}
