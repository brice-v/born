package nn

import (
	"github.com/born-ml/born/internal/tensor"
)

// Sequential is a container module that chains multiple modules together.
//
// Each module's output becomes the next module's input, creating a
// sequential pipeline of transformations.
//
// Example:
//
//	model := nn.NewSequential(
//	    nn.NewLinear(784, 128, backend),
//	    nn.NewReLU(),
//	    nn.NewLinear(128, 10, backend),
//	)
//
//	output := model.Forward(input)
//
// This is equivalent to:
//
//	h1 := linear1.Forward(input)
//	h2 := relu.Forward(h1)
//	output := linear2.Forward(h2)
type Sequential[B tensor.Backend] struct {
	modules []Module[B]
}

// NewSequential creates a new Sequential container.
//
// Parameters:
//   - modules: List of modules to chain together
//
// Returns a new Sequential container.
func NewSequential[B tensor.Backend](modules ...Module[B]) *Sequential[B] {
	return &Sequential[B]{
		modules: modules,
	}
}

// Forward applies all modules in sequence.
//
// The output of each module becomes the input to the next module.
//
// Parameters:
//   - input: Input tensor to the first module
//
// Returns the output of the last module.
func (s *Sequential[B]) Forward(input *tensor.Tensor[float32, B]) *tensor.Tensor[float32, B] {
	output := input

	for _, module := range s.modules {
		output = module.Forward(output)
	}

	return output
}

// Parameters returns all trainable parameters from all modules.
//
// Parameters are collected from all modules in the sequence.
func (s *Sequential[B]) Parameters() []*Parameter[B] {
	var params []*Parameter[B]

	for _, module := range s.modules {
		params = append(params, module.Parameters()...)
	}

	return params
}

// Add appends a module to the sequence.
//
// This allows building models incrementally:
//
//	model := nn.NewSequential[Backend]()
//	model.Add(nn.NewLinear(784, 128, backend))
//	model.Add(nn.NewReLU())
//	model.Add(nn.NewLinear(128, 10, backend))
func (s *Sequential[B]) Add(module Module[B]) {
	s.modules = append(s.modules, module)
}

// Len returns the number of modules in the sequence.
func (s *Sequential[B]) Len() int {
	return len(s.modules)
}

// Module returns the module at the given index.
//
// Panics if index is out of bounds.
func (s *Sequential[B]) Module(index int) Module[B] {
	if index < 0 || index >= len(s.modules) {
		panic("Sequential.Module: index out of bounds")
	}
	return s.modules[index]
}
