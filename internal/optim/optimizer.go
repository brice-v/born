// Package optim implements optimization algorithms for training neural networks.
//
// This package provides:
//   - Optimizer interface: Base interface for all optimizers
//   - SGD: Stochastic Gradient Descent with momentum
//   - Adam: Adaptive Moment Estimation
//
// Design inspired by PyTorch's torch.optim but adapted for Go with type safety.
//
// Example usage:
//
//	// Create optimizer
//	optimizer := optim.NewAdam(model.Parameters(), optim.AdamConfig{
//	    LR: 0.001,
//	})
//
//	// Training loop
//	for epoch := range epochs {
//	    loss := computeLoss(model, data)
//
//	    // Compute gradients
//	    backend.Tape().StartRecording()
//	    output := model.Forward(input)
//	    loss := lossFunc.Forward(output, targets)
//	    grads := autodiff.Backward(loss, backend)
//
//	    // Update parameters
//	    optimizer.Step(grads)
//	    optimizer.ZeroGrad()
//	}
package optim

import (
	"github.com/born-ml/born/internal/nn"
	"github.com/born-ml/born/internal/tensor"
)

// Optimizer is the base interface for all optimization algorithms.
//
// Optimizers update model parameters based on computed gradients to
// minimize the loss function during training.
//
// All optimizers must implement:
//   - Step: Apply gradient updates to parameters
//   - ZeroGrad: Clear gradients before next iteration
//   - GetLR: Get current learning rate (for monitoring/scheduling)
type Optimizer interface {
	// Step applies gradient updates to all parameters.
	//
	// Takes a gradient map from Backward() and updates parameters in-place.
	// The gradient map should contain RawTensor -> gradient mapping.
	//
	// Example:
	//   grads := autodiff.Backward(loss, backend)
	//   optimizer.Step(grads)
	Step(grads map[*tensor.RawTensor]*tensor.RawTensor)

	// ZeroGrad clears all parameter gradients.
	//
	// This should be called before each backward pass to prevent
	// gradient accumulation from previous iterations.
	//
	// Example:
	//   optimizer.ZeroGrad()
	//   loss := model.Forward(...)
	//   grads := autodiff.Backward(loss, backend)
	ZeroGrad()

	// GetLR returns the current learning rate.
	//
	// Useful for monitoring and learning rate scheduling.
	GetLR() float32
}

// Config is the base configuration for all optimizers.
type Config struct {
	LR float32 // Learning rate
}

// getGradient safely retrieves gradient for a parameter.
//
// Returns nil if no gradient is found (parameter wasn't part of computation graph).
func getGradient[B tensor.Backend](param *nn.Parameter[B], grads map[*tensor.RawTensor]*tensor.RawTensor) *tensor.RawTensor {
	if param == nil {
		return nil
	}
	return grads[param.Tensor().Raw()]
}
