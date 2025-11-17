package optim

import (
	"github.com/born-ml/born/internal/nn"
	"github.com/born-ml/born/internal/tensor"
)

// SGD implements Stochastic Gradient Descent optimizer with optional momentum.
//
// Update rule without momentum:
//
//	param = param - lr * gradient
//
// Update rule with momentum:
//
//	velocity = momentum * velocity + gradient
//	param = param - lr * velocity
//
// Momentum helps accelerate SGD in relevant directions and dampens oscillations.
//
// Example:
//
//	optimizer := optim.NewSGD(model.Parameters(), optim.SGDConfig{
//	    LR:       0.01,
//	    Momentum: 0.9,
//	})
//
//	for epoch := range epochs {
//	    loss := train_step(model, batch)
//	    grads := autodiff.Backward(loss, backend)
//	    optimizer.Step(grads)
//	    optimizer.ZeroGrad()
//	}
type SGD[B tensor.Backend] struct {
	params     []*nn.Parameter[B]
	lr         float32
	momentum   float32
	velocities map[*nn.Parameter[B]]*tensor.Tensor[float32, B]
	backend    B
}

// SGDConfig holds configuration for SGD optimizer.
type SGDConfig struct {
	LR       float32 // Learning rate (default: 0.01)
	Momentum float32 // Momentum factor (default: 0.0, range: [0, 1))
}

// NewSGD creates a new SGD optimizer.
//
// Parameters:
//   - params: Model parameters to optimize
//   - config: SGD configuration (LR, Momentum)
//
// Returns a new SGD optimizer.
//
// Example:
//
//	sgd := optim.NewSGD(model.Parameters(), optim.SGDConfig{
//	    LR:       0.01,
//	    Momentum: 0.9,
//	})
func NewSGD[B tensor.Backend](params []*nn.Parameter[B], config SGDConfig, backend B) *SGD[B] {
	// Set defaults
	if config.LR == 0 {
		config.LR = 0.01
	}

	return &SGD[B]{
		params:     params,
		lr:         config.LR,
		momentum:   config.Momentum,
		velocities: make(map[*nn.Parameter[B]]*tensor.Tensor[float32, B]),
		backend:    backend,
	}
}

// Step performs a single optimization step.
//
// Applies gradient descent update to all parameters:
//   - Without momentum: param -= lr * grad
//   - With momentum: velocity = momentum * velocity + grad, param -= lr * velocity
//
// Parameters with no gradient (not in computational graph) are skipped.
func (s *SGD[B]) Step(grads map[*tensor.RawTensor]*tensor.RawTensor) {
	for _, param := range s.params {
		// Get gradient for this parameter
		grad := getGradient(param, grads)
		if grad == nil {
			// Parameter didn't participate in forward pass, skip
			continue
		}

		// Convert RawTensor gradient to Tensor for operations
		gradTensor := tensor.New[float32, B](grad, s.backend)

		if s.momentum == 0 {
			// Simple SGD: param -= lr * grad
			s.updateParameter(param, gradTensor)
		} else {
			// SGD with momentum
			s.updateParameterWithMomentum(param, gradTensor)
		}
	}
}

// updateParameter performs simple SGD update without momentum.
func (s *SGD[B]) updateParameter(param *nn.Parameter[B], grad *tensor.Tensor[float32, B]) {
	// param -= lr * grad
	// Scale gradient by learning rate
	lrTensor := tensor.Full[float32](tensor.Shape{1}, s.lr, s.backend)
	scaledGrad := grad.Mul(lrTensor)

	// Update parameter in-place
	updated := param.Tensor().Sub(scaledGrad)

	// Copy updated values back to parameter tensor
	paramData := param.Tensor().Raw().AsFloat32()
	updatedData := updated.Raw().AsFloat32()
	copy(paramData, updatedData)
}

// updateParameterWithMomentum performs SGD update with momentum.
func (s *SGD[B]) updateParameterWithMomentum(param *nn.Parameter[B], grad *tensor.Tensor[float32, B]) {
	// Get or initialize velocity
	velocity, exists := s.velocities[param]
	if !exists {
		// Initialize velocity to zeros with same shape as parameter
		velocity = tensor.Zeros[float32](param.Tensor().Shape(), s.backend)
		s.velocities[param] = velocity
	}

	// velocity = momentum * velocity + grad
	momentumTensor := tensor.Full[float32](tensor.Shape{1}, s.momentum, s.backend)
	velocityScaled := velocity.Mul(momentumTensor)
	newVelocity := velocityScaled.Add(grad)

	// Update velocity
	velocityData := velocity.Raw().AsFloat32()
	newVelocityData := newVelocity.Raw().AsFloat32()
	copy(velocityData, newVelocityData)

	// param -= lr * velocity
	lrTensor := tensor.Full[float32](tensor.Shape{1}, s.lr, s.backend)
	update := velocity.Mul(lrTensor)
	updated := param.Tensor().Sub(update)

	// Copy updated values back to parameter
	paramData := param.Tensor().Raw().AsFloat32()
	updatedData := updated.Raw().AsFloat32()
	copy(paramData, updatedData)
}

// ZeroGrad clears gradients for all parameters.
func (s *SGD[B]) ZeroGrad() {
	for _, param := range s.params {
		param.ZeroGrad()
	}
}

// GetLR returns the current learning rate.
func (s *SGD[B]) GetLR() float32 {
	return s.lr
}

// SetLR updates the learning rate.
//
// Useful for learning rate scheduling during training.
func (s *SGD[B]) SetLR(lr float32) {
	s.lr = lr
}
