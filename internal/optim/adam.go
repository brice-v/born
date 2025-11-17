package optim

import (
	"math"

	"github.com/born-ml/born/internal/nn"
	"github.com/born-ml/born/internal/tensor"
)

// Adam implements the Adam (Adaptive Moment Estimation) optimizer.
//
// Adam combines ideas from RMSprop and momentum:
//   - Maintains exponential moving averages of gradients (first moment)
//   - Maintains exponential moving averages of squared gradients (second moment)
//   - Applies bias correction to compensate for initialization at zero
//
// Update rule:
//
//	m_t = beta1 * m_{t-1} + (1-beta1) * gradient       // First moment
//	v_t = beta2 * v_{t-1} + (1-beta2) * gradient²      // Second moment
//	m_hat = m_t / (1 - beta1^t)                        // Bias correction
//	v_hat = v_t / (1 - beta2^t)                        // Bias correction
//	param = param - lr * m_hat / (sqrt(v_hat) + eps)  // Parameter update
//
// Adam is particularly well-suited for:
//   - Large datasets and high-dimensional parameter spaces
//   - Non-stationary objectives and sparse gradients
//   - Problems with very noisy/sparse gradients
//
// Reference: "Adam: A Method for Stochastic Optimization" (Kingma & Ba, 2014)
//
// Example:
//
//	optimizer := optim.NewAdam(model.Parameters(), optim.AdamConfig{
//	    LR:    0.001,
//	    Betas: [2]float32{0.9, 0.999},
//	    Eps:   1e-8,
//	})
//
//	for epoch := range epochs {
//	    loss := train_step(model, batch)
//	    grads := autodiff.Backward(loss, backend)
//	    optimizer.Step(grads)
//	    optimizer.ZeroGrad()
//	}
type Adam[B tensor.Backend] struct {
	params  []*nn.Parameter[B]
	lr      float32
	beta1   float32
	beta2   float32
	eps     float32
	t       int                                             // Timestep for bias correction
	m       map[*nn.Parameter[B]]*tensor.Tensor[float32, B] // First moment estimates
	v       map[*nn.Parameter[B]]*tensor.Tensor[float32, B] // Second moment estimates
	backend B
}

// AdamConfig holds configuration for Adam optimizer.
type AdamConfig struct {
	LR    float32    // Learning rate (default: 0.001)
	Betas [2]float32 // Coefficients for computing running averages (default: [0.9, 0.999])
	Eps   float32    // Term for numerical stability (default: 1e-8)
}

// NewAdam creates a new Adam optimizer.
//
// Parameters:
//   - params: Model parameters to optimize
//   - config: Adam configuration (LR, Betas, Eps)
//
// Returns a new Adam optimizer with default hyperparameters if not specified.
//
// Default hyperparameters:
//   - LR: 0.001
//   - Beta1: 0.9
//   - Beta2: 0.999
//   - Eps: 1e-8
func NewAdam[B tensor.Backend](params []*nn.Parameter[B], config AdamConfig, backend B) *Adam[B] {
	// Set defaults
	if config.LR == 0 {
		config.LR = 0.001
	}
	if config.Betas[0] == 0 {
		config.Betas[0] = 0.9
	}
	if config.Betas[1] == 0 {
		config.Betas[1] = 0.999
	}
	if config.Eps == 0 {
		config.Eps = 1e-8
	}

	return &Adam[B]{
		params:  params,
		lr:      config.LR,
		beta1:   config.Betas[0],
		beta2:   config.Betas[1],
		eps:     config.Eps,
		t:       0,
		m:       make(map[*nn.Parameter[B]]*tensor.Tensor[float32, B]),
		v:       make(map[*nn.Parameter[B]]*tensor.Tensor[float32, B]),
		backend: backend,
	}
}

// Step performs a single optimization step using Adam algorithm.
//
// Applies Adam update to all parameters:
//  1. Update biased first moment estimate
//  2. Update biased second moment estimate
//  3. Compute bias-corrected moment estimates
//  4. Update parameters
//
// Parameters with no gradient are skipped.
func (a *Adam[B]) Step(grads map[*tensor.RawTensor]*tensor.RawTensor) {
	// Increment timestep
	a.t++

	// Compute bias correction factors
	// bias_correction1 = 1 - beta1^t
	// bias_correction2 = 1 - beta2^t
	biasCorrection1 := float32(1.0 - math.Pow(float64(a.beta1), float64(a.t)))
	biasCorrection2 := float32(1.0 - math.Pow(float64(a.beta2), float64(a.t)))

	for _, param := range a.params {
		// Get gradient for this parameter
		grad := getGradient(param, grads)
		if grad == nil {
			// Parameter didn't participate in forward pass, skip
			continue
		}

		// Convert RawTensor gradient to Tensor
		gradTensor := tensor.New[float32, B](grad, a.backend)

		// Get or initialize first moment (m)
		m, mExists := a.m[param]
		if !mExists {
			m = tensor.Zeros[float32](param.Tensor().Shape(), a.backend)
			a.m[param] = m
		}

		// Get or initialize second moment (v)
		v, vExists := a.v[param]
		if !vExists {
			v = tensor.Zeros[float32](param.Tensor().Shape(), a.backend)
			a.v[param] = v
		}

		// Update moments and parameter
		a.updateParameter(param, gradTensor, m, v, biasCorrection1, biasCorrection2)
	}
}

// updateParameter performs Adam update for a single parameter.
func (a *Adam[B]) updateParameter(
	param *nn.Parameter[B],
	grad *tensor.Tensor[float32, B],
	m, v *tensor.Tensor[float32, B],
	biasCorrection1, biasCorrection2 float32,
) {
	// Get raw data for in-place updates
	gradData := grad.Raw().AsFloat32()
	mData := m.Raw().AsFloat32()
	vData := v.Raw().AsFloat32()
	paramData := param.Tensor().Raw().AsFloat32()

	// Update moments and parameters element-wise
	for i := range paramData {
		g := gradData[i]

		// Update biased first moment estimate
		// m_t = beta1 * m_{t-1} + (1-beta1) * grad
		mData[i] = a.beta1*mData[i] + (1.0-a.beta1)*g

		// Update biased second raw moment estimate
		// v_t = beta2 * v_{t-1} + (1-beta2) * grad²
		vData[i] = a.beta2*vData[i] + (1.0-a.beta2)*g*g

		// Compute bias-corrected first moment estimate
		// m_hat = m_t / (1 - beta1^t)
		mHat := mData[i] / biasCorrection1

		// Compute bias-corrected second raw moment estimate
		// v_hat = v_t / (1 - beta2^t)
		vHat := vData[i] / biasCorrection2

		// Update parameter
		// param = param - lr * m_hat / (sqrt(v_hat) + eps)
		paramData[i] -= a.lr * mHat / (float32(math.Sqrt(float64(vHat))) + a.eps)
	}
}

// ZeroGrad clears gradients for all parameters.
func (a *Adam[B]) ZeroGrad() {
	for _, param := range a.params {
		param.ZeroGrad()
	}
}

// GetLR returns the current learning rate.
func (a *Adam[B]) GetLR() float32 {
	return a.lr
}

// SetLR updates the learning rate.
//
// Useful for learning rate scheduling during training.
func (a *Adam[B]) SetLR(lr float32) {
	a.lr = lr
}

// GetTimestep returns the current timestep.
//
// Useful for monitoring optimizer state.
func (a *Adam[B]) GetTimestep() int {
	return a.t
}
