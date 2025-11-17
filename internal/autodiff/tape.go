package autodiff

import (
	"github.com/born-ml/born/internal/autodiff/ops"
	"github.com/born-ml/born/internal/tensor"
)

// GradientTape records operations during the forward pass and computes
// gradients during the backward pass using reverse-mode automatic differentiation.
//
// Usage:
//
//	tape := NewGradientTape()
//	tape.StartRecording()
//	// ... perform operations ...
//	gradients := tape.Backward(outputGrad, backend)
type GradientTape struct {
	operations []ops.Operation // Recorded operations (in execution order)
	recording  bool            // Whether tape is currently recording
}

// NewGradientTape creates a new gradient tape.
func NewGradientTape() *GradientTape {
	return &GradientTape{
		operations: make([]ops.Operation, 0, 64), // Pre-allocate for common case
		recording:  false,
	}
}

// StartRecording enables operation recording.
func (t *GradientTape) StartRecording() {
	t.recording = true
}

// StopRecording disables operation recording.
func (t *GradientTape) StopRecording() {
	t.recording = false
}

// IsRecording returns true if the tape is currently recording operations.
func (t *GradientTape) IsRecording() bool {
	return t.recording
}

// Record adds an operation to the tape.
// Only records if the tape is currently recording.
func (t *GradientTape) Record(op ops.Operation) {
	if t.recording {
		t.operations = append(t.operations, op)
	}
}

// Clear resets the tape, removing all recorded operations.
// Recording state is preserved.
func (t *GradientTape) Clear() {
	t.operations = t.operations[:0]
	// Note: recording state is preserved, call StopRecording() explicitly if needed
}

// Backward computes gradients for all inputs by walking the tape in reverse.
//
// Algorithm:
//  1. Start with the output gradient (typically ones for scalar loss)
//  2. Walk operations in reverse order
//  3. For each operation, compute input gradients using chain rule
//  4. Accumulate gradients when the same tensor is used multiple times
//
// Returns a map from RawTensor to its accumulated gradient.
//
// Example:
//
//	// y = (x + 2) * 3
//	// dy/dx = 3
//	tape := NewGradientTape()
//	tape.StartRecording()
//	// ... operations recorded ...
//	gradients := tape.Backward(ones, backend)
//	dydx := gradients[x.Raw()]
func (t *GradientTape) Backward(outputGrad *tensor.RawTensor, backend tensor.Backend) map[*tensor.RawTensor]*tensor.RawTensor {
	if len(t.operations) == 0 {
		return make(map[*tensor.RawTensor]*tensor.RawTensor)
	}

	// Stop recording during backward pass to prevent recording gradient operations
	wasRecording := t.recording
	t.recording = false
	defer func() {
		t.recording = wasRecording
	}()

	// Map to accumulate gradients for each tensor
	grads := make(map[*tensor.RawTensor]*tensor.RawTensor)

	// Initialize with output gradient
	lastOp := t.operations[len(t.operations)-1]
	grads[lastOp.Output()] = outputGrad

	// Walk tape backwards
	for i := len(t.operations) - 1; i >= 0; i-- {
		op := t.operations[i]
		opOutput := op.Output()

		// Get gradient for this operation's output
		opOutputGrad, hasGrad := grads[opOutput]
		if !hasGrad {
			// No gradient flows to this operation (e.g., unused result)
			continue
		}

		// Compute gradients for inputs
		inputGrads := op.Backward(opOutputGrad, backend)

		// Accumulate gradients for each input
		for j, input := range op.Inputs() {
			inputGrad := inputGrads[j]

			if existing, ok := grads[input]; ok {
				// Accumulate: grad += inputGrad
				grads[input] = backend.Add(existing, inputGrad)
			} else {
				// First gradient for this input
				grads[input] = inputGrad
			}
		}
	}

	return grads
}

// NumOps returns the number of recorded operations.
func (t *GradientTape) NumOps() int {
	return len(t.operations)
}
