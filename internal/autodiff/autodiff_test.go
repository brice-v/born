package autodiff_test

import (
	"math"
	"testing"

	"github.com/born-ml/born/internal/autodiff"
	"github.com/born-ml/born/internal/backend/cpu"
	"github.com/born-ml/born/internal/tensor"
)

// TestAutodiffBackend_Name tests the Name method.
func TestAutodiffBackend_Name(t *testing.T) {
	backend := autodiff.New(cpu.New())
	expected := "Autodiff(CPU)"
	if backend.Name() != expected {
		t.Errorf("Name() = %s, want %s", backend.Name(), expected)
	}
}

// TestAutodiffBackend_Device tests the Device method.
func TestAutodiffBackend_Device(t *testing.T) {
	backend := autodiff.New(cpu.New())
	if backend.Device() != tensor.CPU {
		t.Errorf("Device() = %v, want %v", backend.Device(), tensor.CPU)
	}
}

// TestTape_Recording tests tape recording on/off.
func TestTape_Recording(t *testing.T) {
	backend := autodiff.New(cpu.New())
	tape := backend.Tape()

	// Initially not recording
	if tape.IsRecording() {
		t.Error("Tape should not be recording initially")
	}

	// Start recording
	tape.StartRecording()
	if !tape.IsRecording() {
		t.Error("Tape should be recording after StartRecording()")
	}

	// Stop recording
	tape.StopRecording()
	if tape.IsRecording() {
		t.Error("Tape should not be recording after StopRecording()")
	}
}

// TestTape_Clear tests tape clearing.
func TestTape_Clear(t *testing.T) {
	backend := autodiff.New(cpu.New())
	tape := backend.Tape()

	tape.StartRecording()

	// Perform some operations
	a, _ := tensor.FromSlice([]float32{1, 2}, tensor.Shape{2}, backend)
	b, _ := tensor.FromSlice([]float32{3, 4}, tensor.Shape{2}, backend)
	backend.Add(a.Raw(), b.Raw())

	if tape.NumOps() == 0 {
		t.Error("Tape should have recorded operations")
	}

	// Clear tape
	tape.Clear()

	if tape.NumOps() != 0 {
		t.Errorf("Tape should be empty after Clear(), got %d ops", tape.NumOps())
	}

	// Note: Clear() preserves recording state (by design)
	// This allows clearing tape between epochs without stopping recording
	if !tape.IsRecording() {
		t.Error("Tape should still be recording after Clear() (recording state preserved)")
	}
}

// TestAutodiffBackend_Add_RecordsOperation tests that Add records operations.
func TestAutodiffBackend_Add_RecordsOperation(t *testing.T) {
	backend := autodiff.New(cpu.New())
	tape := backend.Tape()

	tape.StartRecording()

	a, _ := tensor.FromSlice([]float32{1, 2}, tensor.Shape{2}, backend)
	b, _ := tensor.FromSlice([]float32{3, 4}, tensor.Shape{2}, backend)

	result := backend.Add(a.Raw(), b.Raw())

	// Verify forward pass
	expected := []float32{4, 6}
	actual := result.AsFloat32()
	for i, v := range expected {
		if actual[i] != v {
			t.Errorf("Add result[%d] = %f, want %f", i, actual[i], v)
		}
	}

	// Verify operation was recorded
	if tape.NumOps() != 1 {
		t.Errorf("Expected 1 operation recorded, got %d", tape.NumOps())
	}
}

// TestAutodiffBackend_Mul_RecordsOperation tests that Mul records operations.
func TestAutodiffBackend_Mul_RecordsOperation(t *testing.T) {
	backend := autodiff.New(cpu.New())
	tape := backend.Tape()

	tape.StartRecording()

	a, _ := tensor.FromSlice([]float32{2, 3}, tensor.Shape{2}, backend)
	b, _ := tensor.FromSlice([]float32{4, 5}, tensor.Shape{2}, backend)

	result := backend.Mul(a.Raw(), b.Raw())

	// Verify forward pass
	expected := []float32{8, 15}
	actual := result.AsFloat32()
	for i, v := range expected {
		if actual[i] != v {
			t.Errorf("Mul result[%d] = %f, want %f", i, actual[i], v)
		}
	}

	// Verify operation was recorded
	if tape.NumOps() != 1 {
		t.Errorf("Expected 1 operation recorded, got %d", tape.NumOps())
	}
}

// TestAutodiffBackend_NoRecording tests that operations are not recorded when tape is off.
func TestAutodiffBackend_NoRecording(t *testing.T) {
	backend := autodiff.New(cpu.New())
	tape := backend.Tape()

	// Don't start recording

	a, _ := tensor.FromSlice([]float32{1, 2}, tensor.Shape{2}, backend)
	b, _ := tensor.FromSlice([]float32{3, 4}, tensor.Shape{2}, backend)

	backend.Add(a.Raw(), b.Raw())

	// Verify no operations were recorded
	if tape.NumOps() != 0 {
		t.Errorf("Expected 0 operations recorded (tape off), got %d", tape.NumOps())
	}
}

// TestBackward_SimpleAddition tests backward pass for simple addition.
func TestBackward_SimpleAddition(t *testing.T) {
	backend := autodiff.New(cpu.New())
	tape := backend.Tape()

	tape.StartRecording()

	// y = a + b
	a, _ := tensor.FromSlice([]float32{2, 3}, tensor.Shape{2}, backend)
	b, _ := tensor.FromSlice([]float32{4, 5}, tensor.Shape{2}, backend)

	resultRaw := backend.Add(a.Raw(), b.Raw())
	result := tensor.New[float32](resultRaw, backend)

	// Compute gradients
	gradients := autodiff.Backward(result, backend)

	// dy/da = 1, dy/db = 1
	gradA := gradients[a.Raw()]
	gradB := gradients[b.Raw()]

	if gradA == nil || gradB == nil {
		t.Fatal("Expected gradients for both inputs")
	}

	expectedGrad := []float32{1, 1}

	actualGradA := gradA.AsFloat32()
	actualGradB := gradB.AsFloat32()

	for i, v := range expectedGrad {
		if actualGradA[i] != v {
			t.Errorf("grad_a[%d] = %f, want %f", i, actualGradA[i], v)
		}
		if actualGradB[i] != v {
			t.Errorf("grad_b[%d] = %f, want %f", i, actualGradB[i], v)
		}
	}
}

// TestBackward_SimpleMultiplication tests backward pass for multiplication.
func TestBackward_SimpleMultiplication(t *testing.T) {
	backend := autodiff.New(cpu.New())
	tape := backend.Tape()

	tape.StartRecording()

	// y = a * b
	a, _ := tensor.FromSlice([]float32{2, 3}, tensor.Shape{2}, backend)
	b, _ := tensor.FromSlice([]float32{4, 5}, tensor.Shape{2}, backend)

	resultRaw := backend.Mul(a.Raw(), b.Raw())
	result := tensor.New[float32](resultRaw, backend)

	// Compute gradients
	gradients := autodiff.Backward(result, backend)

	// dy/da = b, dy/db = a
	gradA := gradients[a.Raw()]
	gradB := gradients[b.Raw()]

	if gradA == nil || gradB == nil {
		t.Fatal("Expected gradients for both inputs")
	}

	expectedGradA := []float32{4, 5} // b values
	expectedGradB := []float32{2, 3} // a values

	actualGradA := gradA.AsFloat32()
	actualGradB := gradB.AsFloat32()

	for i, v := range expectedGradA {
		if actualGradA[i] != v {
			t.Errorf("grad_a[%d] = %f, want %f", i, actualGradA[i], v)
		}
	}

	for i, v := range expectedGradB {
		if actualGradB[i] != v {
			t.Errorf("grad_b[%d] = %f, want %f", i, actualGradB[i], v)
		}
	}
}

// TestBackward_ChainRule tests gradient computation with chain rule.
func TestBackward_ChainRule(t *testing.T) {
	backend := autodiff.New(cpu.New())
	tape := backend.Tape()

	tape.StartRecording()

	// y = (x + 2) * 3
	// dy/dx = 3
	x, _ := tensor.FromSlice([]float32{1}, tensor.Shape{1}, backend)
	two, _ := tensor.FromSlice([]float32{2}, tensor.Shape{1}, backend)
	three, _ := tensor.FromSlice([]float32{3}, tensor.Shape{1}, backend)

	temp := backend.Add(x.Raw(), two.Raw())
	resultRaw := backend.Mul(temp, three.Raw())
	result := tensor.New[float32](resultRaw, backend)

	// Compute gradients
	gradients := autodiff.Backward(result, backend)

	gradX := gradients[x.Raw()]
	if gradX == nil {
		t.Fatal("Expected gradient for x")
	}

	actualGrad := gradX.AsFloat32()[0]
	expectedGrad := float32(3.0)

	if math.Abs(float64(actualGrad-expectedGrad)) > 1e-6 {
		t.Errorf("grad_x = %f, want %f", actualGrad, expectedGrad)
	}
}

// TestBackward_GradientAccumulation tests that gradients accumulate correctly.
func TestBackward_GradientAccumulation(t *testing.T) {
	backend := autodiff.New(cpu.New())
	tape := backend.Tape()

	tape.StartRecording()

	// y = x + x (x used twice)
	// dy/dx = 2
	x, _ := tensor.FromSlice([]float32{3}, tensor.Shape{1}, backend)

	resultRaw := backend.Add(x.Raw(), x.Raw())
	result := tensor.New[float32](resultRaw, backend)

	// Compute gradients
	gradients := autodiff.Backward(result, backend)

	gradX := gradients[x.Raw()]
	if gradX == nil {
		t.Fatal("Expected gradient for x")
	}

	actualGrad := gradX.AsFloat32()[0]
	expectedGrad := float32(2.0)

	if math.Abs(float64(actualGrad-expectedGrad)) > 1e-6 {
		t.Errorf("grad_x = %f, want %f (gradient should accumulate)", actualGrad, expectedGrad)
	}
}

// TestReLU_Forward tests ReLU forward pass.
func TestReLU_Forward(t *testing.T) {
	backend := autodiff.New(cpu.New())

	input, _ := tensor.FromSlice([]float32{-2, -1, 0, 1, 2}, tensor.Shape{5}, backend)

	result := backend.ReLU(input.Raw())

	expected := []float32{0, 0, 0, 1, 2}
	actual := result.AsFloat32()

	for i, v := range expected {
		if actual[i] != v {
			t.Errorf("ReLU result[%d] = %f, want %f", i, actual[i], v)
		}
	}
}

// TestReLU_Backward tests ReLU backward pass.
func TestReLU_Backward(t *testing.T) {
	backend := autodiff.New(cpu.New())
	tape := backend.Tape()

	tape.StartRecording()

	// y = ReLU(x)
	x, _ := tensor.FromSlice([]float32{-2, -1, 0, 1, 2}, tensor.Shape{5}, backend)

	resultRaw := backend.ReLU(x.Raw())
	result := tensor.New[float32](resultRaw, backend)

	// Compute gradients
	gradients := autodiff.Backward(result, backend)

	gradX := gradients[x.Raw()]
	if gradX == nil {
		t.Fatal("Expected gradient for x")
	}

	// dy/dx = 1 if x > 0, else 0
	expected := []float32{0, 0, 0, 1, 1}
	actual := gradX.AsFloat32()

	for i, v := range expected {
		if actual[i] != v {
			t.Errorf("grad_x[%d] = %f, want %f", i, actual[i], v)
		}
	}
}

// TestMatMul_Backward tests MatMul backward pass.
func TestMatMul_Backward(t *testing.T) {
	backend := autodiff.New(cpu.New())
	tape := backend.Tape()

	tape.StartRecording()

	// C = A @ B
	// A: 2x3, B: 3x2 -> C: 2x2
	A, _ := tensor.FromSlice([]float32{
		1, 2, 3,
		4, 5, 6,
	}, tensor.Shape{2, 3}, backend)

	B, _ := tensor.FromSlice([]float32{
		7, 8,
		9, 10,
		11, 12,
	}, tensor.Shape{3, 2}, backend)

	resultRaw := backend.MatMul(A.Raw(), B.Raw())
	result := tensor.New[float32](resultRaw, backend)

	// Compute gradients
	gradients := autodiff.Backward(result, backend)

	gradA := gradients[A.Raw()]
	gradB := gradients[B.Raw()]

	if gradA == nil || gradB == nil {
		t.Fatal("Expected gradients for both matrices")
	}

	// Verify shapes
	if !gradA.Shape().Equal(A.Shape()) {
		t.Errorf("grad_A shape = %v, want %v", gradA.Shape(), A.Shape())
	}
	if !gradB.Shape().Equal(B.Shape()) {
		t.Errorf("grad_B shape = %v, want %v", gradB.Shape(), B.Shape())
	}

	// Gradients should be non-zero
	gradAData := gradA.AsFloat32()
	gradBData := gradB.AsFloat32()

	allZero := true
	for _, v := range gradAData {
		if v != 0 {
			allZero = false
			break
		}
	}
	if allZero {
		t.Error("grad_A should not be all zeros")
	}

	allZero = true
	for _, v := range gradBData {
		if v != 0 {
			allZero = false
			break
		}
	}
	if allZero {
		t.Error("grad_B should not be all zeros")
	}
}

// TestSubtraction_Backward tests Sub backward pass.
func TestSubtraction_Backward(t *testing.T) {
	backend := autodiff.New(cpu.New())
	tape := backend.Tape()

	tape.StartRecording()

	// y = a - b
	a, _ := tensor.FromSlice([]float32{5, 6}, tensor.Shape{2}, backend)
	b, _ := tensor.FromSlice([]float32{2, 3}, tensor.Shape{2}, backend)

	resultRaw := backend.Sub(a.Raw(), b.Raw())
	result := tensor.New[float32](resultRaw, backend)

	// Compute gradients
	gradients := autodiff.Backward(result, backend)

	// dy/da = 1, dy/db = -1
	gradA := gradients[a.Raw()]
	gradB := gradients[b.Raw()]

	if gradA == nil || gradB == nil {
		t.Fatal("Expected gradients for both inputs")
	}

	expectedGradA := []float32{1, 1}
	expectedGradB := []float32{-1, -1}

	actualGradA := gradA.AsFloat32()
	actualGradB := gradB.AsFloat32()

	for i, v := range expectedGradA {
		if actualGradA[i] != v {
			t.Errorf("grad_a[%d] = %f, want %f", i, actualGradA[i], v)
		}
	}

	for i, v := range expectedGradB {
		if math.Abs(float64(actualGradB[i]-v)) > 1e-6 {
			t.Errorf("grad_b[%d] = %f, want %f", i, actualGradB[i], v)
		}
	}
}

// TestDivision_Backward tests Div backward pass.
func TestDivision_Backward(t *testing.T) {
	backend := autodiff.New(cpu.New())
	tape := backend.Tape()

	tape.StartRecording()

	// y = a / b
	a, _ := tensor.FromSlice([]float32{6, 12}, tensor.Shape{2}, backend)
	b, _ := tensor.FromSlice([]float32{2, 3}, tensor.Shape{2}, backend)

	resultRaw := backend.Div(a.Raw(), b.Raw())
	result := tensor.New[float32](resultRaw, backend)

	// Compute gradients
	gradients := autodiff.Backward(result, backend)

	// dy/da = 1/b, dy/db = -a/b²
	gradA := gradients[a.Raw()]
	gradB := gradients[b.Raw()]

	if gradA == nil || gradB == nil {
		t.Fatal("Expected gradients for both inputs")
	}

	// dy/da = 1/b = [1/2, 1/3] = [0.5, 0.333...]
	expectedGradA := []float32{0.5, 1.0 / 3.0}

	// dy/db = -a/b² = [-6/4, -12/9] = [-1.5, -1.333...]
	expectedGradB := []float32{-1.5, -4.0 / 3.0}

	actualGradA := gradA.AsFloat32()
	actualGradB := gradB.AsFloat32()

	for i, v := range expectedGradA {
		if math.Abs(float64(actualGradA[i]-v)) > 1e-5 {
			t.Errorf("grad_a[%d] = %f, want %f", i, actualGradA[i], v)
		}
	}

	for i, v := range expectedGradB {
		if math.Abs(float64(actualGradB[i]-v)) > 1e-5 {
			t.Errorf("grad_b[%d] = %f, want %f", i, actualGradB[i], v)
		}
	}
}

// TestAutodiffBackend_Inner tests the Inner() method.
func TestAutodiffBackend_Inner(t *testing.T) {
	cpuBackend := cpu.New()
	backend := autodiff.New(cpuBackend)

	inner := backend.Inner()
	if inner.Name() != cpuBackend.Name() {
		t.Errorf("Inner().Name() = %s, want %s", inner.Name(), cpuBackend.Name())
	}
}

// TestReLU_Forward_Float64 tests ReLU forward pass with float64.
func TestReLU_Forward_Float64(t *testing.T) {
	backend := autodiff.New(cpu.New())

	input, _ := tensor.FromSlice([]float64{-2.5, -1.0, 0.0, 1.5, 2.0}, tensor.Shape{5}, backend)

	result := backend.ReLU(input.Raw())

	expected := []float64{0, 0, 0, 1.5, 2.0}
	actual := result.AsFloat64()

	for i, v := range expected {
		if actual[i] != v {
			t.Errorf("ReLU float64 result[%d] = %f, want %f", i, actual[i], v)
		}
	}
}

// TestReLU_Backward_Float64 tests ReLU backward pass with float64.
func TestReLU_Backward_Float64(t *testing.T) {
	backend := autodiff.New(cpu.New())
	tape := backend.Tape()

	tape.StartRecording()

	// y = ReLU(x)
	x, _ := tensor.FromSlice([]float64{-2, -1, 0, 1, 2}, tensor.Shape{5}, backend)

	resultRaw := backend.ReLU(x.Raw())
	result := tensor.New[float64](resultRaw, backend)

	// Compute gradients
	gradients := autodiff.Backward(result, backend)

	gradX := gradients[x.Raw()]
	if gradX == nil {
		t.Fatal("Expected gradient for x")
	}

	// dy/dx = 1 if x > 0, else 0
	expected := []float64{0, 0, 0, 1, 1}
	actual := gradX.AsFloat64()

	for i, v := range expected {
		if actual[i] != v {
			t.Errorf("grad_x float64[%d] = %f, want %f", i, actual[i], v)
		}
	}
}

// TestBackward_Float64 tests backward pass with float64 operations.
func TestBackward_Float64(t *testing.T) {
	backend := autodiff.New(cpu.New())
	tape := backend.Tape()

	tape.StartRecording()

	// y = a * b
	a, _ := tensor.FromSlice([]float64{2.5, 3.5}, tensor.Shape{2}, backend)
	b, _ := tensor.FromSlice([]float64{4.0, 5.0}, tensor.Shape{2}, backend)

	resultRaw := backend.Mul(a.Raw(), b.Raw())
	result := tensor.New[float64](resultRaw, backend)

	// Compute gradients
	gradients := autodiff.Backward(result, backend)

	// dy/da = b, dy/db = a
	gradA := gradients[a.Raw()]
	gradB := gradients[b.Raw()]

	if gradA == nil || gradB == nil {
		t.Fatal("Expected gradients for both inputs")
	}

	expectedGradA := []float64{4.0, 5.0} // b values
	expectedGradB := []float64{2.5, 3.5} // a values

	actualGradA := gradA.AsFloat64()
	actualGradB := gradB.AsFloat64()

	for i, v := range expectedGradA {
		if actualGradA[i] != v {
			t.Errorf("grad_a float64[%d] = %f, want %f", i, actualGradA[i], v)
		}
	}

	for i, v := range expectedGradB {
		if actualGradB[i] != v {
			t.Errorf("grad_b float64[%d] = %f, want %f", i, actualGradB[i], v)
		}
	}
}
