package nn_test

import (
	"testing"

	"github.com/born-ml/born/internal/backend/cpu"
	"github.com/born-ml/born/internal/nn"
	"github.com/born-ml/born/internal/tensor"
)

func TestEmbedding_Forward_Basic(t *testing.T) {
	backend := cpu.New()

	// Create embedding: 5 embeddings, dimension 3
	embed := nn.NewEmbedding[*cpu.CPUBackend](5, 3, backend)

	// Set known weights for testing
	weightData := []float32{
		1.0, 2.0, 3.0, // embedding 0
		4.0, 5.0, 6.0, // embedding 1
		7.0, 8.0, 9.0, // embedding 2
		10.0, 11.0, 12.0, // embedding 3
		13.0, 14.0, 15.0, // embedding 4
	}
	weight, err := tensor.FromSlice[float32](weightData, tensor.Shape{5, 3}, backend)
	if err != nil {
		t.Fatalf("Failed to create weight: %v", err)
	}
	embed.Weight.SetGrad(nil) // Clear any existing gradient
	embed.Weight = nn.NewParameter("weight", weight)

	// Lookup indices [0, 1, 2]
	indices, err := tensor.FromSlice[int32]([]int32{0, 1, 2}, tensor.Shape{3}, backend)
	if err != nil {
		t.Fatalf("Failed to create indices: %v", err)
	}

	// Forward pass
	output := embed.Forward(indices)

	// Check shape: [3, 3]
	expectedShape := tensor.Shape{3, 3}
	if !shapesEqual(output.Shape(), expectedShape) {
		t.Errorf("Expected shape %v, got %v", expectedShape, output.Shape())
	}

	// Check values
	expected := []float32{
		1.0, 2.0, 3.0, // embedding 0
		4.0, 5.0, 6.0, // embedding 1
		7.0, 8.0, 9.0, // embedding 2
	}
	actual := output.Data()
	if !slicesAlmostEqual(actual, expected, 1e-6) {
		t.Errorf("Expected %v, got %v", expected, actual)
	}
}

func TestEmbedding_Forward_Batched(t *testing.T) {
	backend := cpu.New()

	// Create embedding: 10 embeddings, dimension 4
	embed := nn.NewEmbedding[*cpu.CPUBackend](10, 4, backend)

	// Set known weights
	weightData := make([]float32, 10*4)
	for i := range weightData {
		weightData[i] = float32(i)
	}
	weight, err := tensor.FromSlice[float32](weightData, tensor.Shape{10, 4}, backend)
	if err != nil {
		t.Fatalf("Failed to create weight: %v", err)
	}
	embed.Weight = nn.NewParameter("weight", weight)

	// Batched indices [2, 3]: batch_size=2, seq_len=3
	indices, err := tensor.FromSlice[int32](
		[]int32{0, 1, 2, 3, 4, 5},
		tensor.Shape{2, 3},
		backend,
	)
	if err != nil {
		t.Fatalf("Failed to create indices: %v", err)
	}

	// Forward pass
	output := embed.Forward(indices)

	// Check shape: [2, 3, 4]
	expectedShape := tensor.Shape{2, 3, 4}
	if !shapesEqual(output.Shape(), expectedShape) {
		t.Errorf("Expected shape %v, got %v", expectedShape, output.Shape())
	}

	// Check first batch, first token (index 0)
	outData := output.Data()
	expected0 := []float32{0, 1, 2, 3}
	actual0 := outData[0:4]
	if !slicesAlmostEqual(actual0, expected0, 1e-6) {
		t.Errorf("Expected first embedding %v, got %v", expected0, actual0)
	}

	// Check second batch, third token (index 5)
	expected5 := []float32{20, 21, 22, 23} // embedding 5: indices 20-23
	actual5 := outData[20:24]              // offset: (1*3 + 2) * 4 = 20
	if !slicesAlmostEqual(actual5, expected5, 1e-6) {
		t.Errorf("Expected embedding 5: %v, got %v", expected5, actual5)
	}
}

func TestEmbedding_Forward_OutOfBounds(t *testing.T) {
	backend := cpu.New()

	embed := nn.NewEmbedding[*cpu.CPUBackend](5, 3, backend)

	tests := []struct {
		name    string
		indices []int32
		shape   tensor.Shape
	}{
		{"negative index", []int32{-1}, tensor.Shape{1}},
		{"index too large", []int32{5}, tensor.Shape{1}},
		{"mixed valid and invalid", []int32{0, 1, 10}, tensor.Shape{3}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			indices, err := tensor.FromSlice[int32](tt.indices, tt.shape, backend)
			if err != nil {
				t.Fatalf("Failed to create indices: %v", err)
			}

			// Should panic
			defer func() {
				if r := recover(); r == nil {
					t.Errorf("Expected panic for out of bounds index")
				}
			}()

			embed.Forward(indices)
		})
	}
}

func TestEmbedding_Forward_RepeatedIndices(t *testing.T) {
	backend := cpu.New()

	embed := nn.NewEmbedding[*cpu.CPUBackend](3, 2, backend)

	// Set known weights
	weightData := []float32{
		1.0, 2.0, // embedding 0
		3.0, 4.0, // embedding 1
		5.0, 6.0, // embedding 2
	}
	weight, err := tensor.FromSlice[float32](weightData, tensor.Shape{3, 2}, backend)
	if err != nil {
		t.Fatalf("Failed to create weight: %v", err)
	}
	embed.Weight = nn.NewParameter("weight", weight)

	// Repeated indices: [0, 1, 0, 2, 1]
	indices, err := tensor.FromSlice[int32]([]int32{0, 1, 0, 2, 1}, tensor.Shape{5}, backend)
	if err != nil {
		t.Fatalf("Failed to create indices: %v", err)
	}

	output := embed.Forward(indices)

	// Check shape: [5, 2]
	expectedShape := tensor.Shape{5, 2}
	if !shapesEqual(output.Shape(), expectedShape) {
		t.Errorf("Expected shape %v, got %v", expectedShape, output.Shape())
	}

	// Check values: each index should get correct embedding
	expected := []float32{
		1.0, 2.0, // index 0
		3.0, 4.0, // index 1
		1.0, 2.0, // index 0 (repeated)
		5.0, 6.0, // index 2
		3.0, 4.0, // index 1 (repeated)
	}
	actual := output.Data()
	if !slicesAlmostEqual(actual, expected, 1e-6) {
		t.Errorf("Expected %v, got %v", expected, actual)
	}
}

func TestEmbedding_Forward_DifferentShapes(t *testing.T) {
	backend := cpu.New()

	embed := nn.NewEmbedding[*cpu.CPUBackend](10, 5, backend)

	tests := []struct {
		name          string
		indicesShape  tensor.Shape
		expectedShape tensor.Shape
	}{
		{"1D indices", tensor.Shape{3}, tensor.Shape{3, 5}},
		{"2D indices", tensor.Shape{2, 4}, tensor.Shape{2, 4, 5}},
		{"3D indices", tensor.Shape{2, 3, 4}, tensor.Shape{2, 3, 4, 5}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create indices with the specified shape
			numIndices := tt.indicesShape.NumElements()
			indicesData := make([]int32, numIndices)
			for i := range indicesData {
				indicesData[i] = int32(i % 10) // Valid indices
			}

			indices, err := tensor.FromSlice[int32](indicesData, tt.indicesShape, backend)
			if err != nil {
				t.Fatalf("Failed to create indices: %v", err)
			}

			output := embed.Forward(indices)

			if !shapesEqual(output.Shape(), tt.expectedShape) {
				t.Errorf("Expected shape %v, got %v", tt.expectedShape, output.Shape())
			}
		})
	}
}

func TestEmbedding_Parameters(t *testing.T) {
	backend := cpu.New()

	embed := nn.NewEmbedding[*cpu.CPUBackend](100, 50, backend)

	params := embed.Parameters()
	if len(params) != 1 {
		t.Errorf("Expected 1 parameter, got %d", len(params))
	}

	if params[0] != embed.Weight {
		t.Errorf("Expected parameter to be weight")
	}

	// Check weight shape
	expectedShape := tensor.Shape{100, 50}
	if !shapesEqual(embed.Weight.Tensor().Shape(), expectedShape) {
		t.Errorf("Expected weight shape %v, got %v", expectedShape, embed.Weight.Tensor().Shape())
	}
}

func TestNewEmbeddingWithWeight(t *testing.T) {
	backend := cpu.New()

	// Create custom weight
	weightData := []float32{
		0.1, 0.2, 0.3,
		0.4, 0.5, 0.6,
		0.7, 0.8, 0.9,
	}
	weight, err := tensor.FromSlice[float32](weightData, tensor.Shape{3, 3}, backend)
	if err != nil {
		t.Fatalf("Failed to create weight: %v", err)
	}

	// Create embedding with custom weight
	embed := nn.NewEmbeddingWithWeight(weight)

	if embed.NumEmbed != 3 {
		t.Errorf("Expected NumEmbed=3, got %d", embed.NumEmbed)
	}
	if embed.EmbedDim != 3 {
		t.Errorf("Expected EmbedDim=3, got %d", embed.EmbedDim)
	}

	// Check forward pass uses the custom weight
	indices, err := tensor.FromSlice[int32]([]int32{0, 1, 2}, tensor.Shape{3}, backend)
	if err != nil {
		t.Fatalf("Failed to create indices: %v", err)
	}

	output := embed.Forward(indices)
	actual := output.Data()
	if !slicesAlmostEqual(actual, weightData, 1e-6) {
		t.Errorf("Expected output %v, got %v", weightData, actual)
	}
}

func TestNewEmbeddingWithWeight_InvalidShape(t *testing.T) {
	backend := cpu.New()

	// Create 1D weight (invalid)
	weightData := []float32{1.0, 2.0, 3.0}
	weight, err := tensor.FromSlice[float32](weightData, tensor.Shape{3}, backend)
	if err != nil {
		t.Fatalf("Failed to create weight: %v", err)
	}

	// Should panic
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic for 1D weight")
		}
	}()

	nn.NewEmbeddingWithWeight(weight)
}

// Helper functions.

func shapesEqual(a, b tensor.Shape) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

//nolint:unparam // tolerance parameter allows flexible comparison in future tests
func slicesAlmostEqual(a, b []float32, tolerance float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		diff := a[i] - b[i]
		if diff < 0 {
			diff = -diff
		}
		if diff > tolerance {
			return false
		}
	}
	return true
}
