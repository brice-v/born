package cpu

import (
	"fmt"

	"github.com/born-ml/born/internal/tensor"
)

// MatMul performs matrix multiplication.
// For 2D tensors: (M, K) @ (K, N) -> (M, N)
// Uses naive O(n³) implementation for Phase 1.
// TODO: Integrate with gonum/blas for better performance in Phase 2.
func (cpu *CPUBackend) MatMul(a, b *tensor.RawTensor) *tensor.RawTensor {
	aShape := a.Shape()
	bShape := b.Shape()

	// Validate dimensions
	if len(aShape) != 2 || len(bShape) != 2 {
		panic(fmt.Sprintf("matmul: only 2D tensors supported, got %dD and %dD", len(aShape), len(bShape)))
	}

	m, k := aShape[0], aShape[1]
	kAlt, n := bShape[0], bShape[1]

	if k != kAlt {
		panic(fmt.Sprintf("matmul: shape mismatch [%d,%d] @ [%d,%d]", m, k, kAlt, n))
	}

	// Create result tensor
	result, err := tensor.NewRaw(tensor.Shape{m, n}, a.DType(), cpu.device)
	if err != nil {
		panic(fmt.Sprintf("matmul: failed to create result tensor: %v", err))
	}

	// Dispatch to type-specific implementation
	switch a.DType() {
	case tensor.Float32:
		matmulFloat32(result.AsFloat32(), a.AsFloat32(), b.AsFloat32(), m, k, n)
	case tensor.Float64:
		matmulFloat64(result.AsFloat64(), a.AsFloat64(), b.AsFloat64(), m, k, n)
	case tensor.Int32:
		matmulInt32(result.AsInt32(), a.AsInt32(), b.AsInt32(), m, k, n)
	case tensor.Int64:
		matmulInt64(result.AsInt64(), a.AsInt64(), b.AsInt64(), m, k, n)
	default:
		panic(fmt.Sprintf("matmul: unsupported dtype %s", a.DType()))
	}

	return result
}

// matmulFloat32 performs naive matrix multiplication for float32.
// C[i,j] = sum_k A[i,k] * B[k,j]
// TODO: Replace with gonum/blas SGEMM for better performance.
func matmulFloat32(c, a, b []float32, m, k, n int) {
	// Initialize C to zero
	for i := range c {
		c[i] = 0
	}

	// Naive O(n³) implementation
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			sum := float32(0)
			for kIdx := 0; kIdx < k; kIdx++ {
				sum += a[i*k+kIdx] * b[kIdx*n+j]
			}
			c[i*n+j] = sum
		}
	}
}

func matmulFloat64(c, a, b []float64, m, k, n int) {
	for i := range c {
		c[i] = 0
	}

	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			sum := float64(0)
			for kIdx := 0; kIdx < k; kIdx++ {
				sum += a[i*k+kIdx] * b[kIdx*n+j]
			}
			c[i*n+j] = sum
		}
	}
}

func matmulInt32(c, a, b []int32, m, k, n int) {
	for i := range c {
		c[i] = 0
	}

	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			sum := int32(0)
			for kIdx := 0; kIdx < k; kIdx++ {
				sum += a[i*k+kIdx] * b[kIdx*n+j]
			}
			c[i*n+j] = sum
		}
	}
}

func matmulInt64(c, a, b []int64, m, k, n int) {
	for i := range c {
		c[i] = 0
	}

	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			sum := int64(0)
			for kIdx := 0; kIdx < k; kIdx++ {
				sum += a[i*k+kIdx] * b[kIdx*n+j]
			}
			c[i*n+j] = sum
		}
	}
}
