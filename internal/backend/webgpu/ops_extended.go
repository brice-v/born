//go:build windows

package webgpu

import (
	"github.com/born-ml/born/internal/tensor"
)

// Extended operations - stubs for now (not yet implemented on GPU).

// Scalar operations.

// MulScalar multiplies tensor elements by a scalar (not yet implemented).
func (b *Backend) MulScalar(_ *tensor.RawTensor, _ any) *tensor.RawTensor {
	panic("webgpu: MulScalar not implemented yet")
}

// AddScalar adds a scalar to tensor elements (not yet implemented).
func (b *Backend) AddScalar(_ *tensor.RawTensor, _ any) *tensor.RawTensor {
	panic("webgpu: AddScalar not implemented yet")
}

// SubScalar subtracts a scalar from tensor elements (not yet implemented).
func (b *Backend) SubScalar(_ *tensor.RawTensor, _ any) *tensor.RawTensor {
	panic("webgpu: SubScalar not implemented yet")
}

// DivScalar divides tensor elements by a scalar (not yet implemented).
func (b *Backend) DivScalar(_ *tensor.RawTensor, _ any) *tensor.RawTensor {
	panic("webgpu: DivScalar not implemented yet")
}

// Math operations.

// Log computes natural logarithm element-wise (not yet implemented).
func (b *Backend) Log(_ *tensor.RawTensor) *tensor.RawTensor {
	panic("webgpu: Log not implemented yet")
}

// Activation functions.

// Softmax applies softmax along the specified dimension.
// Currently supports 2D tensors with dim=-1 (last dimension).
func (b *Backend) Softmax(x *tensor.RawTensor, dim int) *tensor.RawTensor {
	// Normalize negative dim
	ndim := len(x.Shape())
	if dim < 0 {
		dim = ndim + dim
	}

	// Current WebGPU implementation only supports 2D with dim=1 (last dimension)
	if ndim != 2 {
		panic("webgpu: Softmax currently only supports 2D tensors")
	}
	if dim != 1 {
		panic("webgpu: Softmax currently only supports dim=-1 (last dimension)")
	}

	result, err := b.runSoftmax(x)
	if err != nil {
		panic("webgpu: Softmax: " + err.Error())
	}
	return result
}

// Comparison operations.

// Greater performs element-wise greater-than comparison (not yet implemented).
func (b *Backend) Greater(_, _ *tensor.RawTensor) *tensor.RawTensor {
	panic("webgpu: Greater not implemented yet")
}

// Lower performs element-wise less-than comparison (not yet implemented).
func (b *Backend) Lower(_, _ *tensor.RawTensor) *tensor.RawTensor {
	panic("webgpu: Lower not implemented yet")
}

// GreaterEqual performs element-wise greater-or-equal comparison (not yet implemented).
func (b *Backend) GreaterEqual(_, _ *tensor.RawTensor) *tensor.RawTensor {
	panic("webgpu: GreaterEqual not implemented yet")
}

// LowerEqual performs element-wise less-or-equal comparison (not yet implemented).
func (b *Backend) LowerEqual(_, _ *tensor.RawTensor) *tensor.RawTensor {
	panic("webgpu: LowerEqual not implemented yet")
}

// Equal performs element-wise equality comparison (not yet implemented).
func (b *Backend) Equal(_, _ *tensor.RawTensor) *tensor.RawTensor {
	panic("webgpu: Equal not implemented yet")
}

// NotEqual performs element-wise inequality comparison (not yet implemented).
func (b *Backend) NotEqual(_, _ *tensor.RawTensor) *tensor.RawTensor {
	panic("webgpu: NotEqual not implemented yet")
}

// Boolean operations.

// Or performs element-wise logical OR (not yet implemented).
func (b *Backend) Or(_, _ *tensor.RawTensor) *tensor.RawTensor {
	panic("webgpu: Or not implemented yet")
}

// And performs element-wise logical AND (not yet implemented).
func (b *Backend) And(_, _ *tensor.RawTensor) *tensor.RawTensor {
	panic("webgpu: And not implemented yet")
}

// Not performs element-wise logical NOT (not yet implemented).
func (b *Backend) Not(_ *tensor.RawTensor) *tensor.RawTensor {
	panic("webgpu: Not not implemented yet")
}

// Reduction operations.

// Sum computes the sum of all elements (not yet implemented).
func (b *Backend) Sum(_ *tensor.RawTensor) *tensor.RawTensor {
	panic("webgpu: Sum not implemented yet")
}

// Argmax returns indices of maximum values along dimension (not yet implemented).
func (b *Backend) Argmax(_ *tensor.RawTensor, _ int) *tensor.RawTensor {
	panic("webgpu: Argmax not implemented yet")
}

// Shape operations.

// Expand broadcasts tensor to new shape (not yet implemented).
func (b *Backend) Expand(_ *tensor.RawTensor, _ tensor.Shape) *tensor.RawTensor {
	panic("webgpu: Expand not implemented yet")
}

// Type conversion.

// Cast converts tensor to different data type (not yet implemented).
func (b *Backend) Cast(_ *tensor.RawTensor, _ tensor.DataType) *tensor.RawTensor {
	panic("webgpu: Cast not implemented yet")
}
