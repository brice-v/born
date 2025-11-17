package cpu

import (
	"github.com/born-ml/born/internal/tensor"
)

// Int32 operations

func addInplaceInt32(a, b []int32) {
	for i := range a {
		a[i] += b[i]
	}
}

func subInplaceInt32(a, b []int32) {
	for i := range a {
		a[i] -= b[i]
	}
}

func mulInplaceInt32(a, b []int32) {
	for i := range a {
		a[i] *= b[i]
	}
}

func divInplaceInt32(a, b []int32) {
	for i := range a {
		a[i] /= b[i]
	}
}

func addVectorizedInt32(dst, a, b []int32) {
	for i := range a {
		dst[i] = a[i] + b[i]
	}
}

func subVectorizedInt32(dst, a, b []int32) {
	for i := range a {
		dst[i] = a[i] - b[i]
	}
}

func mulVectorizedInt32(dst, a, b []int32) {
	for i := range a {
		dst[i] = a[i] * b[i]
	}
}

func divVectorizedInt32(dst, a, b []int32) {
	for i := range a {
		dst[i] = a[i] / b[i]
	}
}

func addBroadcastInt32(dst, a, b []int32, aShape, bShape, outShape tensor.Shape) {
	outStrides := outShape.ComputeStrides()
	aStrides := computeBroadcastStridesForShape(aShape, outShape)
	bStrides := computeBroadcastStridesForShape(bShape, outShape)

	n := outShape.NumElements()
	for i := 0; i < n; i++ {
		aIdx := computeFlatIndex(i, outStrides, aStrides)
		bIdx := computeFlatIndex(i, outStrides, bStrides)
		dst[i] = a[aIdx] + b[bIdx]
	}
}

func subBroadcastInt32(dst, a, b []int32, aShape, bShape, outShape tensor.Shape) {
	outStrides := outShape.ComputeStrides()
	aStrides := computeBroadcastStridesForShape(aShape, outShape)
	bStrides := computeBroadcastStridesForShape(bShape, outShape)

	n := outShape.NumElements()
	for i := 0; i < n; i++ {
		aIdx := computeFlatIndex(i, outStrides, aStrides)
		bIdx := computeFlatIndex(i, outStrides, bStrides)
		dst[i] = a[aIdx] - b[bIdx]
	}
}

func mulBroadcastInt32(dst, a, b []int32, aShape, bShape, outShape tensor.Shape) {
	outStrides := outShape.ComputeStrides()
	aStrides := computeBroadcastStridesForShape(aShape, outShape)
	bStrides := computeBroadcastStridesForShape(bShape, outShape)

	n := outShape.NumElements()
	for i := 0; i < n; i++ {
		aIdx := computeFlatIndex(i, outStrides, aStrides)
		bIdx := computeFlatIndex(i, outStrides, bStrides)
		dst[i] = a[aIdx] * b[bIdx]
	}
}

func divBroadcastInt32(dst, a, b []int32, aShape, bShape, outShape tensor.Shape) {
	outStrides := outShape.ComputeStrides()
	aStrides := computeBroadcastStridesForShape(aShape, outShape)
	bStrides := computeBroadcastStridesForShape(bShape, outShape)

	n := outShape.NumElements()
	for i := 0; i < n; i++ {
		aIdx := computeFlatIndex(i, outStrides, aStrides)
		bIdx := computeFlatIndex(i, outStrides, bStrides)
		dst[i] = a[aIdx] / b[bIdx]
	}
}

func transposeInt32(dst, src []int32, shape tensor.Shape, axes []int) {
	ndim := len(shape)
	srcStrides := shape.ComputeStrides()

	// Compute destination shape and strides
	dstShape := make(tensor.Shape, ndim)
	for i, ax := range axes {
		dstShape[i] = shape[ax]
	}
	dstStrides := dstShape.ComputeStrides()

	// Transpose data
	n := shape.NumElements()
	for i := 0; i < n; i++ {
		// Compute multi-dimensional coordinates in source
		coords := make([]int, ndim)
		idx := i
		for dim := 0; dim < ndim; dim++ {
			coords[dim] = idx / srcStrides[dim]
			idx %= srcStrides[dim]
		}

		// Permute coordinates according to axes
		permutedCoords := make([]int, ndim)
		for dstDim, srcDim := range axes {
			permutedCoords[dstDim] = coords[srcDim]
		}

		// Compute flat index in destination
		dstIdx := 0
		for dim := 0; dim < ndim; dim++ {
			dstIdx += permutedCoords[dim] * dstStrides[dim]
		}

		dst[dstIdx] = src[i]
	}
}

// Int64 operations

func addInplaceInt64(a, b []int64) {
	for i := range a {
		a[i] += b[i]
	}
}

func subInplaceInt64(a, b []int64) {
	for i := range a {
		a[i] -= b[i]
	}
}

func mulInplaceInt64(a, b []int64) {
	for i := range a {
		a[i] *= b[i]
	}
}

func divInplaceInt64(a, b []int64) {
	for i := range a {
		a[i] /= b[i]
	}
}

func addVectorizedInt64(dst, a, b []int64) {
	for i := range a {
		dst[i] = a[i] + b[i]
	}
}

func subVectorizedInt64(dst, a, b []int64) {
	for i := range a {
		dst[i] = a[i] - b[i]
	}
}

func mulVectorizedInt64(dst, a, b []int64) {
	for i := range a {
		dst[i] = a[i] * b[i]
	}
}

func divVectorizedInt64(dst, a, b []int64) {
	for i := range a {
		dst[i] = a[i] / b[i]
	}
}

func addBroadcastInt64(dst, a, b []int64, aShape, bShape, outShape tensor.Shape) {
	outStrides := outShape.ComputeStrides()
	aStrides := computeBroadcastStridesForShape(aShape, outShape)
	bStrides := computeBroadcastStridesForShape(bShape, outShape)

	n := outShape.NumElements()
	for i := 0; i < n; i++ {
		aIdx := computeFlatIndex(i, outStrides, aStrides)
		bIdx := computeFlatIndex(i, outStrides, bStrides)
		dst[i] = a[aIdx] + b[bIdx]
	}
}

func subBroadcastInt64(dst, a, b []int64, aShape, bShape, outShape tensor.Shape) {
	outStrides := outShape.ComputeStrides()
	aStrides := computeBroadcastStridesForShape(aShape, outShape)
	bStrides := computeBroadcastStridesForShape(bShape, outShape)

	n := outShape.NumElements()
	for i := 0; i < n; i++ {
		aIdx := computeFlatIndex(i, outStrides, aStrides)
		bIdx := computeFlatIndex(i, outStrides, bStrides)
		dst[i] = a[aIdx] - b[bIdx]
	}
}

func mulBroadcastInt64(dst, a, b []int64, aShape, bShape, outShape tensor.Shape) {
	outStrides := outShape.ComputeStrides()
	aStrides := computeBroadcastStridesForShape(aShape, outShape)
	bStrides := computeBroadcastStridesForShape(bShape, outShape)

	n := outShape.NumElements()
	for i := 0; i < n; i++ {
		aIdx := computeFlatIndex(i, outStrides, aStrides)
		bIdx := computeFlatIndex(i, outStrides, bStrides)
		dst[i] = a[aIdx] * b[bIdx]
	}
}

func divBroadcastInt64(dst, a, b []int64, aShape, bShape, outShape tensor.Shape) {
	outStrides := outShape.ComputeStrides()
	aStrides := computeBroadcastStridesForShape(aShape, outShape)
	bStrides := computeBroadcastStridesForShape(bShape, outShape)

	n := outShape.NumElements()
	for i := 0; i < n; i++ {
		aIdx := computeFlatIndex(i, outStrides, aStrides)
		bIdx := computeFlatIndex(i, outStrides, bStrides)
		dst[i] = a[aIdx] / b[bIdx]
	}
}

func transposeInt64(dst, src []int64, shape tensor.Shape, axes []int) {
	ndim := len(shape)
	srcStrides := shape.ComputeStrides()

	// Compute destination shape and strides
	dstShape := make(tensor.Shape, ndim)
	for i, ax := range axes {
		dstShape[i] = shape[ax]
	}
	dstStrides := dstShape.ComputeStrides()

	// Transpose data
	n := shape.NumElements()
	for i := 0; i < n; i++ {
		// Compute multi-dimensional coordinates in source
		coords := make([]int, ndim)
		idx := i
		for dim := 0; dim < ndim; dim++ {
			coords[dim] = idx / srcStrides[dim]
			idx %= srcStrides[dim]
		}

		// Permute coordinates according to axes
		permutedCoords := make([]int, ndim)
		for dstDim, srcDim := range axes {
			permutedCoords[dstDim] = coords[srcDim]
		}

		// Compute flat index in destination
		dstIdx := 0
		for dim := 0; dim < ndim; dim++ {
			dstIdx += permutedCoords[dim] * dstStrides[dim]
		}

		dst[dstIdx] = src[i]
	}
}
