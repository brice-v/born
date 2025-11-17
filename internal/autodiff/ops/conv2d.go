package ops

import (
	"github.com/born-ml/born/internal/tensor"
)

// Conv2DOp records a 2D convolution operation for autodiff.
//
// Forward: output = Conv2D(input, kernel, stride, padding)
//
// Backward (gradients):
//   - d_input:  "transposed convolution" or "deconvolution" of d_output with kernel
//   - d_kernel: convolution of input with d_output
//
// References:
//   - "A guide to convolution arithmetic for deep learning" (Dumoulin & Visin, 2016)
//   - CS231n: Convolutional Neural Networks for Visual Recognition
type Conv2DOp struct {
	input   *tensor.RawTensor
	kernel  *tensor.RawTensor
	output  *tensor.RawTensor
	stride  int
	padding int
}

// NewConv2DOp creates a new Conv2D operation.
func NewConv2DOp(input, kernel, output *tensor.RawTensor, stride, padding int) *Conv2DOp {
	return &Conv2DOp{
		input:   input,
		kernel:  kernel,
		output:  output,
		stride:  stride,
		padding: padding,
	}
}

// Inputs returns the input tensors.
func (op *Conv2DOp) Inputs() []*tensor.RawTensor {
	return []*tensor.RawTensor{op.input, op.kernel}
}

// Output returns the output tensor.
func (op *Conv2DOp) Output() *tensor.RawTensor {
	return op.output
}

// Backward computes gradients for Conv2D.
//
// Given:
//   - outputGrad: ∂L/∂output [N, C_out, H_out, W_out]
//
// Compute:
//   - inputGrad:  ∂L/∂input  [N, C_in, H, W]
//   - kernelGrad: ∂L/∂kernel [C_out, C_in, K_h, K_w]
//
// Gradient formulas:
//
//  1. Input gradient (transposed convolution):
//     ∂L/∂input = TransposedConv2D(∂L/∂output, kernel, stride, padding)
//
//     This is essentially a "backward" convolution where we propagate
//     the output gradients back to input positions using the same kernel.
//
//  2. Kernel gradient (convolution):
//     ∂L/∂kernel[c_out, c_in, kh, kw] = Σ_{n,h,w} input[n,c_in,h+kh,w+kw] * ∂L/∂output[n,c_out,h,w]
//
//     This computes how much each kernel weight contributed to the loss
//     by correlating input patches with output gradients.
func (op *Conv2DOp) Backward(outputGrad *tensor.RawTensor, backend tensor.Backend) []*tensor.RawTensor {
	// Extract shapes
	inputShape := op.input.Shape()
	kernelShape := op.kernel.Shape()
	outputGradShape := outputGrad.Shape()

	N := inputShape[0]
	CIn := inputShape[1]
	H := inputShape[2]
	W := inputShape[3]
	COut := kernelShape[0]
	KH := kernelShape[2]
	KW := kernelShape[3]
	HOut := outputGradShape[2]
	WOut := outputGradShape[3]

	// 1. Compute input gradient (transposed convolution)
	inputGrad := conv2dBackwardInput(
		outputGrad, op.kernel,
		N, CIn, H, W,
		COut, KH, KW,
		HOut, WOut,
		op.stride, op.padding,
		backend.Device(),
	)

	// 2. Compute kernel gradient
	kernelGrad := conv2dBackwardKernel(
		outputGrad, op.input,
		N, CIn, H, W,
		COut, KH, KW,
		HOut, WOut,
		op.stride, op.padding,
		backend.Device(),
	)

	return []*tensor.RawTensor{inputGrad, kernelGrad}
}

// conv2dBackwardInput computes gradient w.r.t. input.
//
// Algorithm: Transposed convolution (full convolution)
//   - For each input position (n, c_in, h, w):
//   - Sum contributions from all output positions that used this input
//   - Each contribution is: outputGrad[n, c_out, h_out, w_out] * kernel[c_out, c_in, kh, kw]
//   - Where the kernel weight at (kh, kw) was used for this input-output pair
//
// Implementation:
//   - Flip kernel spatially (convolution → correlation transform)
//   - Perform "full" convolution with appropriate padding
//   - Adjust for stride by inserting zeros between outputGrad elements
func conv2dBackwardInput(
	outputGrad, kernel *tensor.RawTensor,
	N, CIn, H, W int,
	COut, KH, KW int,
	HOut, WOut int,
	stride, padding int,
	device tensor.Device,
) *tensor.RawTensor {
	inputGrad, err := tensor.NewRaw(tensor.Shape{N, CIn, H, W}, outputGrad.DType(), device)
	if err != nil {
		panic(err)
	}

	// Dispatch by dtype
	switch outputGrad.DType() {
	case tensor.Float32:
		conv2dBackwardInputFloat32(
			inputGrad, outputGrad, kernel,
			N, CIn, H, W, COut, KH, KW, HOut, WOut,
			stride, padding,
		)
	case tensor.Float64:
		conv2dBackwardInputFloat64(
			inputGrad, outputGrad, kernel,
			N, CIn, H, W, COut, KH, KW, HOut, WOut,
			stride, padding,
		)
	default:
		panic("conv2dBackwardInput: unsupported dtype")
	}

	return inputGrad
}

// conv2dBackwardInputFloat32 computes input gradient for float32.
func conv2dBackwardInputFloat32(
	inputGrad, outputGrad, kernel *tensor.RawTensor,
	N, CIn, H, W, COut, KH, KW, HOut, WOut, stride, padding int,
) {
	inputGradData := inputGrad.AsFloat32()
	outputGradData := outputGrad.AsFloat32()
	kernelData := kernel.AsFloat32()

	// Initialize to zero
	for i := range inputGradData {
		inputGradData[i] = 0.0
	}

	// For each batch
	for n := 0; n < N; n++ {
		// For each output gradient position
		for outH := 0; outH < HOut; outH++ {
			for outW := 0; outW < WOut; outW++ {
				// For each output channel
				for cOut := 0; cOut < COut; cOut++ {
					outputGradIdx := n*COut*HOut*WOut + cOut*HOut*WOut + outH*WOut + outW
					gradVal := outputGradData[outputGradIdx]

					// Distribute this gradient to all input positions
					// that contributed to this output
					for cIn := 0; cIn < CIn; cIn++ {
						for kh := 0; kh < KH; kh++ {
							for kw := 0; kw < KW; kw++ {
								// Input position
								h := outH*stride - padding + kh
								w := outW*stride - padding + kw

								// Check bounds
								if h >= 0 && h < H && w >= 0 && w < W {
									kernelIdx := cOut*CIn*KH*KW + cIn*KH*KW + kh*KW + kw
									inputGradIdx := n*CIn*H*W + cIn*H*W + h*W + w

									// Accumulate gradient
									inputGradData[inputGradIdx] += gradVal * kernelData[kernelIdx]
								}
							}
						}
					}
				}
			}
		}
	}
}

// conv2dBackwardInputFloat64 computes input gradient for float64.
func conv2dBackwardInputFloat64(
	inputGrad, outputGrad, kernel *tensor.RawTensor,
	N, CIn, H, W, COut, KH, KW, HOut, WOut, stride, padding int,
) {
	inputGradData := inputGrad.AsFloat64()
	outputGradData := outputGrad.AsFloat64()
	kernelData := kernel.AsFloat64()

	for i := range inputGradData {
		inputGradData[i] = 0.0
	}

	for n := 0; n < N; n++ {
		for outH := 0; outH < HOut; outH++ {
			for outW := 0; outW < WOut; outW++ {
				for cOut := 0; cOut < COut; cOut++ {
					outputGradIdx := n*COut*HOut*WOut + cOut*HOut*WOut + outH*WOut + outW
					gradVal := outputGradData[outputGradIdx]

					for cIn := 0; cIn < CIn; cIn++ {
						for kh := 0; kh < KH; kh++ {
							for kw := 0; kw < KW; kw++ {
								h := outH*stride - padding + kh
								w := outW*stride - padding + kw

								if h >= 0 && h < H && w >= 0 && w < W {
									kernelIdx := cOut*CIn*KH*KW + cIn*KH*KW + kh*KW + kw
									inputGradIdx := n*CIn*H*W + cIn*H*W + h*W + w
									inputGradData[inputGradIdx] += gradVal * kernelData[kernelIdx]
								}
							}
						}
					}
				}
			}
		}
	}
}

// conv2dBackwardKernel computes gradient w.r.t. kernel.
//
// Algorithm: Convolution of input with outputGrad
//   - For each kernel position (c_out, c_in, kh, kw):
//   - Sum over all batch samples and output positions
//   - Each contribution is: input[n, c_in, h, w] * outputGrad[n, c_out, h_out, w_out]
//   - Where h = h_out * stride - padding + kh
//     w = w_out * stride - padding + kw
func conv2dBackwardKernel(
	outputGrad, input *tensor.RawTensor,
	N, CIn, H, W, COut, KH, KW, HOut, WOut, stride, padding int,
	device tensor.Device,
) *tensor.RawTensor {
	kernelGrad, err := tensor.NewRaw(tensor.Shape{COut, CIn, KH, KW}, outputGrad.DType(), device)
	if err != nil {
		panic(err)
	}

	switch outputGrad.DType() {
	case tensor.Float32:
		conv2dBackwardKernelFloat32(
			kernelGrad, outputGrad, input,
			N, CIn, H, W, COut, KH, KW, HOut, WOut,
			stride, padding,
		)
	case tensor.Float64:
		conv2dBackwardKernelFloat64(
			kernelGrad, outputGrad, input,
			N, CIn, H, W, COut, KH, KW, HOut, WOut,
			stride, padding,
		)
	default:
		panic("conv2dBackwardKernel: unsupported dtype")
	}

	return kernelGrad
}

// conv2dBackwardKernelFloat32 computes kernel gradient for float32.
func conv2dBackwardKernelFloat32(
	kernelGrad, outputGrad, input *tensor.RawTensor,
	N, CIn, H, W, COut, KH, KW, HOut, WOut, stride, padding int,
) {
	kernelGradData := kernelGrad.AsFloat32()
	outputGradData := outputGrad.AsFloat32()
	inputData := input.AsFloat32()

	// Initialize to zero
	for i := range kernelGradData {
		kernelGradData[i] = 0.0
	}

	// For each kernel weight
	for cOut := 0; cOut < COut; cOut++ {
		for cIn := 0; cIn < CIn; cIn++ {
			for kh := 0; kh < KH; kh++ {
				for kw := 0; kw < KW; kw++ {
					sum := float32(0.0)

					// Accumulate gradient over all batch and output positions
					for n := 0; n < N; n++ {
						for outH := 0; outH < HOut; outH++ {
							for outW := 0; outW < WOut; outW++ {
								// Input position corresponding to this kernel weight
								h := outH*stride - padding + kh
								w := outW*stride - padding + kw

								// Check bounds
								if h >= 0 && h < H && w >= 0 && w < W {
									inputIdx := n*CIn*H*W + cIn*H*W + h*W + w
									outputGradIdx := n*COut*HOut*WOut + cOut*HOut*WOut + outH*WOut + outW

									sum += inputData[inputIdx] * outputGradData[outputGradIdx]
								}
							}
						}
					}

					kernelIdx := cOut*CIn*KH*KW + cIn*KH*KW + kh*KW + kw
					kernelGradData[kernelIdx] = sum
				}
			}
		}
	}
}

// conv2dBackwardKernelFloat64 computes kernel gradient for float64.
func conv2dBackwardKernelFloat64(
	kernelGrad, outputGrad, input *tensor.RawTensor,
	N, CIn, H, W, COut, KH, KW, HOut, WOut, stride, padding int,
) {
	kernelGradData := kernelGrad.AsFloat64()
	outputGradData := outputGrad.AsFloat64()
	inputData := input.AsFloat64()

	for i := range kernelGradData {
		kernelGradData[i] = 0.0
	}

	for cOut := 0; cOut < COut; cOut++ {
		for cIn := 0; cIn < CIn; cIn++ {
			for kh := 0; kh < KH; kh++ {
				for kw := 0; kw < KW; kw++ {
					sum := float64(0.0)

					for n := 0; n < N; n++ {
						for outH := 0; outH < HOut; outH++ {
							for outW := 0; outW < WOut; outW++ {
								h := outH*stride - padding + kh
								w := outW*stride - padding + kw

								if h >= 0 && h < H && w >= 0 && w < W {
									inputIdx := n*CIn*H*W + cIn*H*W + h*W + w
									outputGradIdx := n*COut*HOut*WOut + cOut*HOut*WOut + outH*WOut + outW
									sum += inputData[inputIdx] * outputGradData[outputGradIdx]
								}
							}
						}
					}

					kernelIdx := cOut*CIn*KH*KW + cIn*KH*KW + kh*KW + kw
					kernelGradData[kernelIdx] = sum
				}
			}
		}
	}
}
