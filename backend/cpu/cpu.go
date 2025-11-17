// Copyright 2025 Born ML Framework. All rights reserved.
// Use of this source code is governed by an Apache 2.0
// license that can be found in the LICENSE file.

package cpu

import (
	"github.com/born-ml/born/internal/backend/cpu"
)

// Backend represents the CPU backend implementation.
type Backend = cpu.CPUBackend

// New creates a new CPU backend.
//
// Example:
//
//	import (
//	    "github.com/born-ml/born/backend/cpu"
//	    "github.com/born-ml/born/tensor"
//	)
//
//	func main() {
//	    backend := cpu.New()
//	    x := tensor.Zeros[float32](tensor.Shape{2, 3}, backend)
//	}
func New() *Backend {
	return cpu.New()
}
