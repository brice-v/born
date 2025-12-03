package serialization

import (
	"time"

	"github.com/born-ml/born/internal/tensor"
)

// Format constants.
const (
	MagicBytes        = "BORN"
	FormatVersion     = 1    // v1: Basic format without checksum
	FormatVersionV2   = 2    // v2: With SHA-256 checksum
	HeaderAlignment   = 64   // Align tensor data to 64 bytes for optimal performance
	FixedHeaderSizeV2 = 64   // v2 fixed header size (0x40 bytes)
	ChecksumSize      = 32   // SHA-256 checksum size (32 bytes)
	ChecksumOffsetV2  = 0x20 // Checksum offset in v2 fixed header
)

// Flags for the .born format.
const (
	FlagCompressed   uint32 = 1 << 0 // bit 0: gzip compression
	FlagHasOptimizer uint32 = 1 << 1 // bit 1: optimizer state included
	FlagHasMetadata  uint32 = 1 << 2 // bit 2: custom metadata included
)

// Header represents the JSON header in a .born file.
type Header struct {
	FormatVersion  int               `json:"format_version"`       // Version of the .born format
	BornVersion    string            `json:"born_version"`         // Version of Born that created this file
	ModelType      string            `json:"model_type"`           // Type of model (e.g., "Sequential", "Linear")
	CreatedAt      time.Time         `json:"created_at"`           // When the file was created
	Tensors        []TensorMeta      `json:"tensors"`              // Tensor metadata
	Metadata       map[string]string `json:"metadata"`             // Custom metadata
	CheckpointMeta *CheckpointMeta   `json:"checkpoint,omitempty"` // Checkpoint metadata (optional)
}

// CheckpointMeta contains training state information for checkpoints.
type CheckpointMeta struct {
	IsCheckpoint    bool           `json:"is_checkpoint"`    // Whether this is a checkpoint file
	Epoch           int            `json:"epoch"`            // Training epoch number
	Step            int64          `json:"step"`             // Training step number
	Loss            float64        `json:"loss"`             // Loss value at checkpoint
	OptimizerType   string         `json:"optimizer_type"`   // Optimizer type ("SGD", "Adam", etc.)
	OptimizerConfig map[string]any `json:"optimizer_config"` // Optimizer hyperparameters
	TrainingMeta    map[string]any `json:"training_meta"`    // Additional training metadata
}

// TensorMeta describes a tensor in the .born file.
type TensorMeta struct {
	Name   string `json:"name"`   // Tensor name (e.g., "layer.0.weight")
	DType  string `json:"dtype"`  // Data type (e.g., "float32", "float64")
	Shape  []int  `json:"shape"`  // Tensor shape
	Offset int64  `json:"offset"` // Offset in the data section (bytes from start of tensor data)
	Size   int64  `json:"size"`   // Size in bytes
}

// dtypeToString converts tensor.DataType to string representation.
func dtypeToString(dt tensor.DataType) string {
	switch dt {
	case tensor.Float32:
		return "float32"
	case tensor.Float64:
		return "float64"
	case tensor.Int32:
		return "int32"
	case tensor.Int64:
		return "int64"
	case tensor.Uint8:
		return "uint8"
	case tensor.Bool:
		return "bool"
	default:
		return "unknown"
	}
}

// stringToDtype converts string representation to tensor.DataType.
func stringToDtype(s string) (tensor.DataType, bool) {
	switch s {
	case "float32":
		return tensor.Float32, true
	case "float64":
		return tensor.Float64, true
	case "int32":
		return tensor.Int32, true
	case "int64":
		return tensor.Int64, true
	case "uint8":
		return tensor.Uint8, true
	case "bool":
		return tensor.Bool, true
	default:
		return 0, false
	}
}
