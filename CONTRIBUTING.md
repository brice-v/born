# Contributing to Born

**Status**: 🚧 Document in progress - Project in early development

Thank you for your interest in contributing to Born! This document will guide you through the contribution process.

**Note**: Born is currently in the initialization phase. Full contributing guidelines will be established as the project matures.

---

## Getting Started

### Prerequisites

- **Go 1.25+** - [Download](https://go.dev/dl/)
- **Git** - Version control
- **Make** (optional) - Build automation

### Setup Development Environment

```bash
# Clone repository (once created)
git clone https://github.com/born-ml/born.git
cd born

# Install dependencies
go mod download

# Run tests
go test ./...

# Build
go build ./...
```

---

## Development Workflow

### 1. Check Project Status

Before starting, check GitHub Issues and Discussions for:
- Current project roadmap and priorities
- Available tasks and features
- Technical discussions and decisions

### 2. Pick a Task

- Check GitHub Issues for tasks labeled `good-first-issue` or `help-wanted`
- Comment on the issue to claim it
- Reference the issue in your PR

### 3. Development

- Create a feature branch: `git checkout -b feature/your-feature`
- Write code following our [Code Standards](#code-standards)
- Write tests (coverage > 70%)
- Update documentation

### 4. Testing

```bash
# Run all tests
go test ./...

# Run with race detector
go test -race ./...

# Check coverage
go test -cover ./...

# Run benchmarks
go test -bench=. ./...
```

### 5. Submit

- Commit with clear messages (see [Commit Guidelines](#commit-guidelines))
- Push to your branch
- Create Pull Request
- Link to related GitHub issue

---

## Code Standards

### Go Style

Follow [Effective Go](https://go.dev/doc/effective_go) and:

**Naming**:
- Exported: `PascalCase`
- Unexported: `camelCase`
- Constants: `PascalCase` or `UPPER_SNAKE` for groups
- Interfaces: `-er` suffix when appropriate (`Backend`, `Optimizer`)

**Example**:
```go
// Good
type CPUBackend struct {
    memoryPool *MemoryPool
}

func (b *CPUBackend) Add(a, b *RawTensor) *RawTensor {
    result := b.allocate(a.Shape())
    addFloat32(result.data, a.data, b.data)
    return result
}

// Bad
type cpu_backend struct {  // ❌ wrong naming
    memory_pool *MemoryPool  // ❌ unexported should be camelCase
}

func (b *cpu_backend) ADD(a, b *RawTensor) *RawTensor {  // ❌ all caps
    // ...
}
```

### Comments

- All exported functions/types MUST have doc comments
- Comments start with the name of the thing being described
- Use complete sentences

```go
// Tensor represents a multi-dimensional array with type T and backend B.
// It provides operations for mathematical computations and automatic
// differentiation when used with an autodiff backend.
type Tensor[T DType, B Backend] struct {
    // ...
}

// Add performs element-wise addition of two tensors.
// Returns a new tensor containing the result.
// Panics if shapes are incompatible.
func (t *Tensor[T, B]) Add(other *Tensor[T, B]) *Tensor[T, B] {
    // ...
}
```

### Error Handling

**Return errors** for expected failures:
```go
func (b *CPUBackend) Allocate(size int) (*RawTensor, error) {
    if size <= 0 {
        return nil, fmt.Errorf("invalid size: %d", size)
    }
    // ...
}
```

**Panic** for programmer errors:
```go
func (t *Tensor[T, B]) Shape() []int {
    if t.raw == nil {
        panic("tensor is nil")
    }
    return t.raw.shape
}
```

### Testing

**Test naming**: `TestFunctionName`
```go
func TestTensorAdd(t *testing.T) {
    t.Run("compatible shapes", func(t *testing.T) {
        // ...
    })

    t.Run("incompatible shapes", func(t *testing.T) {
        // ...
    })
}
```

**Table-driven tests** for multiple cases:
```go
func TestMatMul(t *testing.T) {
    tests := []struct {
        name   string
        aShape []int
        bShape []int
        want   []int
    }{
        {"2x2 × 2x2", []int{2, 2}, []int{2, 2}, []int{2, 2}},
        {"2x3 × 3x4", []int{2, 3}, []int{3, 4}, []int{2, 4}},
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            // ...
        })
    }
}
```

---

## Commit Guidelines

### Format

```
type(scope): brief description

Detailed explanation if needed.

Fixes #123
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style (formatting, no logic change)
- `refactor`: Code restructuring (no behavior change)
- `perf`: Performance improvement
- `test`: Adding/updating tests
- `chore`: Maintenance tasks

### Examples

```bash
feat(tensor): implement Add operation with broadcasting

Adds element-wise addition for tensors with automatic broadcasting
following NumPy-style rules.

- Implemented broadcastShapes helper
- Added comprehensive tests
- Updated documentation

Closes #42
```

```bash
fix(cpu): resolve memory leak in pooling

Fixed memory pool not releasing buffers properly in edge case
where tensors were freed before gradients computed.

Fixes #87
```

---

## Documentation

### When to Update Docs

- New features → Update API docs and examples
- Breaking changes → Update migration guide
- Bug fixes → May need example updates
- Architecture changes → Update design docs

### Documentation Types

**Code Documentation**: Go doc comments
```go
// Package tensor provides multi-dimensional array operations.
package tensor
```

**User Documentation**: `docs/` (Markdown)
- Tutorials
- API guides
- Examples
- Architecture decisions

---

## Pull Request Process

### Before Submitting

- [ ] Code follows style guidelines
- [ ] Tests written and passing
- [ ] Documentation updated
- [ ] Commit messages are clear
- [ ] No merge conflicts

### PR Description

Include:
- **What**: Brief description of changes
- **Why**: Motivation and context
- **How**: Implementation approach
- **Testing**: How you tested
- **Related**: Link to GitHub issue

**Template**:
```markdown
## Summary
Brief description of what this PR does.

## Motivation
Why is this change needed?

## Changes
- Change 1
- Change 2

## Testing
- [ ] Unit tests added/updated
- [ ] Manual testing performed
- [ ] Benchmarks run (if performance-related)

## Related
Closes #123
```

### Review Process

1. Automated checks run (tests, linting)
2. Code review by maintainer(s)
3. Feedback addressed
4. Approval and merge

---

## Areas for Contribution

### High Priority

- **Core Tensor Operations** - Mathematical operations
- **CPU Backend** - SIMD optimizations
- **Testing** - Increase coverage
- **Documentation** - Examples and tutorials

### Medium Priority

- **GPU Backends** - CUDA, Vulkan expertise welcome
- **Optimization** - Performance improvements
- **Examples** - Real-world use cases

### Always Welcome

- **Bug Reports** - Detailed issue reports
- **Bug Fixes** - Any bug fixes appreciated
- **Documentation** - Improvements and corrections
- **Examples** - Additional examples

---

## Communication

**Project is in early development**. Communication channels will be established as project matures.

**For Now**:
- Follow development in GitHub repository
- Check GitHub Issues for updates
- Join GitHub Discussions when available

**Coming Soon**:
- GitHub Discussions
- Discord/Slack channel (maybe)

---

## Code of Conduct

**Be respectful and professional**. We're building something cool together.

Detailed Code of Conduct will be established as community grows.

---

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (TBD, likely MIT or Apache 2.0).

---

## Questions?

**For contributors**:
- GitHub Issues
- GitHub Discussions (coming soon)
- Project documentation in `docs/`

---

**Thank you for contributing to Born!**

Together we're building production-ready ML for Go.

---

**Last Updated**: 2025-11-17
