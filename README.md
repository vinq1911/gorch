# gorch

A PyTorch-like deep learning framework in Go, backed by Apple Metal GPU.

Pure Go framework logic (tensors, autograd, nn modules, optimizers) with Apple's Accelerate (BLAS/vDSP/vForce) for CPU and Metal/MPS for GPU. Designed for Apple Silicon's unified memory architecture — zero-copy data sharing between CPU and GPU.

**97.2% accuracy on MNIST** with a 2-layer MLP trained in **~1 second** on M4.

## Features

- **Tensor** — N-dimensional arrays on CPU or Metal GPU with unified memory
- **Autograd** — reverse-mode automatic differentiation with topological sort
- **Accelerate CPU** — BLAS (cblas_sgemm), vDSP vector ops, vForce transcendentals
- **Metal GPU** — element-wise ops via custom .metal kernels, matrix multiply via MPS
- **nn** — Module interface, Linear, Conv2d, MaxPool2d, Flatten, ReLU, Sigmoid, Tanh, Sequential
- **optim** — SGD (with momentum), Adam
- **Loss** — MSELoss, CrossEntropyLoss
- **Ops** — Exp, Log, Softmax, LogSoftmax, and all standard element-wise ops
- **Data** — DataLoader with shuffle, batching; built-in MNIST reader with auto-download

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Go 1.26+
- Xcode Command Line Tools (for Metal framework headers)

## Quick Start

```go
package main

import (
    "fmt"

    g "github.com/vinq1911/gorch"
    "github.com/vinq1911/gorch/nn"
    "github.com/vinq1911/gorch/optim"
)

func main() {
    // XOR dataset
    inputs := g.NewTensor([]float32{0,0, 0,1, 1,0, 1,1}, 4, 2)
    targets := g.NewTensor([]float32{0, 1, 1, 0}, 4, 1)

    // Define model
    model := nn.NewSequential(
        nn.NewLinear(2, 8),
        nn.NewReLU(),
        nn.NewLinear(8, 1),
    )

    // Train
    opt := optim.NewAdam(model.Parameters(), 0.01)
    for epoch := 0; epoch < 500; epoch++ {
        opt.ZeroGrad()
        pred := model.Forward(inputs)
        loss := g.MSELoss(pred, targets)
        loss.Backward()
        opt.Step()

        if epoch%100 == 0 {
            fmt.Printf("epoch %d  loss: %.6f\n", epoch, loss.Data()[0])
        }
    }

    // Predict
    pred := model.Forward(inputs)
    fmt.Printf("predictions: %v\n", pred.Data())
}
```

## MNIST Example

```go
package main

import (
    "fmt"

    g "github.com/vinq1911/gorch"
    "github.com/vinq1911/gorch/data"
    "github.com/vinq1911/gorch/nn"
    "github.com/vinq1911/gorch/optim"
)

func main() {
    trainSet, _ := data.LoadMNIST("./mnist_data", true)
    testSet, _ := data.LoadMNIST("./mnist_data", false)

    model := nn.NewSequential(
        nn.NewLinear(784, 128),
        nn.NewReLU(),
        nn.NewLinear(128, 10),
    )
    opt := optim.NewAdam(model.Parameters(), 0.001)
    loader := data.NewDataLoader(trainSet, 64, true)

    for epoch := 0; epoch < 3; epoch++ {
        loader.Reset()
        for {
            inputs, targets := loader.Next()
            if inputs == nil {
                break
            }
            opt.ZeroGrad()
            logits := model.Forward(inputs)
            loss := g.CrossEntropyLoss(logits, targets)
            loss.Backward()
            opt.Step()
        }
        fmt.Printf("Epoch %d complete\n", epoch+1)
    }
    // Evaluate: ~97% accuracy after 3 epochs
}
```

## Metal GPU

Tensors can be moved to Metal GPU for accelerated compute. Unified memory means no explicit copies — Go and the GPU read/write the same physical memory.

```go
gpu, _ := g.InitMetal()

a := g.Rand(1024).ToMetal(gpu.Dev)
b := g.Rand(1024).ToMetal(gpu.Dev)
c := g.Add(a, b) // runs on GPU via Metal kernel

// MPS-accelerated matrix multiply
x := g.Rand(512, 512).ToMetal(gpu.Dev)
w := g.Rand(512, 512).ToMetal(gpu.Dev)
y := g.MatMul(x, w) // runs on GPU via MPS
```

## Architecture

```
gorch (Go)                         ← tensors, autograd, nn, optim, data
  |                    |
  v  CGo               v  CGo
metal/shim.m (ObjC)    accelerate/shim.c
  |                    |
  v                    v
Metal + MPS            Accelerate (BLAS/vDSP/vForce)
  |                    |
Apple GPU              Apple CPU (NEON SIMD, AMX)
```

## Project Structure

```
gorch/
  tensor.go            Tensor type, creation, indexing, device transfer
  conv.go              Conv2d forward/backward (im2col + BLAS sgemm)
  pool.go              MaxPool2d, Flatten
  ops.go               Ops with 3-tier dispatch: Metal GPU → Accelerate CPU → fallback
  autograd.go          Reverse-mode automatic differentiation
  loss.go              Loss functions (MSELoss, CrossEntropyLoss)
  accelerate/
    shim.h / shim.c    C wrapper for Accelerate BLAS/vDSP/vForce
    accelerate.go      Go bindings for Accelerate
  metal/
    shim.h / shim.m    Objective-C Metal bridge
    metal.go           Go bindings for Metal device, buffers, pipelines
    kernels.go         Metal shader source for element-wise ops
  nn/
    module.go          Linear, Conv2d, MaxPool2d, Flatten, Sequential, activations
  optim/
    optim.go           SGD and Adam optimizers
  data/
    dataloader.go      Batched DataLoader with shuffle
    mnist.go           MNIST dataset reader with auto-download
  e2e/
    mnist_test.go      End-to-end MNIST training (97.2% accuracy, ~1s)
    fashion_test.go    Fashion-MNIST MLP benchmark (88.1% accuracy)
    cnn_fashion_test.go Fashion-MNIST CNN benchmark (90.6% accuracy)
```

## Running Tests

```bash
# Unit tests (fast, no network)
CGO_ENABLED=1 go test ./... -v

# End-to-end tests (downloads MNIST, trains model)
CGO_ENABLED=1 go test ./e2e/ -tags e2e -v -timeout 10m
```

## Roadmap

- [x] Tensors with CPU/Metal dual dispatch
- [x] Reverse-mode autograd
- [x] nn.Linear, Sequential, ReLU/Sigmoid/Tanh
- [x] SGD (momentum) and Adam optimizers
- [x] MSELoss, CrossEntropyLoss
- [x] Softmax, LogSoftmax, Exp, Log
- [x] DataLoader with batching and shuffle
- [x] MNIST training (97.2% accuracy, ~1s on M4)
- [x] Accelerate CPU backend (BLAS, vDSP, vForce) — 30x training speedup
- [x] Conv2d (im2col + BLAS), MaxPool2d, Flatten — CNN support
- [x] Fashion-MNIST CNN: 90.6% accuracy (vs 88.1% MLP)
- [ ] Dropout, BatchNorm, LayerNorm
- [ ] Broadcasting
- [ ] Save/Load model weights
- [ ] GPU autograd (backward pass on Metal)
- [ ] Embedding layer
- [ ] Attention / Transformer blocks

## License

MIT
