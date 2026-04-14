# gorch

A PyTorch-like deep learning framework in Go, backed by Apple Metal GPU.

Pure Go framework logic (tensors, autograd, nn modules, optimizers) with Apple's Accelerate (BLAS/vDSP/vForce) for CPU and Metal/MPS for GPU. Designed for Apple Silicon's unified memory architecture — zero-copy data sharing between CPU and GPU.

**Loads pretrained GPT-2 and generates coherent text at 40 tokens/second on M4.**

## Highlights

- **GPT-2 inference** — load pretrained weights from HuggingFace safetensors, generate text with temperature/top-k sampling
- **97.2% MNIST** in ~1 second, **90.6% Fashion-MNIST CNN**, **97.4% breast cancer diagnosis** in 64ms
- **Pipeline parallelism** — split transformer blocks across processes via TCP (fragmind)
- **628x matmul speedup** via Apple Accelerate BLAS over naive Go loops

## Features

- **Tensor** — N-dimensional arrays on CPU or Metal GPU with unified memory
- **Autograd** — reverse-mode automatic differentiation with topological sort
- **Broadcasting** — NumPy-compatible shape broadcasting (AddB, SubB, MulB, DivB)
- **Accelerate CPU** — BLAS (cblas_sgemm), vDSP vector ops, vForce transcendentals
- **Metal GPU** — element-wise ops via custom .metal kernels, matrix multiply via MPS
- **nn** — Linear, Conv2d, MaxPool2d, Flatten, Embedding, LayerNorm, MultiHeadAttention, TransformerBlock, BatchNorm1d, Dropout, Sequential
- **GPT** — decoder-only transformer LM with pretrained weight loading (GPT-2 compatible)
- **optim** — SGD (momentum), Adam, StepLR, CosineAnnealingLR, WarmupCosineScheduler
- **Loss** — MSELoss, CrossEntropyLoss
- **Ops** — Exp, Log, GELU, Softmax, LogSoftmax, Transpose, MaskFill, EmbeddingLookup, ScaledMatMul
- **Data** — DataLoader, MNIST, Fashion-MNIST, Wine Quality, Breast Cancer, generic CSV
- **Model I/O** — Safetensors load/save (F32, F16, BF16), BPE tokenizer (GPT-2 compatible)
- **Text generation** — greedy decoding, temperature sampling, top-k, top-p, KV cache
- **Fragmind** — pipeline-parallel inference across processes via TCP

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

    model := nn.NewSequential(
        nn.NewLinear(2, 8),
        nn.NewReLU(),
        nn.NewLinear(8, 1),
    )

    opt := optim.NewAdam(model.Parameters(), 0.01)
    for epoch := 0; epoch < 500; epoch++ {
        opt.ZeroGrad()
        loss := g.MSELoss(model.Forward(inputs), targets)
        loss.Backward()
        opt.Step()
    }
    fmt.Println(model.Forward(inputs).Data()) // [~0, ~1, ~1, ~0]
}
```

## Pretrained GPT-2 Inference

```go
package main

import (
    "fmt"
    "github.com/vinq1911/gorch/model"
)

func main() {
    // Download GPT-2 from HuggingFace (131 MB, cached)
    model.DownloadGPT2("openai-community/gpt2", "./gpt2_cache")

    tok, _ := model.LoadTokenizer("./gpt2_cache/vocab.json", "./gpt2_cache/merges.txt")
    cfg := model.GPT2Small()
    gpt, _ := model.LoadGPT2("./gpt2_cache", cfg)

    // Generate text
    text := model.GenerateText(gpt, tok, "The meaning of life is", model.DefaultGenerateConfig())
    fmt.Println(text)
}
```

Output: `"The meaning of life is not the same as the meaning of death."`

## Pipeline-Parallel Inference (Fragmind)

Split a model across processes for distributed inference:

```go
import "github.com/vinq1911/gorch/fragmind"

// Split GPT-2 into 2 fragments
frags := fragmind.SplitGPT(gpt, 2)
// Fragment 0: Embedding + Blocks[0:6]
// Fragment 1: Blocks[6:12] + LM Head

// Run locally (in-process pipeline)
logits := fragmind.PipelineInfer(frags, tokenIDs)

// Or serve Fragment 1 over TCP
server := fragmind.NewFragmentServer(frags[1], ":8080")
server.Start()
// Fragment 0 sends activations to Fragment 1 over the network
```

All split configurations produce bit-identical output to the unsplit model.

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
  tensor.go            Tensor type, creation, indexing, reshape, transpose, device transfer
  broadcast.go         NumPy-compatible broadcasting (AddB, SubB, MulB, DivB)
  attention_ops.go     MaskFill, EmbeddingLookup, ScaledMatMul, CausalMask
  conv.go              Conv2d forward/backward (im2col + BLAS sgemm)
  pool.go              MaxPool2d, Flatten
  ops.go               Ops with 3-tier dispatch + GELU activation
  autograd.go          Reverse-mode automatic differentiation
  loss.go              MSELoss, CrossEntropyLoss
  accelerate/
    shim.h / shim.c    C wrapper for Accelerate BLAS/vDSP/vForce
    accelerate.go      Go bindings
  metal/
    shim.h / shim.m    Objective-C Metal bridge
    metal.go           Go bindings for Metal device, buffers, pipelines
    kernels.go         Metal shader source for element-wise ops
  nn/
    module.go          Linear, Conv2d, MaxPool2d, Flatten, Sequential, activations
    embedding.go       Token/position embedding lookup
    layernorm.go       Layer normalization with learnable gamma/beta
    batchnorm.go       Batch normalization with running stats, train/eval
    attention.go       Multi-head self-attention with causal masking
    transformer.go     Pre-norm transformer block (attn + FFN(GELU) + residuals)
    gpt.go             GPT decoder-only language model
    dropout.go         Dropout with inverted scaling, train/eval modes
  optim/
    optim.go           SGD (momentum) and Adam optimizers
    scheduler.go       StepLR, CosineAnnealingLR, WarmupCosineScheduler
  model/
    safetensors.go     Load/save safetensors format (F32/F16/BF16)
    gpt2_loader.go     Download + load GPT-2 from HuggingFace safetensors
    generate.go        Text generation: greedy, temperature, top-k, top-p, KV cache
    tokenizer.go       BPE tokenizer (GPT-2 compatible) + char-level tokenizer
  fragmind/
    fragment.go        Pipeline-parallel inference: split model, TCP transport
  data/
    dataloader.go      Batched DataLoader with shuffle
    mnist.go           MNIST/Fashion-MNIST reader with auto-download
    tabular.go         Generic CSV loader with normalize + train/test split
    wine.go            UCI Wine Quality dataset
    breast_cancer.go   UCI Breast Cancer Wisconsin dataset
  e2e/
    mnist_test.go          MNIST training (97.2%)
    fashion_test.go        Fashion-MNIST MLP (88.1%)
    cnn_fashion_test.go    Fashion-MNIST CNN (90.6%)
    realworld_test.go      Wine + Breast Cancer multi-arch harness
    comprehensive_test.go  Full benchmark suite (7 benchmarks)
    pretrained_test.go     GPT-2 pretrained inference
    fragmind_test.go       Pipeline parallelism verification
    improvements_test.go   BatchNorm/GELU/Dropout/LR ablation study
  doc/
    training-guide.md              How to train with gorch
    decisions.md                   Architecture decision records
    explainer.md / .pdf            Non-technical explainer
    comprehensive-report.pdf       Full benchmark report
    pretrained-inference-report.pdf GPT-2 inference results
    fragmind-report.pdf            Pipeline parallelism results
    improvements-report.pdf        BatchNorm/GELU/Dropout ablation
    fashion-mnist-report.md        Fashion-MNIST detailed results
    realworld-report.pdf           Wine + BC + Fashion results
```

## Running Tests

```bash
# Unit tests (fast, no network) — 112 tests
CGO_ENABLED=1 go test ./... -v

# End-to-end tests (downloads data, trains models)
CGO_ENABLED=1 go test ./e2e/ -tags e2e -v -timeout 30m

# Specific e2e tests
CGO_ENABLED=1 go test ./e2e/ -tags e2e -run TestPretrainedGPT2 -v -timeout 30m
CGO_ENABLED=1 go test ./e2e/ -tags e2e -run TestFragmindPipeline -v -timeout 30m
CGO_ENABLED=1 go test ./e2e/ -tags e2e -run TestComprehensiveBenchmark -v -timeout 30m
```

## Benchmark Results (M4)

| Benchmark | Result | Time |
|-----------|--------|------|
| MNIST digit classification | 97.0% | 1.3s |
| Fashion-MNIST (MLP) | 88.9% | 5.4s |
| Fashion-MNIST (CNN) | 90.6% | 44.6s |
| Fashion-MNIST (BN+GELU+Dropout+CosLR) | 90.4% | 16.5s |
| Breast cancer diagnosis | 95.6% | 64ms |
| Wine quality prediction | 64.4% | 297ms |
| GPT-2 pretrained inference | 40 tok/s | — |
| GPT char-level training | loss 3.79→0.009 | 1.1s |
| Fragmind 2-frag local | 39.7 tok/s | <3% overhead |
| Accelerate matmul 512x512 | 158 µs | 628x vs naive |

## Roadmap

- [x] Tensors with CPU/Metal dual dispatch
- [x] Reverse-mode autograd
- [x] nn.Linear, Sequential, ReLU/Sigmoid/Tanh
- [x] SGD (momentum) and Adam optimizers
- [x] MSELoss, CrossEntropyLoss
- [x] Softmax, LogSoftmax, Exp, Log, GELU
- [x] DataLoader with batching and shuffle
- [x] MNIST training (97.2% accuracy, ~1s on M4)
- [x] Accelerate CPU backend (BLAS, vDSP, vForce) — 30x training speedup
- [x] Conv2d (im2col + BLAS), MaxPool2d, Flatten — CNN support
- [x] Fashion-MNIST CNN: 90.6% accuracy
- [x] Embedding, LayerNorm, MultiHeadAttention, TransformerBlock
- [x] GPT decoder-only language model
- [x] BatchNorm1d, Dropout, GELU activation
- [x] LR schedulers (StepLR, CosineAnnealing, WarmupCosine)
- [x] Safetensors load/save (F32, F16, BF16)
- [x] BPE tokenizer (GPT-2 compatible)
- [x] Pretrained GPT-2 inference (40 tok/s, coherent text generation)
- [x] Text generation (temperature, top-k, top-p sampling, KV cache)
- [x] Broadcasting (NumPy-compatible shape broadcasting)
- [x] Fragmind pipeline-parallel inference (TCP transport)
- [x] Real-world benchmarks: Wine Quality, Breast Cancer, Fashion-MNIST
- [x] Improvement ablation study (BN+GELU+Dropout+CosLR → +1.5%)
- [ ] GPU autograd (backward pass on Metal)
- [ ] KV cache integration into GPT forward pass
- [ ] Pretrained model fine-tuning
- [ ] ONNX export/import

## License

MIT
