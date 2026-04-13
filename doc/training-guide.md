# Training with gorch

This guide walks through how to train neural networks using gorch — from basic concepts to transformers.

## Core concepts

Gorch follows the same pattern as PyTorch:

1. **Define a model** using `nn` modules
2. **Forward pass** — compute predictions
3. **Compute loss** — measure how wrong the predictions are
4. **Backward pass** — compute gradients via autograd
5. **Optimizer step** — update parameters

```
for each epoch:
    for each batch:
        optimizer.ZeroGrad()       // clear old gradients
        predictions = model.Forward(inputs)
        loss = LossFunction(predictions, targets)
        loss.Backward()            // compute gradients
        optimizer.Step()           // update weights
```

## Tensors

Tensors are N-dimensional arrays of float32 values.

```go
import g "github.com/vinq1911/gorch"

// Create tensors
x := g.Zeros(3, 4)             // 3x4 matrix of zeros
y := g.Ones(5)                 // vector of ones
z := g.Rand(2, 3)              // uniform random in [0, 1)
w := g.RandN(10, 5)            // standard normal
c := g.Full(3.14, 2, 2)        // constant fill

// From data
t := g.NewTensor([]float32{1, 2, 3, 4}, 2, 2)

// Properties
t.Shape()    // [2, 2]
t.Size()     // 4
t.Dim()      // 2
t.Data()     // []float32{1, 2, 3, 4}
t.At(0, 1)   // 2.0

// Reshape and transpose
r := g.ReshapeOp(t, 4)       // (4,)
tr := g.Transpose2D(t)       // (2, 2) transposed
```

### Gradient tracking

Tensors that should be learned need gradient tracking enabled:

```go
w := g.RandN(10, 5)
w.SetRequiresGrad(true)

// After backward pass:
w.Grad()     // gradient tensor
w.ZeroGrad() // reset for next iteration
```

When you use `nn.NewLinear(...)`, gradients are enabled automatically for weights and biases.

## Building models

### Single layer

```go
import "github.com/vinq1911/gorch/nn"

layer := nn.NewLinear(784, 10)  // 784 inputs, 10 outputs
out := layer.Forward(input)     // input shape: (batch, 784)
params := layer.Parameters()    // [weight, bias]
```

### Multi-layer with Sequential

```go
model := nn.NewSequential(
    nn.NewLinear(784, 128),
    nn.NewReLU(),
    nn.NewDropout(0.2),         // 20% dropout for regularization
    nn.NewLinear(128, 64),
    nn.NewReLU(),
    nn.NewLinear(64, 10),
)
```

### CNN model

```go
model := nn.NewSequential(
    nn.NewConv2d(1, 16, 3, 1, 1),  // (1,28,28) -> (16,28,28)
    nn.NewReLU(),
    nn.NewMaxPool2d(2, 2),          // -> (16,14,14)
    nn.NewConv2d(16, 32, 3, 1, 1),  // -> (32,14,14)
    nn.NewReLU(),
    nn.NewMaxPool2d(2, 2),          // -> (32,7,7)
    nn.NewFlatten(),                // -> (1568)
    nn.NewLinear(1568, 10),         // -> (10)
)
```

### GPT language model

```go
model := nn.NewGPT(
    50257,  // vocab size
    256,    // embedding dim
    4,      // attention heads
    4,      // transformer layers
    128,    // max sequence length
)

// Forward pass: token IDs → logits
tokens := []int{1, 5, 10, 20}
logits := model.Forward(tokens)  // (4, 50257)
```

### Available layers

| Layer | Constructor | Description |
|-------|-------------|-------------|
| `Linear` | `nn.NewLinear(in, out)` | Fully connected, Kaiming init |
| `Conv2d` | `nn.NewConv2d(inC, outC, kernel, stride, pad)` | 2D convolution (im2col + BLAS) |
| `MaxPool2d` | `nn.NewMaxPool2d(kernel, stride)` | 2D max pooling |
| `Flatten` | `nn.NewFlatten()` | Reshape 4D→2D for CNN→MLP transition |
| `Embedding` | `nn.NewEmbedding(vocab, dim)` | Token/position lookup table |
| `LayerNorm` | `nn.NewLayerNorm(dim)` | Normalize last dimension |
| `MultiHeadAttention` | `nn.NewMultiHeadAttention(dim, heads)` | Self-attention with causal masking |
| `TransformerBlock` | `nn.NewTransformerBlock(dim, heads)` | Pre-norm transformer (attn + FFN) |
| `Dropout` | `nn.NewDropout(p)` | Random zeroing with inverted scaling |
| `ReLU` | `nn.NewReLU()` | max(0, x) activation |
| `Sigmoid` | `nn.NewSigmoid()` | Sigmoid activation |
| `Tanh` | `nn.NewTanh()` | Tanh activation |
| `Sequential` | `nn.NewSequential(layers...)` | Chain layers in order |
| `GPT` | `nn.NewGPT(vocab, dim, heads, layers, maxSeq)` | Decoder-only transformer LM |

## Loss functions

### MSELoss — for regression

```go
loss := g.MSELoss(pred, targets)   // mean((pred - target)^2)
```

### CrossEntropyLoss — for classification

Takes raw logits (pre-softmax) and integer class labels:

```go
logits := model.Forward(inputs)                // (batch, numClasses)
targets := g.NewTensor(labels, batchSize, 1)   // integer labels as float32
loss := g.CrossEntropyLoss(logits, targets)     // scalar tensor
```

## Optimizers

### SGD and Adam

```go
import "github.com/vinq1911/gorch/optim"

opt := optim.NewSGD(model.Parameters(), 0.01, 0.9)   // lr=0.01, momentum=0.9
opt := optim.NewAdam(model.Parameters(), 0.001)        // default betas
```

### Learning rate scheduling

```go
// Step decay: multiply LR by 0.1 every 30 epochs
sched := optim.NewStepLR(opt, 0.001, 30, 0.1, opt.SetLR)

// Cosine annealing: smooth decay from 0.001 to 0
sched := optim.NewCosineAnnealingLR(opt, 0.001, 0.0, 100, opt.SetLR)

// Warmup + cosine (GPT-style): 10 warmup steps, then cosine decay
sched := optim.NewWarmupCosineScheduler(0.001, 0.0, 10, 1000, opt.SetLR)

// Call after each epoch (or step):
sched.Step()
```

## Dropout

```go
dropout := nn.NewDropout(0.1)

// Training mode (dropout active):
dropout.Train()
out := dropout.Forward(x)

// Evaluation mode (dropout disabled):
dropout.Eval()
out := dropout.Forward(x)  // pass-through
```

## Saving and loading models

### Save model weights

```go
import "github.com/vinq1911/gorch/model"

params := myModel.Parameters()
nameMap := map[int]string{
    0: "layer1.weight",
    1: "layer1.bias",
    2: "layer2.weight",
    3: "layer2.bias",
}
model.SaveModelWeights("model.safetensors", params, nameMap)
```

### Load model weights

```go
loadMap := map[string]int{
    "layer1.weight": 0,
    "layer1.bias":   1,
    "layer2.weight": 2,
    "layer2.bias":   3,
}
model.LoadModelWeights("model.safetensors", params, loadMap)
```

### Load pretrained safetensors

```go
sf, err := model.LoadSafetensors("pretrained.safetensors")
// sf.Tensors["model.layers.0.weight"] → *gorch.Tensor
// sf.Names → sorted list of tensor names
```

Supports F32, F16, and BF16 (auto-converted to F32).

## Tokenizer

### BPE tokenizer (GPT-2 compatible)

```go
tok, err := model.LoadTokenizer("vocab.json", "merges.txt")
ids := tok.Encode("Hello world")     // []int{15496, 995}
text := tok.Decode(ids)               // "Hello world"
```

### Character-level tokenizer (for testing)

```go
tok := model.NewSimpleTokenizer("the quick brown fox")
ids := tok.Encode("fox")
text := tok.Decode(ids)  // "fox"
```

## Data loading

### Built-in datasets

```go
// MNIST digits
trainSet, _ := data.LoadMNIST("./cache", true)

// Fashion-MNIST clothing
trainSet, _ := data.LoadFashionMNIST("./cache", true)

// UCI Wine Quality
wineDS, _ := data.LoadWineQuality("./cache")
wineDS.Normalize()
train, test := wineDS.TrainTestSplit(0.2)

// UCI Breast Cancer
bcDS, _ := data.LoadBreastCancer("./cache")
bcDS.Normalize()
train, test := bcDS.TrainTestSplit(0.2)
```

### Generic CSV loading

```go
ds, _ := data.LoadCSV("data.csv", ",", -1, true, nil)  // comma-sep, last col = label, skip header
ds.Normalize()
train, test := ds.TrainTestSplit(0.2)
```

### DataLoader

```go
loader := data.NewDataLoader(trainSet, 64, true)  // batch=64, shuffle=true

for epoch := 0; epoch < numEpochs; epoch++ {
    loader.Reset()
    for {
        inputs, targets := loader.Next()
        if inputs == nil { break }
        // ... train ...
    }
}
```

## Complete example: MNIST classifier

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
        nn.NewLinear(784, 128), nn.NewReLU(),
        nn.NewLinear(128, 10),
    )
    opt := optim.NewAdam(model.Parameters(), 0.001)
    loader := data.NewDataLoader(trainSet, 64, true)

    for epoch := 0; epoch < 3; epoch++ {
        loader.Reset()
        for {
            inputs, targets := loader.Next()
            if inputs == nil { break }
            opt.ZeroGrad()
            loss := g.CrossEntropyLoss(model.Forward(inputs), targets)
            loss.Backward()
            opt.Step()
        }
    }
    // Result: 97.2% accuracy in ~1 second on M4
}
```

## Complete example: GPT training

```go
package main

import (
    "fmt"
    g "github.com/vinq1911/gorch"
    "github.com/vinq1911/gorch/model"
    "github.com/vinq1911/gorch/nn"
    "github.com/vinq1911/gorch/optim"
)

func main() {
    corpus := "the quick brown fox jumps over the lazy dog"
    tok := model.NewSimpleTokenizer(corpus)

    gpt := nn.NewGPT(tok.VocabSize(), 64, 4, 2, 128)
    opt := optim.NewAdam(gpt.Parameters(), 0.001)

    // Train: predict next character
    ids := tok.Encode(corpus)
    for step := 0; step < 100; step++ {
        opt.ZeroGrad()
        logits := gpt.Forward(ids[:len(ids)-1])
        targets := make([]float32, len(ids)-1)
        for i := range targets {
            targets[i] = float32(ids[i+1])
        }
        loss := g.CrossEntropyLoss(logits,
            g.NewTensor(targets, len(targets), 1))
        loss.Backward()
        opt.Step()

        if step%20 == 0 {
            fmt.Printf("step %d  loss=%.4f\n", step, loss.Data()[0])
        }
    }
}
```

## Tips

- **Learning rate**: start with 0.001 for Adam, 0.01 for SGD
- **Batch size**: 32-128 is typical. Larger batches need higher learning rate
- **Dropout**: use 0.1-0.3 for regularization. Disable during evaluation
- **LR scheduling**: CosineAnnealing or WarmupCosine generally outperform constant LR
- **CrossEntropyLoss expects raw logits**: don't add Softmax as the last layer
- **ZeroGrad before forward**: always zero gradients at the start of each iteration
- **Save checkpoints**: use SaveModelWeights periodically during long training runs
- **Conv2d uses im2col + BLAS**: 1x1 convolutions skip im2col for zero data duplication
