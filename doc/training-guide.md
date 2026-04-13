# Training with gorch

This guide walks through how to train neural networks using gorch — from basic concepts to a complete MNIST classifier.

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
    nn.NewLinear(128, 64),
    nn.NewReLU(),
    nn.NewLinear(64, 10),
)

// Forward pass
logits := model.Forward(input)  // input: (batch, 784), output: (batch, 10)

// All parameters from all layers
params := model.Parameters()
```

### Available layers

| Layer | Constructor | Description |
|-------|-------------|-------------|
| `Linear` | `nn.NewLinear(in, out)` | Fully connected, Kaiming init |
| `ReLU` | `nn.NewReLU()` | max(0, x) activation |
| `Sigmoid` | `nn.NewSigmoid()` | Sigmoid activation |
| `Tanh` | `nn.NewTanh()` | Tanh activation |
| `Sequential` | `nn.NewSequential(layers...)` | Chain layers in order |

## Loss functions

### MSELoss — for regression

Mean squared error: `mean((pred - target)^2)`

```go
pred := model.Forward(inputs)         // (batch, 1)
loss := g.MSELoss(pred, targets)       // scalar tensor
```

### CrossEntropyLoss — for classification

Takes raw logits (pre-softmax) and integer class labels:

```go
logits := model.Forward(inputs)                // (batch, numClasses)
targets := g.NewTensor(labels, batchSize, 1)   // integer labels as float32
loss := g.CrossEntropyLoss(logits, targets)     // scalar tensor
```

Internally applies log-softmax and negative log-likelihood. No need to add a softmax layer to your model.

## Optimizers

### SGD

```go
import "github.com/vinq1911/gorch/optim"

opt := optim.NewSGD(model.Parameters(), 0.01, 0.9)
// args: params, learning rate, momentum (0 for no momentum)
```

### Adam

```go
opt := optim.NewAdam(model.Parameters(), 0.001)
// uses default betas (0.9, 0.999) and eps (1e-8)
```

### Optimizer loop

```go
opt.ZeroGrad()   // clear gradients from previous step
// ... forward + loss + backward ...
opt.Step()       // update parameters using gradients
```

Always call `ZeroGrad()` before each forward pass. Gradients accumulate by default — without zeroing, you get the sum of all previous gradients.

## Data loading

### DataLoader

Batches a dataset with optional shuffling:

```go
import "github.com/vinq1911/gorch/data"

loader := data.NewDataLoader(dataset, 64, true)  // batch=64, shuffle=true

for epoch := 0; epoch < numEpochs; epoch++ {
    loader.Reset()  // reshuffle at start of each epoch
    for {
        inputs, targets := loader.Next()
        if inputs == nil {
            break  // epoch done
        }
        // ... train on this batch ...
    }
}
```

### MNIST dataset

Built-in MNIST reader with automatic download:

```go
trainSet, err := data.LoadMNIST("./mnist_data", true)   // training split
testSet, err := data.LoadMNIST("./mnist_data", false)    // test split

trainSet.Len()        // 60000
trainSet.InputShape() // [784]  (28x28 flattened, normalized to [0,1])
```

### Custom datasets

Implement the `data.Dataset` interface:

```go
type Dataset interface {
    Len() int
    Get(index int) (input, target []float32)
    InputShape() []int
    TargetShape() []int
}
```

Example:

```go
type MyDataset struct {
    xs [][]float32
    ys [][]float32
}

func (d *MyDataset) Len() int                          { return len(d.xs) }
func (d *MyDataset) InputShape() []int                 { return []int{4} }
func (d *MyDataset) TargetShape() []int                { return []int{1} }
func (d *MyDataset) Get(i int) ([]float32, []float32)  { return d.xs[i], d.ys[i] }
```

## Complete example: MNIST classifier

```go
package main

import (
    "fmt"
    "time"

    g "github.com/vinq1911/gorch"
    "github.com/vinq1911/gorch/data"
    "github.com/vinq1911/gorch/nn"
    "github.com/vinq1911/gorch/optim"
)

func main() {
    // 1. Load data
    trainSet, _ := data.LoadMNIST("./mnist_data", true)
    testSet, _ := data.LoadMNIST("./mnist_data", false)

    // 2. Define model
    model := nn.NewSequential(
        nn.NewLinear(784, 128),
        nn.NewReLU(),
        nn.NewLinear(128, 10),
    )

    // 3. Optimizer
    opt := optim.NewAdam(model.Parameters(), 0.001)

    // 4. Training loop
    loader := data.NewDataLoader(trainSet, 64, true)
    start := time.Now()

    for epoch := 0; epoch < 3; epoch++ {
        loader.Reset()
        var totalLoss float32
        batches := 0

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

            totalLoss += loss.Data()[0]
            batches++
        }

        fmt.Printf("Epoch %d  loss=%.4f  elapsed=%v\n",
            epoch+1, totalLoss/float32(batches), time.Since(start).Round(time.Second))
    }

    // 5. Evaluate
    testLoader := data.NewDataLoader(testSet, 256, false)
    testLoader.Reset()
    correct, total := 0, 0

    for {
        inputs, targets := testLoader.Next()
        if inputs == nil {
            break
        }

        logits := model.Forward(inputs)
        preds := logits.Data()
        tgts := targets.Data()
        batch := inputs.Shape()[0]

        for i := 0; i < batch; i++ {
            // argmax over classes
            best := 0
            for j := 1; j < 10; j++ {
                if preds[i*10+j] > preds[i*10+best] {
                    best = j
                }
            }
            if best == int(tgts[i]) {
                correct++
            }
            total++
        }
    }

    fmt.Printf("Test accuracy: %.2f%% (%d/%d)\n",
        float64(correct)/float64(total)*100, correct, total)
}
```

Expected output on Apple Silicon (M4, Accelerate-backed):

```
Epoch 1  loss=0.2950  elapsed=0.4s
Epoch 2  loss=0.1303  elapsed=0.7s
Epoch 3  loss=0.0903  elapsed=1.0s
Test accuracy: 97.22% (9722/10000)
```

## Metal GPU acceleration

Move tensors to Metal for GPU-accelerated compute:

```go
gpu, _ := g.InitMetal()

// Move data to GPU
a := g.Rand(1000, 1000).ToMetal(gpu.Dev)
b := g.Rand(1000, 1000).ToMetal(gpu.Dev)

// Ops automatically dispatch to Metal kernels
c := g.Add(a, b)        // element-wise on GPU
d := g.MatMul(a, b)     // MPS-accelerated matmul
e := g.ReLU(c)           // GPU kernel

// Move back to CPU when needed
e.ToCPU()
```

Unified memory means no explicit copies — Go writes data, the GPU reads it from the same physical address. This is Apple Silicon's key advantage.

## Tips

- **Learning rate**: start with 0.001 for Adam, 0.01 for SGD
- **Batch size**: 32-128 is typical. Larger batches are faster but may need higher learning rate
- **Kaiming init**: `nn.NewLinear` uses Kaiming initialization by default — good for ReLU networks
- **CrossEntropyLoss expects raw logits**: don't add Softmax as the last layer when using CrossEntropyLoss
- **ZeroGrad before forward**: always zero gradients at the start of each iteration, not the end
