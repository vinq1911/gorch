# Fashion-MNIST Training Report

**Framework:** gorch (Go + Apple Accelerate + Metal)
**Date:** 2026-04-13
**Hardware:** Apple Silicon (M-series)
**Dataset:** Fashion-MNIST (60,000 train / 10,000 test, 28x28 grayscale, 10 classes)

## Task

Classify grayscale images of clothing items into 10 categories:
T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot.

Fashion-MNIST is a drop-in replacement for MNIST that is significantly harder.
Published MLP baselines achieve ~87-89% accuracy; CNNs reach ~93%.

## Results: Architecture Comparison

| Model | Params | Test Accuracy | Training Time | Final Loss |
|-------|--------|--------------|---------------|------------|
| Small (784→128→10) | 101770 | **88.09%** | 2.172s | 0.2604 |
| Medium (784→256→128→10) | 235146 | **88.14%** | 4.646s | 0.2268 |
| Large (784→512→256→128→10) | 567434 | **88.10%** | 9.873s | 0.2180 |

## Training Curves (Loss per Epoch)

| Epoch | Small (784→128→10) | Medium (784→256→128→10) | Large (784→512→256→128→10) |
|-------|--------|--------|--------|
| 1 | 0.5483 | 0.5082 | 0.4866 |
| 2 | 0.4007 | 0.3693 | 0.3576 |
| 3 | 0.3648 | 0.3276 | 0.3174 |
| 4 | 0.3356 | 0.3018 | 0.2944 |
| 5 | 0.3212 | 0.2838 | 0.2768 |
| 6 | 0.3054 | 0.2682 | 0.2598 |
| 7 | 0.2902 | 0.2603 | 0.2465 |
| 8 | 0.2814 | 0.2457 | 0.2380 |
| 9 | 0.2734 | 0.2359 | 0.2295 |
| 10 | 0.2604 | 0.2268 | 0.2180 |

## Per-Class Accuracy (Large (784→512→256→128→10))

| Class | Name | Accuracy | Correct/Total |
|-------|------|----------|---------------|
| 0 | T-shirt/top | 83.1% | 831/1000 |
| 1 | Trouser | 97.1% | 971/1000 |
| 2 | Pullover | 88.7% | 887/1000 |
| 3 | Dress | 89.6% | 896/1000 |
| 4 | Coat | 70.6% | 706/1000 |
| 5 | Sandal | 96.2% | 962/1000 |
| 6 | Shirt | 67.7% | 677/1000 |
| 7 | Sneaker | 98.5% | 985/1000 |
| 8 | Bag | 96.8% | 968/1000 |
| 9 | Ankle boot | 92.7% | 927/1000 |

## Analysis

**Easiest class:** Sneaker (98.5%)
**Hardest class:** Shirt (67.7%)

### What gorch does well

- Full training loop works end-to-end: data loading, forward, loss, backward, optimizer step
- Accelerate BLAS makes CPU training fast (seconds, not minutes)
- Autograd correctly propagates gradients through multi-layer networks
- Accuracy is competitive with published MLP baselines
- DataLoader with shuffle provides proper stochastic training

### Current limitations

- No Conv2d — limited to MLP architectures (flattened pixels)
- No Dropout or BatchNorm — may overfit on harder datasets
- No learning rate scheduling — fixed LR throughout training
- No data augmentation — could improve generalization
- Shirt vs T-shirt/top confusion is expected — they look similar even to humans

### Comparison to published baselines

| Method | Accuracy | Source |
|--------|----------|--------|
| **gorch MLP (best)** | **88.1%** | This report |
| 2-layer MLP (256) | 87.1% | Zalando benchmark |
| 3-layer MLP (256+128) | 88.3% | Zalando benchmark |
| CNN (2 conv + 2 FC) | 91.6% | Zalando benchmark |
| ResNet-18 | 93.6% | Literature |
