# Real-World Training Report

**Framework:** gorch (Go + Apple Accelerate + Metal)
**Date:** 2026-04-13
**Hardware:** Apple Silicon (M-series)

## Executive Summary

Three real-world datasets tested to validate gorch as a production-capable ML framework:

| Dataset | Task | Best Accuracy | Training Time | Published Baseline |
|---------|------|--------------|---------------|-------------------|
| Wine Quality (Red) | 6-class | **65.3%** | 249ms | ~55-65% MLP |
| Breast Cancer Wisconsin (Diagnostic) | 2-class | **98.2%** | 20ms | ~95-97% MLP |
| Fashion-MNIST | 10-class | **88.0%** | 14.97s | ~88% MLP |

---

## Wine Quality (Red)

- **Samples:** 1599
- **Features:** 11
- **Classes:** 6

### Architecture Comparison

| Model | Params | Accuracy | Training Time | Final Loss |
|-------|--------|----------|---------------|------------|
| Small (11→32→6) | 582 | **59.69%** | 59ms | 0.8535 |
| Medium (11→64→32→6) | 3046 | **63.12%** | 101ms | 0.6217 |
| Large (11→128→64→32→6) | 12070 | **65.31%** | 249ms | 0.2859 |

### Per-Class Accuracy (Large (11→128→64→32→6))

| Class | Accuracy | Correct/Total |
|-------|----------|---------------|
| Quality 3 | 0.0% | 0/2 |
| Quality 4 | 7.1% | 1/14 |
| Quality 5 | 72.2% | 96/133 |
| Quality 6 | 76.9% | 100/130 |
| Quality 7 | 31.6% | 12/38 |
| Quality 8 | 0.0% | 0/3 |

### Training Loss Curve (Large (11→128→64→32→6))

| Epoch | Loss |
|-------|------|
| 1 | 1.4631 |
| 6 | 0.8606 |
| 11 | 0.7673 |
| 16 | 0.6826 |
| 21 | 0.6141 |
| 26 | 0.5538 |
| 31 | 0.4810 |
| 36 | 0.4238 |
| 41 | 0.3640 |
| 46 | 0.3117 |
| 50 | 0.2859 |

---

## Breast Cancer Wisconsin (Diagnostic)

- **Samples:** 569
- **Features:** 30
- **Classes:** 2

### Architecture Comparison

| Model | Params | Accuracy | Training Time | Final Loss |
|-------|--------|----------|---------------|------------|
| Small (30→16→2) | 530 | **98.25%** | 20ms | 0.0526 |
| Medium (30→64→32→2) | 4130 | **98.25%** | 38ms | 0.0064 |
| Large (30→128→64→32→2) | 14370 | **98.25%** | 151ms | 0.0006 |

### Per-Class Accuracy (Small (30→16→2))

| Class | Accuracy | Correct/Total |
|-------|----------|---------------|
| Benign | 98.8% | 85/86 |
| Malignant | 96.4% | 27/28 |

### Training Loss Curve (Small (30→16→2))

| Epoch | Loss |
|-------|------|
| 1 | 0.9509 |
| 6 | 0.2601 |
| 11 | 0.1682 |
| 16 | 0.1051 |
| 21 | 0.1059 |
| 26 | 0.0776 |
| 31 | 0.0734 |
| 36 | 0.0657 |
| 41 | 0.0588 |
| 46 | 0.0612 |
| 50 | 0.0526 |

---

## Fashion-MNIST

- **Samples:** 70000
- **Features:** 784
- **Classes:** 10

### Architecture Comparison

| Model | Params | Accuracy | Training Time | Final Loss |
|-------|--------|----------|---------------|------------|
| MLP (784→256→128→10) | 235146 | **87.98%** | 14.97s | 0.2272 |

### Per-Class Accuracy (MLP (784→256→128→10))

| Class | Accuracy | Correct/Total |
|-------|----------|---------------|
| T-shirt/top | 85.0% | 850/1000 |
| Trouser | 96.7% | 967/1000 |
| Pullover | 87.2% | 872/1000 |
| Dress | 89.0% | 890/1000 |
| Coat | 71.8% | 718/1000 |
| Sandal | 94.6% | 946/1000 |
| Shirt | 65.0% | 650/1000 |
| Sneaker | 98.3% | 983/1000 |
| Bag | 98.1% | 981/1000 |
| Ankle boot | 94.1% | 941/1000 |

### Training Loss Curve (MLP (784→256→128→10))

| Epoch | Loss |
|-------|------|
| 1 | 0.4820 |
| 2 | 0.3616 |
| 3 | 0.3255 |
| 4 | 0.3014 |
| 5 | 0.2842 |
| 6 | 0.2678 |
| 7 | 0.2543 |
| 8 | 0.2444 |
| 9 | 0.2344 |
| 10 | 0.2272 |
