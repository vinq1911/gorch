# Real-World Training Report

**Framework:** gorch (Go + Apple Accelerate + Metal)
**Date:** 2026-04-13
**Hardware:** Apple Silicon (M-series)

## Executive Summary

Three real-world datasets tested to validate gorch as a production-capable ML framework:

| Dataset | Task | Best Accuracy | Training Time | Published Baseline |
|---------|------|--------------|---------------|-------------------|
| Wine Quality (Red) | 6-class | **59.7%** | 243ms | ~55-65% MLP |
| Breast Cancer Wisconsin (Diagnostic) | 2-class | **97.4%** | 20ms | ~95-97% MLP |
| Fashion-MNIST | 10-class | **88.9%** | 14.874s | ~88% MLP |

---

## Wine Quality (Red)

- **Samples:** 1599
- **Features:** 11
- **Classes:** 6

### Architecture Comparison

| Model | Params | Accuracy | Training Time | Final Loss |
|-------|--------|----------|---------------|------------|
| Small (11→32→6) | 582 | **57.50%** | 64ms | 0.8603 |
| Medium (11→64→32→6) | 3046 | **59.38%** | 119ms | 0.6245 |
| Large (11→128→64→32→6) | 12070 | **59.69%** | 243ms | 0.2690 |

### Per-Class Accuracy (Large (11→128→64→32→6))

| Class | Accuracy | Correct/Total |
|-------|----------|---------------|
| Quality 3 | 0.0% | 0/1 |
| Quality 4 | 0.0% | 0/9 |
| Quality 5 | 73.9% | 99/134 |
| Quality 6 | 57.4% | 74/129 |
| Quality 7 | 39.1% | 18/46 |
| Quality 8 | 0.0% | 0/1 |

### Training Loss Curve (Large (11→128→64→32→6))

| Epoch | Loss |
|-------|------|
| 1 | 1.3952 |
| 6 | 0.8785 |
| 11 | 0.7578 |
| 16 | 0.6821 |
| 21 | 0.5898 |
| 26 | 0.5314 |
| 31 | 0.4580 |
| 36 | 0.3909 |
| 41 | 0.3597 |
| 46 | 0.2905 |
| 50 | 0.2690 |

---

## Breast Cancer Wisconsin (Diagnostic)

- **Samples:** 569
- **Features:** 30
- **Classes:** 2

### Architecture Comparison

| Model | Params | Accuracy | Training Time | Final Loss |
|-------|--------|----------|---------------|------------|
| Small (30→16→2) | 530 | **97.37%** | 20ms | 0.0365 |
| Medium (30→64→32→2) | 4130 | **97.37%** | 38ms | 0.0042 |
| Large (30→128→64→32→2) | 14370 | **96.49%** | 143ms | 0.0005 |

### Per-Class Accuracy (Small (30→16→2))

| Class | Accuracy | Correct/Total |
|-------|----------|---------------|
| Benign | 98.6% | 70/71 |
| Malignant | 95.3% | 41/43 |

### Training Loss Curve (Small (30→16→2))

| Epoch | Loss |
|-------|------|
| 1 | 0.7836 |
| 6 | 0.1962 |
| 11 | 0.1351 |
| 16 | 0.1031 |
| 21 | 0.0929 |
| 26 | 0.0644 |
| 31 | 0.0551 |
| 36 | 0.0490 |
| 41 | 0.0499 |
| 46 | 0.0398 |
| 50 | 0.0365 |

---

## Fashion-MNIST

- **Samples:** 70000
- **Features:** 784
- **Classes:** 10

### Architecture Comparison

| Model | Params | Accuracy | Training Time | Final Loss |
|-------|--------|----------|---------------|------------|
| MLP (784→256→128→10) | 235146 | **88.93%** | 14.874s | 0.2234 |

### Per-Class Accuracy (MLP (784→256→128→10))

| Class | Accuracy | Correct/Total |
|-------|----------|---------------|
| T-shirt/top | 84.7% | 847/1000 |
| Trouser | 97.4% | 974/1000 |
| Pullover | 79.4% | 794/1000 |
| Dress | 92.9% | 929/1000 |
| Coat | 81.0% | 810/1000 |
| Sandal | 97.0% | 970/1000 |
| Shirt | 67.1% | 671/1000 |
| Sneaker | 95.6% | 956/1000 |
| Bag | 97.6% | 976/1000 |
| Ankle boot | 96.6% | 966/1000 |

### Training Loss Curve (MLP (784→256→128→10))

| Epoch | Loss |
|-------|------|
| 1 | 0.4788 |
| 2 | 0.3605 |
| 3 | 0.3234 |
| 4 | 0.3014 |
| 5 | 0.2838 |
| 6 | 0.2679 |
| 7 | 0.2546 |
| 8 | 0.2430 |
| 9 | 0.2317 |
| 10 | 0.2234 |
