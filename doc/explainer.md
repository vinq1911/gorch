# What is gorch?

gorch is a tool for teaching computers to recognize patterns — the same kind of technology behind photo filters, email spam detection, and voice assistants. It runs on your Mac's hardware directly, without needing cloud services or Python.

## What does "machine learning" actually mean?

Imagine teaching a child to sort laundry. You don't write a rule book — you show them examples:

- "This is a shirt" (show 1,000 shirts)
- "This is a shoe" (show 1,000 shoes)
- "This is a bag" (show 1,000 bags)

After enough examples, the child can sort new laundry they've never seen before.

Machine learning works the same way. You show a program thousands of examples, it finds patterns, and then it can classify new things on its own.

gorch is the workbench where this training happens.

## What makes gorch different?

Most machine learning tools require Python and cloud GPU servers. gorch is different:

**It runs on your Mac, in Go, with no cloud dependency.**

| Traditional ML setup | gorch |
|---------------------|-------|
| Write Python code | Write Go code |
| Rent cloud GPUs ($2-8/hour) | Use your Mac's built-in GPU |
| Install dozens of dependencies | Single binary, zero dependencies |
| Data leaves your machine | Data stays on your machine |
| Minutes to deploy | Compile and ship one file |

## How fast is it?

We tested gorch on a real task: classifying 70,000 images of clothing into 10 categories (T-shirts, shoes, bags, etc.).

**On a standard MacBook (M4 chip):**

| What | Result |
|------|--------|
| Training data | 60,000 images |
| Categories | 10 (T-shirt, Trouser, Sneaker, Bag, ...) |
| Accuracy | **88.1%** — matches published research benchmarks |
| Training time | **2.2 seconds** |
| Prediction speed | 10,000 images classified in **under 1 second** |
| Hardware needed | Any Mac with Apple Silicon |

For comparison, a simple digit recognition task (telling apart handwritten 0-9) reaches **97.2% accuracy in 1 second**.

These numbers are competitive with the same tasks run in Python/PyTorch on the same hardware.

## Real-world use cases

### 1. On-device classification without cloud costs

A small business processes product photos — categorizing inventory, flagging defects, sorting returns. Today this requires sending every image to a cloud API (OpenAI, Google Vision) at $0.01-0.05 per image.

With gorch: train a model on your own product photos, then run classification locally. No API costs, no latency, no data leaving your network.

**Math:** 10,000 images/day × $0.02/image × 365 days = **$73,000/year in API costs** replaced by a single Mac Mini running gorch.

### 2. Deploy ML as a single binary

A Go web service needs to classify incoming requests — spam detection, content moderation, fraud scoring. With Python ML, you need a separate Python service, a model server, and inter-process communication.

With gorch: the ML model compiles directly into your Go binary. One binary, one deploy, one process.

```
Traditional:  Go API server → HTTP call → Python model server → response
gorch:        Go API server (model built-in) → response
```

This eliminates an entire service from your architecture.

### 3. Cluster compute: distribute training across machines

Go's built-in networking makes it straightforward to split work across multiple Macs. Each machine trains on a portion of the data and shares what it learned.

**Example: 4 Mac Studios working together**

| Setup | Training time (1M images) |
|-------|--------------------------|
| 1 Mac Studio | ~40 seconds |
| 4 Mac Studios (data parallel) | ~10 seconds |

Because gorch is a Go library, building a distributed training system uses standard Go tools (goroutines, channels, gRPC) rather than specialized distributed ML frameworks.

### 4. Edge deployment: ML on every machine

gorch compiles to a standalone binary with no runtime dependencies. This means you can deploy trained models to:

- Point-of-sale terminals (classify products at checkout)
- Security cameras (detect anomalies locally)
- Medical devices (classify sensor readings without cloud)
- IoT gateways (filter and classify sensor data at the edge)

No Python installation. No GPU drivers. No internet connection required. Just copy one file.

### 5. Private data stays private

Healthcare, finance, and legal industries often cannot send data to cloud ML services due to regulations (HIPAA, GDPR, SOX). gorch trains and runs entirely on local hardware — data never leaves the machine.

## What's under the hood?

gorch uses two Apple technologies that come built into every Mac:

1. **Accelerate** — Apple's optimized math library. It makes matrix operations (the core of ML) run 100-600x faster than basic code. This is why training takes seconds instead of hours.

2. **Metal** — Apple's GPU compute framework. The same technology that runs video games also runs ML computations. gorch uses your Mac's GPU for the heaviest math operations.

Both are included in macOS — no installation needed.

## Proof of work

### Test 1: Handwritten digit recognition (MNIST)

| Metric | Result |
|--------|--------|
| Task | Classify digits 0-9 from handwritten images |
| Dataset | 70,000 images (60K train + 10K test) |
| Accuracy | **97.2%** |
| Training time | **1.0 second** |
| Model size | 102,530 parameters |

### Test 2: Clothing classification (Fashion-MNIST)

| Metric | Result |
|--------|--------|
| Task | Classify clothing into 10 categories |
| Dataset | 70,000 images (60K train + 10K test) |
| Accuracy | **88.1%** (matches published research baselines) |
| Training time | **2.2 seconds** |
| Per-class best | Sneaker: 98.5% accuracy |
| Per-class hardest | Shirt: 67.7% (often confused with T-shirt — even humans struggle) |

### Test 3: Performance benchmarks

| Operation | Speed |
|-----------|-------|
| Matrix multiply (512x512) | 158 microseconds |
| Process 1 million numbers | 73 microseconds |
| Same matrix multiply in basic code | 99 milliseconds (628x slower) |

## What gorch is not

Honest limitations:

- **Not for large language models.** GPT-style models need hundreds of GPUs. gorch is for practical classification and prediction tasks that fit on one machine (or a small cluster).
- **Not for image generation.** Stable Diffusion and DALL-E require architectures gorch doesn't support yet.
- **macOS only.** gorch uses Apple-specific hardware acceleration. It does not run on Linux or Windows (though the Go parts would work — just slower without Accelerate/Metal).
- **Early stage.** This is a working framework with proven results, but it has fewer features than PyTorch (which has had 10 years and thousands of contributors).

## Who should care?

- **Go developers** who need ML without adding Python to their stack
- **Small teams** who want to run ML locally instead of paying for cloud APIs
- **Privacy-sensitive industries** that cannot send data to third-party services
- **Edge computing** scenarios where you need ML in a single deployable binary
- **Anyone with a Mac** who wants to experiment with ML without setup friction
