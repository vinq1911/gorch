# Train a Tiny GPT Model with Gorch

Instructions for Cursor (or any AI coding assistant) to create and train a small GPT language model using the gorch framework, producing a safetensors file that can be used with fragmind-pigeon's distributed inference demo.

## Goal

Train a ~1-5M parameter GPT model on a readily available text dataset, save weights as safetensors, and verify it generates coherent text.

## Architecture

Use gorch's existing `nn.NewGPT()` with these hyperparameters:

```go
// ~1M params: good for quick training, fits in memory
gpt := nn.NewGPT(
    vocabSize,  // from BPE tokenizer (typically 256-1024 for tiny models)
    dim:      128,
    numHeads: 4,
    numLayers: 4,
    maxSeq:   128,
)

// ~5M params: better quality, still trains in minutes
gpt := nn.NewGPT(
    vocabSize,
    dim:      256,
    numHeads: 8,
    numLayers: 6,
    maxSeq:   256,
)
```

## Dataset Options (readily available, no auth needed)

Pick ONE of these:

### Option A: Shakespeare (tiny-shakespeare, ~1MB)
```
URL: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
Size: ~1MB, ~40,000 lines
Good for: Quick experiments, recognizable output style
```

### Option B: WikiText-2 (~13MB raw text)
```
URL: https://huggingface.co/datasets/wikitext/resolve/main/wikitext-2-raw-v1/train-00000-of-00001.parquet
Alternative: Download wikitext-2-raw-v1 via HuggingFace datasets
Good for: More diverse vocabulary, encyclopedic style
```

### Option C: TinyStories (~2GB, best quality but larger)
```
URL: https://huggingface.co/datasets/roneneldan/TinyStories
Good for: Best coherent output from small models (designed for this purpose)
Note: Large download, use a subset (first 100K stories)
```

**Recommended: Option A (Shakespeare)** — smallest, fastest, most recognizable.

## Implementation Steps

### Step 1: Create `cmd/train_tiny_gpt/main.go`

```go
//go:build darwin

package main

import (
    "fmt"
    "io"
    "math/rand"
    "net/http"
    "os"
    "time"

    g "github.com/vinq1911/gorch"
    "github.com/vinq1911/gorch/nn"
    "github.com/vinq1911/gorch/model"
    "github.com/vinq1911/gorch/optim"
)

const (
    dim       = 128
    heads     = 4
    layers    = 4
    maxSeq    = 128
    batchSeqs = 4     // number of sequences per batch (process sequentially)
    lr        = 3e-4
    epochs    = 3
    savePath  = "tiny-gpt.safetensors"
)
```

### Step 2: Download and tokenize data

```go
func downloadShakespeare(path string) error {
    if _, err := os.Stat(path); err == nil {
        return nil // already exists
    }
    resp, err := http.Get("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")
    if err != nil { return err }
    defer resp.Body.Close()
    f, err := os.Create(path)
    if err != nil { return err }
    defer f.Close()
    _, err = io.Copy(f, resp.Body)
    return err
}
```

For tokenizer, use `model.NewSimpleTokenizer(text)` (character-level) for simplicity. This gives ~65 unique chars for Shakespeare.

For better quality, implement a small BPE vocabulary:
1. Load the text
2. Build vocab from unique characters (or use `model.LoadTokenizer` with a pre-built vocab)
3. Encode entire text as `[]int`

### Step 3: Create data batches

```go
// Slice the encoded text into training sequences
func makeBatches(encoded []int, seqLen int) [][]int {
    var batches [][]int
    for i := 0; i+seqLen+1 <= len(encoded); i += seqLen {
        batches = append(batches, encoded[i:i+seqLen+1]) // +1 for target
    }
    return batches
}
```

Each batch: input = tokens[0:seqLen], target = tokens[1:seqLen+1].

### Step 4: Training loop

```go
func train(gpt *nn.GPT, batches [][]int, seqLen int) {
    opt := optim.NewAdam(gpt.Parameters(), lr)

    for epoch := 0; epoch < epochs; epoch++ {
        // Shuffle batches
        rand.Shuffle(len(batches), func(i, j int) {
            batches[i], batches[j] = batches[j], batches[i]
        })

        var totalLoss float64
        for i, batch := range batches {
            input := batch[:seqLen]
            target := batch[1 : seqLen+1]

            opt.ZeroGrad()
            logits := gpt.Forward(input) // (seqLen, vocabSize)

            // Build target tensor
            targetF := make([]float32, seqLen)
            for j, t := range target {
                targetF[j] = float32(t)
            }
            targetT := g.NewTensor(targetF, seqLen, 1)

            loss := g.CrossEntropyLoss(logits, targetT)
            loss.Backward()
            opt.Step()

            totalLoss += float64(loss.Data()[0])

            if (i+1) % 100 == 0 {
                fmt.Printf("  epoch %d batch %d/%d loss=%.4f\n",
                    epoch+1, i+1, len(batches), totalLoss/float64(i+1))
            }
        }
        fmt.Printf("Epoch %d/%d avg_loss=%.4f\n",
            epoch+1, epochs, totalLoss/float64(len(batches)))
    }
}
```

### Step 5: Save weights

```go
func saveWeights(gpt *nn.GPT, path string) error {
    params := gpt.Parameters()
    nameMap := make(map[int]string)

    // Name the parameters to match standard GPT naming
    idx := 0
    nameMap[idx] = "token_embed.weight"; idx++
    nameMap[idx] = "pos_embed.weight"; idx++
    for l := 0; l < layers; l++ {
        prefix := fmt.Sprintf("blocks.%d", l)
        nameMap[idx] = prefix + ".attn.wq.weight"; idx++
        nameMap[idx] = prefix + ".attn.wq.bias"; idx++
        nameMap[idx] = prefix + ".attn.wk.weight"; idx++
        nameMap[idx] = prefix + ".attn.wk.bias"; idx++
        nameMap[idx] = prefix + ".attn.wv.weight"; idx++
        nameMap[idx] = prefix + ".attn.wv.bias"; idx++
        nameMap[idx] = prefix + ".attn.wo.weight"; idx++
        nameMap[idx] = prefix + ".attn.wo.bias"; idx++
        nameMap[idx] = prefix + ".ffn1.weight"; idx++
        nameMap[idx] = prefix + ".ffn1.bias"; idx++
        nameMap[idx] = prefix + ".ffn2.weight"; idx++
        nameMap[idx] = prefix + ".ffn2.bias"; idx++
        nameMap[idx] = prefix + ".norm1.weight"; idx++
        nameMap[idx] = prefix + ".norm1.bias"; idx++
        nameMap[idx] = prefix + ".norm2.weight"; idx++
        nameMap[idx] = prefix + ".norm2.bias"; idx++
    }
    nameMap[idx] = "final_norm.weight"; idx++
    nameMap[idx] = "final_norm.bias"; idx++
    nameMap[idx] = "lm_head.weight"; idx++
    nameMap[idx] = "lm_head.bias"; idx++

    return model.SaveModelWeights(path, params, nameMap)
}
```

### Step 6: Generate text to verify

```go
func generate(gpt *nn.GPT, tok *model.SimpleTokenizer, prompt string, maxTokens int) string {
    ids := tok.Encode(prompt)
    for i := 0; i < maxTokens; i++ {
        seq := ids
        if len(seq) > maxSeq {
            seq = seq[len(seq)-maxSeq:]
        }
        logits := gpt.Forward(seq)
        seqLen := len(seq)
        vocabSize := tok.VocabSize()
        lastLogits := logits.Data()[(seqLen-1)*vocabSize : seqLen*vocabSize]

        // Temperature sampling
        next := sampleWithTemperature(lastLogits, 0.8)
        ids = append(ids, next)
    }
    return tok.Decode(ids)
}
```

### Step 7: Put it all together in main()

```go
func main() {
    // Download data
    downloadShakespeare("/tmp/shakespeare.txt")
    text, _ := os.ReadFile("/tmp/shakespeare.txt")

    // Tokenize
    tok := model.NewSimpleTokenizer(string(text))
    encoded := tok.Encode(string(text))
    fmt.Printf("Vocab: %d chars, %d tokens\n", tok.VocabSize(), len(encoded))

    // Create model
    gpt := nn.NewGPT(tok.VocabSize(), dim, heads, layers, maxSeq)
    fmt.Printf("Model: %d params\n", gpt.CountParameters())

    // Train
    batches := makeBatches(encoded, maxSeq)
    fmt.Printf("Training: %d batches × %d epochs\n", len(batches), epochs)
    train(gpt, batches, maxSeq)

    // Save
    saveWeights(gpt, savePath)
    fmt.Printf("Saved: %s\n", savePath)

    // Generate
    fmt.Println("\n--- Sample generation ---")
    fmt.Println(generate(gpt, tok, "ROMEO:\n", 200))
}
```

## Expected Results

- **Shakespeare, 4-layer/128-dim**: trains in ~5-10 minutes on M4. After 3 epochs, generates Shakespeare-ish text (correct character patterns, word-like tokens, line breaks in right places). Loss should drop from ~3.5 to ~1.5.

- **Shakespeare, 6-layer/256-dim**: trains in ~20-30 minutes. Noticeably better coherence. Loss should reach ~1.2.

## Using the Trained Model with Fragmind

After training, the safetensors file can be loaded for distributed inference:

```bash
# Train the model
CGO_ENABLED=1 go run ./cmd/train_tiny_gpt/

# Run distributed inference with fragmind
CGO_ENABLED=1 go run ./examples/gorch_gpt_distributed/ \
    -weights tiny-gpt.safetensors \
    -layers 4 -dim 128 -heads 4 \
    -prompt "ROMEO:" -gen 200
```

(The distributed demo needs a `-weights` flag added to load safetensors instead of random init.)

## Key Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `cmd/train_tiny_gpt/main.go` | **CREATE** | Training script (~250 lines) |
| `examples/gorch_gpt_distributed/main.go` | **MODIFY** | Add `-weights` flag for loading safetensors |

## Testing

```bash
# Build and run training
CGO_ENABLED=1 go run ./cmd/train_tiny_gpt/

# Verify output file
ls -la tiny-gpt.safetensors

# Test loading
CGO_ENABLED=1 go test ./model/ -run TestSaveThenLoad -v
```

## Notes

- Character-level tokenizer (SimpleTokenizer) is simple but limited. For better results, build a BPE vocabulary from the training data using `model.LoadTokenizer`.
- Training is sequential (one sequence at a time) since gorch doesn't batch across sequences. This is slow but correct.
- The model is tiny enough to train on CPU. Metal GPU helps with matmul but backward is CPU-only.
- Save the tokenizer vocabulary alongside the weights so inference can reconstruct it.
