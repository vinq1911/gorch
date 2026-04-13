//go:build darwin

package model

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
	"unicode/utf8"
)

// BPETokenizer implements byte-pair encoding tokenization.
// Compatible with GPT-2/GPT-NeoX style vocab.json + merges.txt.
type BPETokenizer struct {
	Encoder    map[string]int // token string → ID
	Decoder    map[int]string // ID → token string
	BPERanks   map[[2]string]int // merge pair → priority rank
	VocabSize  int
	ByteEncode map[byte]rune // byte → unicode char mapping
	ByteDecode map[rune]byte // unicode char → byte mapping
}

// LoadTokenizer loads a BPE tokenizer from vocab.json and merges.txt.
func LoadTokenizer(vocabPath, mergesPath string) (*BPETokenizer, error) {
	// Load vocab.json
	vocabData, err := os.ReadFile(vocabPath)
	if err != nil {
		return nil, fmt.Errorf("read vocab: %w", err)
	}
	var encoder map[string]int
	if err := json.Unmarshal(vocabData, &encoder); err != nil {
		return nil, fmt.Errorf("parse vocab: %w", err)
	}

	// Build decoder
	decoder := make(map[int]string, len(encoder))
	for k, v := range encoder {
		decoder[v] = k
	}

	// Load merges.txt
	mergesFile, err := os.Open(mergesPath)
	if err != nil {
		return nil, fmt.Errorf("open merges: %w", err)
	}
	defer mergesFile.Close()

	bpeRanks := make(map[[2]string]int)
	scanner := bufio.NewScanner(mergesFile)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)
	rank := 0
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "#") || line == "" {
			continue
		}
		parts := strings.SplitN(line, " ", 2)
		if len(parts) != 2 {
			continue
		}
		bpeRanks[[2]string{parts[0], parts[1]}] = rank
		rank++
	}

	tok := &BPETokenizer{
		Encoder:  encoder,
		Decoder:  decoder,
		BPERanks: bpeRanks,
		VocabSize: len(encoder),
	}
	tok.initByteEncoding()

	return tok, nil
}

// initByteEncoding sets up the byte-to-unicode mapping used by GPT-2 tokenizer.
func (t *BPETokenizer) initByteEncoding() {
	t.ByteEncode = make(map[byte]rune)
	t.ByteDecode = make(map[rune]byte)

	// GPT-2 byte encoder: maps bytes to unicode chars to avoid control characters
	n := 0
	for b := 0; b < 256; b++ {
		if (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF) {
			t.ByteEncode[byte(b)] = rune(b)
			t.ByteDecode[rune(b)] = byte(b)
		} else {
			t.ByteEncode[byte(b)] = rune(256 + n)
			t.ByteDecode[rune(256+n)] = byte(b)
			n++
		}
	}
}

// Encode converts text to token IDs.
func (t *BPETokenizer) Encode(text string) []int {
	// Split into words (simplified: split on spaces, keeping spaces as prefix)
	words := splitIntoWords(text)

	var tokens []int
	for _, word := range words {
		// Convert word bytes to unicode representation
		var encoded []rune
		for i := 0; i < len(word); i++ {
			encoded = append(encoded, t.ByteEncode[word[i]])
		}

		// Apply BPE merges
		bpeTokens := t.bpe(encoded)

		for _, tok := range bpeTokens {
			if id, ok := t.Encoder[tok]; ok {
				tokens = append(tokens, id)
			}
			// Unknown tokens are silently dropped
		}
	}
	return tokens
}

// Decode converts token IDs back to text.
func (t *BPETokenizer) Decode(ids []int) string {
	var parts []string
	for _, id := range ids {
		if tok, ok := t.Decoder[id]; ok {
			parts = append(parts, tok)
		}
	}
	joined := strings.Join(parts, "")

	// Convert unicode chars back to bytes
	var result []byte
	for _, r := range joined {
		if b, ok := t.ByteDecode[r]; ok {
			result = append(result, b)
		} else {
			// Fallback: just encode the rune as UTF-8
			var buf [4]byte
			n := utf8.EncodeRune(buf[:], r)
			result = append(result, buf[:n]...)
		}
	}
	return string(result)
}

// bpe applies byte-pair encoding merges to a sequence of unicode characters.
func (t *BPETokenizer) bpe(chars []rune) []string {
	// Start with individual characters as tokens
	word := make([]string, len(chars))
	for i, c := range chars {
		word[i] = string(c)
	}

	for {
		if len(word) < 2 {
			break
		}

		// Find the highest-priority (lowest rank) merge pair
		bestRank := -1
		bestIdx := -1
		for i := 0; i < len(word)-1; i++ {
			pair := [2]string{word[i], word[i+1]}
			if rank, ok := t.BPERanks[pair]; ok {
				if bestRank == -1 || rank < bestRank {
					bestRank = rank
					bestIdx = i
				}
			}
		}

		if bestIdx == -1 {
			break // No more merges possible
		}

		// Apply the merge
		merged := word[bestIdx] + word[bestIdx+1]
		newWord := make([]string, 0, len(word)-1)
		newWord = append(newWord, word[:bestIdx]...)
		newWord = append(newWord, merged)
		newWord = append(newWord, word[bestIdx+2:]...)
		word = newWord
	}

	return word
}

// splitIntoWords splits text into "words" for BPE processing.
// GPT-2 uses a regex pattern; this is a simplified version that handles
// common cases: splits on spaces (keeping the space as prefix of next word).
func splitIntoWords(text string) []string {
	if text == "" {
		return nil
	}

	var words []string
	current := ""

	for i := 0; i < len(text); i++ {
		c := text[i]
		if c == ' ' && current != "" {
			words = append(words, current)
			current = string(c) // space becomes prefix of next word
		} else {
			current += string(c)
		}
	}
	if current != "" {
		words = append(words, current)
	}
	return words
}

// SimpleTokenizer is a minimal character-level tokenizer for testing.
// Maps each unique byte to a token ID.
type SimpleTokenizer struct {
	CharToID map[byte]int
	IDToChar map[int]byte
	VocabSz  int
}

// NewSimpleTokenizer creates a character-level tokenizer from a text corpus.
func NewSimpleTokenizer(text string) *SimpleTokenizer {
	charSet := make(map[byte]bool)
	for i := 0; i < len(text); i++ {
		charSet[text[i]] = true
	}

	charToID := make(map[byte]int)
	idToChar := make(map[int]byte)
	id := 0
	// Sort for deterministic ordering
	var chars []byte
	for c := range charSet {
		chars = append(chars, c)
	}
	sort.Slice(chars, func(i, j int) bool { return chars[i] < chars[j] })

	for _, c := range chars {
		charToID[c] = id
		idToChar[id] = c
		id++
	}

	return &SimpleTokenizer{
		CharToID: charToID,
		IDToChar: idToChar,
		VocabSz:  id,
	}
}

func (t *SimpleTokenizer) Encode(text string) []int {
	ids := make([]int, len(text))
	for i := 0; i < len(text); i++ {
		ids[i] = t.CharToID[text[i]]
	}
	return ids
}

func (t *SimpleTokenizer) Decode(ids []int) string {
	var buf []byte
	for _, id := range ids {
		buf = append(buf, t.IDToChar[id])
	}
	return string(buf)
}

func (t *SimpleTokenizer) VocabSize() int { return t.VocabSz }
