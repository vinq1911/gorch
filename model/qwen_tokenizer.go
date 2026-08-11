//go:build darwin

package model

import (
	"bufio"
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"unicode"
	"unicode/utf8"

	"golang.org/x/text/unicode/norm"
)

// QwenTokenizer implements byte-level BPE tokenization compatible with HF Qwen2Tokenizer semantics.
type QwenTokenizer struct {
	vocab        map[string]int
	idToToken    map[int]string
	bpeRanks     map[[2]string]int
	addedByID    map[int]string
	addedByToken map[string]int
	specials     []qwenSpecialToken

	byteEncoder map[byte]rune
	byteDecoder map[rune]byte

	cache   map[string][]string
	cacheMu sync.RWMutex
}

type qwenSpecialToken struct {
	content string
	id      int
}

// LoadQwenTokenizer loads tokenizer files from a Hugging Face tokenizer directory.
func LoadQwenTokenizer(dir string) (*QwenTokenizer, error) {
	t := &QwenTokenizer{
		vocab:        map[string]int{},
		idToToken:    map[int]string{},
		bpeRanks:     map[[2]string]int{},
		addedByID:    map[int]string{},
		addedByToken: map[string]int{},
		cache:        map[string][]string{},
	}
	t.byteEncoder, t.byteDecoder = qwenBytesToUnicode()

	if err := qwenLoadVocab(filepath.Join(dir, "vocab.json"), t); err != nil {
		return nil, err
	}
	if err := qwenLoadMerges(filepath.Join(dir, "merges.txt"), t); err != nil {
		return nil, err
	}
	if err := qwenLoadTokenizerConfig(filepath.Join(dir, "tokenizer_config.json"), t); err != nil {
		return nil, err
	}

	t.specials = make([]qwenSpecialToken, 0, len(t.addedByToken))
	for s, id := range t.addedByToken {
		t.specials = append(t.specials, qwenSpecialToken{content: s, id: id})
	}
	sort.Slice(t.specials, func(i, j int) bool {
		if len(t.specials[i].content) != len(t.specials[j].content) {
			return len(t.specials[i].content) > len(t.specials[j].content)
		}
		return t.specials[i].content < t.specials[j].content
	})

	return t, nil
}

// Encode tokenizes text into token ids.
func (t *QwenTokenizer) Encode(text string) []int {
	var out []int
	segments := t.qwenSplitSpecial(text)
	for _, seg := range segments {
		if seg.specialID >= 0 {
			out = append(out, seg.specialID)
			continue
		}
		normed := norm.NFC.String(seg.text)
		parts := qwenPreTokenize(normed)
		for _, p := range parts {
			mapped := t.qwenMapBytesToUnicode(p)
			bpe := t.qwenBPE(mapped)
			for _, piece := range bpe {
				if id, ok := t.vocab[piece]; ok {
					out = append(out, id)
				}
			}
		}
	}
	return out
}

// Decode detokenizes ids back to text.
func (t *QwenTokenizer) Decode(ids []int) string {
	var b []byte
	for _, id := range ids {
		if s, ok := t.addedByID[id]; ok {
			b = append(b, s...)
			continue
		}
		tok, ok := t.idToToken[id]
		if !ok {
			continue
		}
		for _, r := range tok {
			if bb, ok := t.byteDecoder[r]; ok {
				b = append(b, bb)
			} else {
				var tmp [utf8.UTFMax]byte
				n := utf8.EncodeRune(tmp[:], r)
				b = append(b, tmp[:n]...)
			}
		}
	}
	return string(b)
}

// VocabSize returns base vocab size plus added token count.
func (t *QwenTokenizer) VocabSize() int {
	return len(t.vocab) + len(t.addedByID)
}

type qwenSegment struct {
	text      string
	specialID int
}

func (t *QwenTokenizer) qwenSplitSpecial(s string) []qwenSegment {
	var out []qwenSegment
	for len(s) > 0 {
		bestPos := -1
		bestTok := ""
		bestID := -1
		for _, sp := range t.specials {
			pos := strings.Index(s, sp.content)
			if pos < 0 {
				continue
			}
			if bestPos == -1 || pos < bestPos || (pos == bestPos && len(sp.content) > len(bestTok)) {
				bestPos = pos
				bestTok = sp.content
				bestID = sp.id
			}
		}
		if bestPos == -1 {
			out = append(out, qwenSegment{text: s, specialID: -1})
			break
		}
		if bestPos > 0 {
			out = append(out, qwenSegment{text: s[:bestPos], specialID: -1})
		}
		out = append(out, qwenSegment{text: bestTok, specialID: bestID})
		s = s[bestPos+len(bestTok):]
	}
	return out
}

func (t *QwenTokenizer) qwenMapBytesToUnicode(s string) string {
	data := []byte(s)
	runes := make([]rune, len(data))
	for i, bb := range data {
		runes[i] = t.byteEncoder[bb]
	}
	return string(runes)
}

func (t *QwenTokenizer) qwenBPE(token string) []string {
	t.cacheMu.RLock()
	if v, ok := t.cache[token]; ok {
		out := make([]string, len(v))
		copy(out, v)
		t.cacheMu.RUnlock()
		return out
	}
	t.cacheMu.RUnlock()

	word := make([]string, 0, utf8.RuneCountInString(token))
	for _, r := range token {
		word = append(word, string(r))
	}
	if len(word) <= 1 {
		t.cacheMu.Lock()
		t.cache[token] = append([]string(nil), word...)
		t.cacheMu.Unlock()
		return word
	}

	for {
		bestRank := int(^uint(0) >> 1)
		bestI := -1
		var bestPair [2]string
		for i := 0; i < len(word)-1; i++ {
			p := [2]string{word[i], word[i+1]}
			if rank, ok := t.bpeRanks[p]; ok && rank < bestRank {
				bestRank = rank
				bestI = i
				bestPair = p
			}
		}
		if bestI < 0 {
			break
		}
		first, second := bestPair[0], bestPair[1]
		newWord := make([]string, 0, len(word))
		i := 0
		for i < len(word) {
			j := -1
			for k := i; k < len(word); k++ {
				if word[k] == first {
					j = k
					break
				}
			}
			if j == -1 {
				newWord = append(newWord, word[i:]...)
				break
			}
			newWord = append(newWord, word[i:j]...)
			i = j
			if i < len(word)-1 && word[i] == first && word[i+1] == second {
				newWord = append(newWord, first+second)
				i += 2
			} else {
				newWord = append(newWord, word[i])
				i++
			}
		}
		word = newWord
		if len(word) == 1 {
			break
		}
	}

	t.cacheMu.Lock()
	t.cache[token] = append([]string(nil), word...)
	t.cacheMu.Unlock()
	return word
}

func qwenLoadVocab(path string, t *QwenTokenizer) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	if err := json.Unmarshal(data, &t.vocab); err != nil {
		return err
	}
	for tok, id := range t.vocab {
		t.idToToken[id] = tok
	}
	return nil
}

func qwenLoadMerges(path string, t *QwenTokenizer) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1024), 1024*1024)

	rank := 0
	first := true
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if first {
			first = false
			if strings.HasPrefix(line, "#version:") {
				continue
			}
		}
		if line == "" {
			continue
		}
		parts := strings.Split(line, " ")
		if len(parts) != 2 {
			continue
		}
		t.bpeRanks[[2]string{parts[0], parts[1]}] = rank
		rank++
	}
	return sc.Err()
}

func qwenLoadTokenizerConfig(path string, t *QwenTokenizer) error {
	type addedEntry struct {
		Content string `json:"content"`
	}
	type cfg struct {
		Added map[string]addedEntry `json:"added_tokens_decoder"`
	}
	var c cfg
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	if err := json.Unmarshal(data, &c); err != nil {
		return err
	}
	for idStr, ent := range c.Added {
		id := 0
		for i := 0; i < len(idStr); i++ {
			id = id*10 + int(idStr[i]-'0')
		}
		t.addedByID[id] = ent.Content
		t.addedByToken[ent.Content] = id
	}
	return nil
}

func qwenPreTokenize(s string) []string {
	r := []rune(s)
	var out []string
	i := 0
	for i < len(r) {
		if tok, n := qwenTryA1(r, i); n > 0 {
			out = append(out, tok)
			i += n
			continue
		}
		if tok, n := qwenTryA2(r, i); n > 0 {
			out = append(out, tok)
			i += n
			continue
		}
		if tok, n := qwenTryA3(r, i); n > 0 {
			out = append(out, tok)
			i += n
			continue
		}
		if tok, n := qwenTryA4(r, i); n > 0 {
			out = append(out, tok)
			i += n
			continue
		}
		if tok, n := qwenTryA5(r, i); n > 0 {
			out = append(out, tok)
			i += n
			continue
		}
		if tok, n := qwenTryA6(r, i); n > 0 {
			out = append(out, tok)
			i += n
			continue
		}
		if tok, n := qwenTryA7(r, i); n > 0 {
			out = append(out, tok)
			i += n
			continue
		}
		out = append(out, string(r[i]))
		i++
	}
	return out
}

func qwenTryA1(r []rune, i int) (string, int) {
	if r[i] != '\'' {
		return "", 0
	}
	sufs := []string{"s", "t", "re", "ve", "m", "ll", "d"}
	for _, s := range sufs {
		if i+1+len(s) > len(r) {
			continue
		}
		ok := true
		for j := 0; j < len(s); j++ {
			c := r[i+1+j]
			if c >= 'A' && c <= 'Z' {
				c = c - 'A' + 'a'
			}
			if c != rune(s[j]) {
				ok = false
				break
			}
		}
		if ok {
			return string(r[i : i+1+len(s)]), 1 + len(s)
		}
	}
	return "", 0
}

func qwenTryA2(r []rune, i int) (string, int) {
	isL := func(x rune) bool { return unicode.IsLetter(x) }
	isN := func(x rune) bool { return unicode.IsNumber(x) }

	if i+1 < len(r) {
		c := r[i]
		if c != '\r' && c != '\n' && !isL(c) && !isN(c) && isL(r[i+1]) {
			j := i + 2
			for j < len(r) && isL(r[j]) {
				j++
			}
			return string(r[i:j]), j - i
		}
	}
	if isL(r[i]) {
		j := i + 1
		for j < len(r) && isL(r[j]) {
			j++
		}
		return string(r[i:j]), j - i
	}
	return "", 0
}

func qwenTryA3(r []rune, i int) (string, int) {
	if unicode.IsNumber(r[i]) {
		return string(r[i]), 1
	}
	return "", 0
}

func qwenTryA4(r []rune, i int) (string, int) {
	j := i
	if r[j] == ' ' {
		j++
		if j >= len(r) {
			return "", 0
		}
	}
	startP := j
	for j < len(r) && !unicode.IsSpace(r[j]) && !unicode.IsLetter(r[j]) && !unicode.IsNumber(r[j]) {
		j++
	}
	if j == startP {
		return "", 0
	}
	for j < len(r) && (r[j] == '\r' || r[j] == '\n') {
		j++
	}
	return string(r[i:j]), j - i
}

func qwenTryA5(r []rune, i int) (string, int) {
	if !unicode.IsSpace(r[i]) {
		return "", 0
	}
	j := i
	lastNL := -1
	for j < len(r) && unicode.IsSpace(r[j]) {
		if r[j] == '\r' || r[j] == '\n' {
			lastNL = j
		}
		j++
	}
	if lastNL < 0 {
		return "", 0
	}
	end := lastNL + 1
	return string(r[i:end]), end - i
}

func qwenTryA6(r []rune, i int) (string, int) {
	if !unicode.IsSpace(r[i]) {
		return "", 0
	}
	j := i
	for j < len(r) && unicode.IsSpace(r[j]) {
		j++
	}
	m := j - i
	if j == len(r) {
		return string(r[i:j]), m
	}
	if m >= 2 {
		return string(r[i : j-1]), m - 1
	}
	return "", 0
}

func qwenTryA7(r []rune, i int) (string, int) {
	if !unicode.IsSpace(r[i]) {
		return "", 0
	}
	j := i + 1
	for j < len(r) && unicode.IsSpace(r[j]) {
		j++
	}
	return string(r[i:j]), j - i
}

func qwenBytesToUnicode() (map[byte]rune, map[rune]byte) {
	var bs []byte
	var rs []rune
	// Ranges iterated as int: a byte loop variable can never exceed 0xFF, so
	// `b <= 0xFF` over byte would wrap around and never terminate.
	for b := int('!'); b <= int('~'); b++ {
		bs = append(bs, byte(b))
		rs = append(rs, rune(b))
	}
	for b := 0xA1; b <= 0xAC; b++ {
		bs = append(bs, byte(b))
		rs = append(rs, rune(b))
	}
	for b := 0xAE; b <= 0xFF; b++ {
		bs = append(bs, byte(b))
		rs = append(rs, rune(b))
	}

	used := make(map[byte]bool, len(bs))
	for _, b := range bs {
		used[b] = true
	}
	n := 0
	for b := 0; b <= 255; b++ {
		bb := byte(b)
		if used[bb] {
			continue
		}
		bs = append(bs, bb)
		rs = append(rs, rune(256+n))
		n++
	}

	enc := make(map[byte]rune, 256)
	dec := make(map[rune]byte, 256)
	for i := range bs {
		enc[bs[i]] = rs[i]
		dec[rs[i]] = bs[i]
	}
	return enc, dec
}
