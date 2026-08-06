//go:build darwin

// Package mimi implements native inference for the kyutai/mimi audio
// codec encoder (plan doc/plans/0006-mimi-native-encoder.md).
package mimi

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
)

const checkpointURL = "https://huggingface.co/kyutai/mimi/resolve/main/model.safetensors"

// minCheckpointBytes is a sanity floor for the kyutai/mimi checkpoint
// (actual size: 384,649,828 bytes). A smaller file is a truncated or
// failed download.
const minCheckpointBytes = 300 << 20

// Download fetches the kyutai/mimi checkpoint into dir and returns the
// local path to model.safetensors. Skips the download if a plausible
// file is already present.
func Download(dir string) (string, error) {
	if err := os.MkdirAll(dir, 0755); err != nil {
		return "", err
	}
	path := filepath.Join(dir, "model.safetensors")

	if fi, err := os.Stat(path); err == nil {
		if fi.Size() >= minCheckpointBytes {
			return path, nil
		}
		return "", fmt.Errorf("existing %s is %d bytes (< %d): truncated download? remove it and retry",
			path, fi.Size(), minCheckpointBytes)
	}

	fmt.Printf("Downloading %s ...\n", checkpointURL)
	resp, err := http.Get(checkpointURL)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return "", fmt.Errorf("HTTP %d for %s", resp.StatusCode, checkpointURL)
	}

	// Download to a temp file and rename so a partial download never
	// masquerades as a complete checkpoint.
	tmp, err := os.CreateTemp(dir, "model.safetensors.download-*")
	if err != nil {
		return "", err
	}
	defer os.Remove(tmp.Name())

	n, err := io.Copy(tmp, resp.Body)
	if closeErr := tmp.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		return "", err
	}
	if n < minCheckpointBytes {
		return "", fmt.Errorf("downloaded %d bytes (< %d): truncated response", n, minCheckpointBytes)
	}
	if err := os.Rename(tmp.Name(), path); err != nil {
		return "", err
	}
	return path, nil
}
