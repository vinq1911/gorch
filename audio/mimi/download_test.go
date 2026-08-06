//go:build darwin

package mimi

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestDownloadSkipsExisting verifies that a plausible existing
// checkpoint short-circuits the download (no network access).
func TestDownloadSkipsExisting(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "model.safetensors")
	// Sparse file at the real checkpoint size; no bytes actually written.
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := f.Truncate(384649828); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}

	got, err := Download(dir)
	if err != nil {
		t.Fatalf("Download with existing file: %v", err)
	}
	if got != path {
		t.Fatalf("Download returned %q, want %q", got, path)
	}
}

// TestDownloadRejectsTruncated verifies that an implausibly small
// existing file is reported instead of silently accepted.
func TestDownloadRejectsTruncated(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "model.safetensors")
	if err := os.WriteFile(path, []byte("not a checkpoint"), 0644); err != nil {
		t.Fatal(err)
	}

	_, err := Download(dir)
	if err == nil {
		t.Fatal("Download accepted a 16-byte checkpoint")
	}
	if !strings.Contains(err.Error(), "truncated") {
		t.Fatalf("unexpected error: %v", err)
	}
}
