//go:build darwin

package model

import (
	"fmt"
	"io"
	"net/http"
	"os"
)

func downloadIfMissing(path, url string) error {
	if _, err := os.Stat(path); err == nil {
		return nil
	}

	fmt.Printf("Downloading %s ...\n", url)
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return fmt.Errorf("HTTP %d for %s", resp.StatusCode, url)
	}

	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	_, err = io.Copy(f, resp.Body)
	return err
}
