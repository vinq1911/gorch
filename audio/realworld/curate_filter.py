"""Reads transcribe_check.py output on stdin; deletes any WAV whose
transcript is not exactly its expected digit (word or numeral form,
punctuation/case-insensitive). Prints what it rejects."""
import os
import re
import sys

WORDS = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
         "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"}

bad = 0
for line in sys.stdin:
    if "\t" not in line:
        continue
    path, text = line.rstrip("\n").split("\t", 1)
    word = os.path.basename(path).split("_")[0]
    norm = re.sub(r"[^a-z0-9]", "", text.lower())
    if norm not in (word, WORDS[word]):
        print(f"  reject {os.path.basename(path)}: {text!r}")
        os.remove(path)
        bad += 1
print(f"rejected {bad}")
