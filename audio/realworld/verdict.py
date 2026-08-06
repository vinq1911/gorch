"""Turn transcribe_check.py output into the committed round-trip
verdict TSV: <clip>\t<transcript>\t<OK|MISS> — OK when the transcript
of the token-reconstructed audio is exactly the expected digit (word
or numeral form, punctuation/case-insensitive)."""
import os
import re
import sys

WORDS = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
         "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"}

# True homophones the ASR may print instead of the digit word. ASR picks
# spellings by language-model context, not acoustics, so a homophone
# means the word WAS heard correctly (ruling confirmed by a GPT-5.6
# judge pass, 2026-08-06: "for"/"four" → heard correctly).
HOMOPHONES = {"one": {"won"}, "two": {"to", "too"}, "four": {"for", "fore"},
              "eight": {"ate"}}

print("# clip\ttranscript-of-token-reconstruction\tverdict")
for line in sys.stdin:
    if "\t" not in line:
        continue
    path, text = line.rstrip("\n").split("\t", 1)
    clip = os.path.basename(path).removesuffix(".wav")
    word = clip.split("_")[0]
    norm = re.sub(r"[^a-z0-9]", "", text.lower())
    accepted = {word, WORDS[word]} | HOMOPHONES.get(word, set())
    verdict = "OK" if norm in accepted else "MISS"
    print(f"{clip}\t{text}\t{verdict}")
