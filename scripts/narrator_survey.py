#!/usr/bin/env python3
"""Survey all Wolne Lektury audiobooks to find narrators with most audio hours."""
import urllib.request, json, time
from collections import defaultdict

print("Fetching audiobook list...")
with urllib.request.urlopen("https://wolnelektury.pl/api/audiobooks/") as r:
    books = json.loads(r.read())
print(f"Total audiobooks: {len(books)}")

narrator_books = defaultdict(list)
errors = 0

for i, book in enumerate(books):
    if i % 100 == 0 and i > 0:
        print(f"  ...checked {i}/{len(books)}")
    try:
        with urllib.request.urlopen(book["href"], timeout=10) as r:
            detail = json.loads(r.read())
        audio_len = detail.get("audio_length", "?")
        for m in detail.get("media", []):
            if m.get("type") == "mp3" and m.get("artist"):
                narrator_books[m["artist"]].append({
                    "slug": book["slug"],
                    "title": book["title"],
                    "audio_length": audio_len,
                    "author": book.get("author", "?")
                })
                break
    except Exception as e:
        errors += 1
    time.sleep(0.05)

print(f"\nDone. Errors: {errors}")
print(f"Unique narrators: {len(narrator_books)}")

def parse_duration(s):
    if not s or s == "?":
        return 0
    parts = s.split(":")
    if len(parts) == 3:
        return int(parts[0])*3600 + int(parts[1])*60 + int(parts[2])
    elif len(parts) == 2:
        return int(parts[0])*60 + int(parts[1])
    return 0

narrator_hours = []
for name, bks in narrator_books.items():
    total_sec = sum(parse_duration(b["audio_length"]) for b in bks)
    narrator_hours.append((name, len(bks), total_sec, bks))

narrator_hours.sort(key=lambda x: -x[2])

header = f"{'Narrator':<35} {'Books':>5} {'Hours':>10}"
print(f"\n{header}")
print("-" * 52)
for name, count, secs, bks in narrator_hours[:30]:
    hrs = secs / 3600
    print(f"{name:<35} {count:>5} {hrs:>9.1f}h")

with open("/root/narrator_survey.json", "w") as f:
    json.dump(
        {name: {"books": bks, "total_seconds": secs}
         for name, count, secs, bks in narrator_hours},
        f, ensure_ascii=False, indent=2
    )
print("\nFull data saved to /root/narrator_survey.json")
