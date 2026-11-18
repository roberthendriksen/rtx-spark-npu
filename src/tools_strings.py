"""Tiny `strings | grep` replacement: print distinct printable strings in a binary that match a regex."""
import mmap
import re
import sys

path, pat, limit = sys.argv[1], sys.argv[2].encode(), int(sys.argv[3]) if len(sys.argv) > 3 else 80
rx, prx, seen = re.compile(rb"[\x20-\x7e]{4,240}"), re.compile(pat, re.I), set()
with open(path, "rb") as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    for m in rx.finditer(mm):
        s = m.group()
        if prx.search(s) and s not in seen:
            seen.add(s)
            print(s.decode("ascii", "replace")[:200])
            if len(seen) >= limit:
                break
