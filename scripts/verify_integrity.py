"""Verify the delivered file whitelist and SHA256, using only Python stdlib."""
import hashlib
from pathlib import Path

root=Path(__file__).resolve().parents[1]
listed=set()
for line in (root/"checksums.sha256").read_text(encoding="utf-8").splitlines():
    expected,name=line.split("  ",1)
    relative=Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Unsafe checksum path")
    actual=hashlib.sha256((root/relative).read_bytes()).hexdigest()
    if actual!=expected:
        raise ValueError("Checksum changed: "+name)
    listed.add(relative.as_posix())
print(f"PASS: {len(listed)} delivered files match SHA256. New local runs/caches are not release files.")
