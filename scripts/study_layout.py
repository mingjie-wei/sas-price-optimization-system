"""Verify the recorded path-only migration without relabeling historical training code."""
from pathlib import Path
import hashlib
import json
ROOT = Path(__file__).resolve().parents[1]

def training_source_digest(path):
    path = Path(path)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = ROOT / 'docs/provenance/study_layout.json'
    if not receipt.exists():
        return actual
    entry = json.loads(receipt.read_text())['source_relocations'].get(str(path.relative_to(ROOT)))
    # Only the exact reviewed relocation is equivalent to its original source.
    # Any later edit returns its new digest and fails the frozen protocol comparison.
    if entry and actual == entry['relocated_sha256']:
        return entry['original_sha256']
    return actual


def historical_source_path(path):
    """Keep frozen protocol keys stable when a source file moves."""
    relative = str(Path(path).relative_to(ROOT))
    receipt = ROOT / 'docs/provenance/study_layout.json'
    if not receipt.exists():
        return relative
    entry = json.loads(receipt.read_text())['source_relocations'].get(relative, {})
    return entry.get('historical_path', relative)
