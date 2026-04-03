"""
setup_dataset.py — Extract and organise the autism image dataset.

Steps:
  1. Extract the zip into  backend/dataset/
  2. Print the raw folder structure (2 levels deep)
  3. Detect whether images are already in class sub-folders
  4. If not, attempt to sort them by common naming patterns
     (e.g. filenames starting with "autistic" / "non_autistic")
  5. Print a per-class image count summary
"""

import os
import re
import shutil
import zipfile
from pathlib import Path
from collections import defaultdict

# ── Paths ──────────────────────────────────────────────────────────────────
ZIP_PATH    = Path(r"C:\Users\hp\Downloads\Compressed\archive.zip")
DATASET_DIR = Path(__file__).parent / "dataset"

# Target class directories
CLASS_DIRS = {
    "autistic":     DATASET_DIR / "autistic",
    "non_autistic": DATASET_DIR / "non_autistic",
}

# Accepted image extensions
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

# ── Helpers ────────────────────────────────────────────────────────────────

def print_tree(root: Path, max_depth: int = 2, _depth: int = 0) -> None:
    """Recursively print directory tree up to max_depth."""
    if _depth > max_depth:
        return
    indent = "  " * _depth
    print(f"{indent}{'📁' if root.is_dir() else '📄'} {root.name}/")
    if root.is_dir():
        children = sorted(root.iterdir())
        # Show first 20 entries to avoid flooding the console
        for i, child in enumerate(children):
            if i == 20:
                print(f"{indent}  … ({len(children) - 20} more)")
                break
            print_tree(child, max_depth, _depth + 1)


def count_images(folder: Path) -> int:
    """Return the total number of image files inside a folder (recursive)."""
    return sum(
        1 for f in folder.rglob("*")
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS
    )


def find_class_folders(root: Path) -> dict[str, Path]:
    """
    Search root for sub-folders whose names suggest a class label.
    Returns a dict mapping normalised label → Path.
    """
    mapping = {}
    for p in root.rglob("*"):
        if not p.is_dir():
            continue
        name_lower = p.name.lower().replace(" ", "_").replace("-", "_")
        if name_lower in ("autistic", "autism"):
            mapping["autistic"] = p
        elif name_lower in ("non_autistic", "non_autism", "nonautistic",
                            "not_autistic", "normal"):
            mapping["non_autistic"] = p
    return mapping


def guess_class_from_filename(name: str) -> str | None:
    """
    Try to infer a class label from a filename.
    Returns 'autistic', 'non_autistic', or None.
    """
    n = name.lower()
    if re.search(r"non[_\-\s]?autis", n):
        return "non_autistic"
    if re.search(r"autis", n):
        return "autistic"
    return None


# ── Step 1: Extract ────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1 — Extracting zip file")
print("=" * 60)

if not ZIP_PATH.exists():
    raise FileNotFoundError(f"Zip not found: {ZIP_PATH}")

DATASET_DIR.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(ZIP_PATH, "r") as zf:
    total = len(zf.namelist())
    print(f"  Archive contains {total} entries.")
    zf.extractall(DATASET_DIR)

print(f"  Extracted to: {DATASET_DIR}\n")


# ── Step 2: Inspect structure ──────────────────────────────────────────────
print("=" * 60)
print("STEP 2 — Folder structure (2 levels deep)")
print("=" * 60)
print_tree(DATASET_DIR, max_depth=2)
print()


# ── Step 3 & 4: Organise images ────────────────────────────────────────────
print("=" * 60)
print("STEP 3 — Organising into class folders")
print("=" * 60)

# Check if class folders already exist with images
existing = find_class_folders(DATASET_DIR)

if len(existing) == 2:
    print("  ✅ Class folders already exist:")
    for label, path in existing.items():
        n = count_images(path)
        print(f"     {label:15s} → {path}  ({n} images)")
else:
    print("  Class folders not found (or incomplete). Attempting to organise…\n")

    # Collect every image file anywhere under dataset/
    all_images = [
        f for f in DATASET_DIR.rglob("*")
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS
        # Skip files already inside our target class dirs
        and not any(f.is_relative_to(d) for d in CLASS_DIRS.values())
    ]
    print(f"  Found {len(all_images)} total image(s) to process.\n")

    # Try to auto-classify by folder name, then by filename
    moved   = defaultdict(int)
    unknown = []

    for img in all_images:
        # 1. Check parent folder name
        parent_lower = img.parent.name.lower().replace(" ", "_").replace("-", "_")
        if parent_lower in ("autistic", "autism"):
            label = "autistic"
        elif parent_lower in ("non_autistic", "non_autism", "nonautistic",
                              "not_autistic", "normal"):
            label = "non_autistic"
        else:
            # 2. Check filename
            label = guess_class_from_filename(img.name)

        if label:
            dest_dir = CLASS_DIRS[label]
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_file = dest_dir / img.name
            # Avoid name collisions
            if dest_file.exists():
                dest_file = dest_dir / f"{img.stem}_{moved[label]}{img.suffix}"
            shutil.move(str(img), str(dest_file))
            moved[label] += 1
        else:
            unknown.append(img)

    if moved:
        print("  Moved by auto-detection:")
        for label, count in moved.items():
            print(f"     {label:15s} → {count} image(s)")

    if unknown:
        print(f"\n  ⚠️  {len(unknown)} image(s) could not be classified automatically.")
        print("  They remain in their original sub-folders inside dataset/.")
        print("  First 10 unclassified files:")
        for f in unknown[:10]:
            print(f"     {f.relative_to(DATASET_DIR)}")


# ── Step 5: Final summary ──────────────────────────────────────────────────
print()
print("=" * 60)
print("STEP 4 — Final image count per class")
print("=" * 60)

for label, path in CLASS_DIRS.items():
    if path.exists():
        n = count_images(path)
        print(f"  {label:15s}: {n:>5} images   ({path})")
    else:
        print(f"  {label:15s}: folder not found / no images placed")

print()
print("Done. Dataset is ready for training.")
