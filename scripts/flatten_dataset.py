#!/usr/bin/env python3
"""
Flatten nested dataset folders into a single flat folder.

Edit the PAIRS list below to configure source -> destination mappings,
then run:

    python scripts/flatten_dataset.py

Or run a single pair:

    python scripts/flatten_dataset.py --src data/train_sharp --dst data/train_sharp_flat

The script copies files and prefixes filenames with their original subfolder name to
avoid collisions.
"""
import os
import shutil
import argparse
import glob

# --- Edit these path pairs to match your dataset layout ---
# Each tuple: (source_parent_dir, destination_flat_dir)
# Using absolute POSIX paths (WSL). Update if you want different targets.
PAIRS = [
    ("/home/tianyu/Pythonproject/SwinIR/data/train_sharp", "/home/tianyu/Pythonproject/ART/data/train_sharp_flat"),
    ("/home/tianyu/Pythonproject/SwinIR/data/train_blur", "/home/tianyu/Pythonproject/ART/data/train_blur_flat"),
]
# ---------------------------------------------------------

EXTS = ["png", "jpg", "jpeg", "bmp", "tif", "tiff"]


def flatten_one(src_parent, dst_dir, numbered=False, simple=False, overwrite=False):
    if not os.path.isdir(src_parent):
        print(f"Source parent not found: {src_parent}")
        return 0
    os.makedirs(dst_dir, exist_ok=True)

    # collect files in deterministic order
    files = []
    for sub in sorted(os.listdir(src_parent)):
        subdir = os.path.join(src_parent, sub)
        if not os.path.isdir(subdir):
            continue
        for ext in EXTS:
            pattern = os.path.join(subdir, f"*.{ext}")
            for f in sorted(glob.glob(pattern)):
                if os.path.isfile(f):
                    files.append((sub, f))

    total = len(files)
    if total == 0:
        print(f"No files found in {src_parent}")
        return 0

    width = len(str(total))
    count = 0
    # determine starting index
    start_idx = 1
    if numbered and simple and not overwrite:
        # continue numbering from existing max index to avoid overwriting
        existing = [os.path.basename(p) for p in os.listdir(dst_dir) if os.path.isfile(os.path.join(dst_dir, p))]
        max_idx = 0
        for name in existing:
            base, extn = os.path.splitext(name)
            if base.isdigit():
                try:
                    v = int(base)
                    if v > max_idx:
                        max_idx = v
                except Exception:
                    continue
        if max_idx > 0:
            start_idx = max_idx + 1

    for delta, (sub, f) in enumerate(files, start=0):
        idx = start_idx + delta
        fname = os.path.basename(f)
        base, extn = os.path.splitext(fname)
        if numbered:
            idx_str = str(idx).zfill(width)
            if simple:
                # produce filenames like 0001.png (preserve extension)
                dest_name = f"{idx_str}{extn}"
            else:
                # produce filenames like 0001-original.png
                dest_name = f"{idx_str}-{fname}"
        else:
            # default: no subfolder prefix, just original filename
            dest_name = fname

        dest = os.path.join(dst_dir, dest_name)
        if os.path.exists(dest):
            if numbered and simple and overwrite:
                # remove existing file to overwrite
                try:
                    os.remove(dest)
                except Exception:
                    pass
            elif numbered and simple and not overwrite:
                # avoid overwrite: increment index until free
                while os.path.exists(dest):
                    idx += 1
                    idx_str = str(idx).zfill(width)
                    dest = os.path.join(dst_dir, f"{idx_str}{extn}")
            else:
                base2, extn2 = os.path.splitext(dest)
                i = 1
                while os.path.exists(f"{base2}_{i}{extn2}"):
                    i += 1
                dest = f"{base2}_{i}{extn2}"
        shutil.copy2(f, dest)
        count += 1

    print(f"Flattened {count} files from {src_parent} -> {dst_dir} (numbered={numbered})")
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=None, help="single source parent dir to flatten")
    parser.add_argument("--dst", default=None, help="single destination flat dir")
    parser.add_argument("--all", action="store_true", help="run all pairs defined in the script")
    # default behavior: simple numbered overwrite so running the script is one-step
    parser.add_argument("--numbered", action="store_true", default=True, help="add sequential numbering to output filenames (default: True)")
    parser.add_argument("--simple", action="store_true", default=True, help="with --numbered, produce simple names like 0001.png (default: True)")
    parser.add_argument("--overwrite", action="store_true", default=True, help="overwrite existing outputs when numbering (default: True)")
    args = parser.parse_args()
    if args.src and args.dst:
        flatten_one(args.src, args.dst, numbered=args.numbered, simple=args.simple, overwrite=args.overwrite)
        return

    if args.all or (not args.src and not args.dst):
        total = 0
        for s, d in PAIRS:
            total += flatten_one(s, d, numbered=args.numbered, simple=args.simple, overwrite=args.overwrite)
        print(f"Total files flattened: {total}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
