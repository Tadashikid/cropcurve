"""Apply a batch of disease and/or pesticide rows to CropCurve's data files.

Usage (run from the project folder):
    python apply_rows.py plantvillage_diseases_rows.csv plantvillage_pesticides_rows.csv

Any file whose header starts with "id," is treated as disease rows; "disease_id," as pesticide rows.
- Disease rows REPLACE existing rows with the same id (or are added if new).
- Pesticide rows: every existing pesticide row for the diseases in the new file is removed
  first, then the new rows are added, so re-running never creates duplicates.
- Backups of the originals are saved as data/diseases.csv.bak and data/pesticides.csv.bak.
"""
import csv, shutil, sys
from pathlib import Path

DATA = Path("data")
TARGETS = {"id": DATA / "diseases.csv", "disease_id": DATA / "pesticides.csv"}


def read(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        return r.fieldnames, list(r)


def write(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: (row.get(k) or "") for k in header})


def main(files):
    if not DATA.is_dir():
        sys.exit("Run this from the CropCurve project folder (the one that contains data/).")
    for name in files:
        new_header, new_rows = read(name)
        key = new_header[0]
        if key not in TARGETS:
            sys.exit(f"{name}: first column must be 'id' or 'disease_id', found '{key}'.")
        target = TARGETS[key]
        header, rows = read(target)
        missing = [c for c in new_header if c not in header]
        if missing:
            sys.exit(f"{name}: columns {missing} are not in {target}. Nothing was changed.")
        shutil.copy(target, target.with_name(target.name + ".bak"))

        if key == "id":
            incoming = {r["id"]: r for r in new_rows}
            replaced = sum(1 for r in rows if r["id"] in incoming)
            out = [incoming.pop(r["id"]) if r["id"] in incoming else r for r in rows]
            out += list(incoming.values())
            print(f"{target}: {replaced} rows replaced, {len(incoming)} added ({len(out)} total).")
        else:
            covered = {r["disease_id"] for r in new_rows}
            out = [r for r in rows if r["disease_id"] not in covered]
            removed = len(rows) - len(out)
            out += new_rows
            print(f"{target}: {removed} old rows removed for {len(covered)} diseases, {len(new_rows)} added ({len(out)} total).")
        write(target, header, out)
    print("Done. Now run: python -m pytest -q   (backups are in data/*.bak)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
