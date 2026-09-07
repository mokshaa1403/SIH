#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from satquery.models.water_segmenter import predict_water

SIZE=(256,256)


def resolve_dataset_path(value: str, data_root: Path | None, project_root: Path) -> Path:
    original = Path(value)
    if original.is_absolute() and original.exists():
        return original
    relative = Path(*original.parts[1:]) if original.parts and original.parts[0] == "datasetsss" else original
    candidates = []
    if data_root is not None:
        candidates.append(data_root / relative)
    candidates.extend([
        Path.cwd() / original,
        project_root / original,
        project_root.parent / original,
        project_root.parent / "datasetsss" / relative,
    ])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = "\n  - ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Could not find dataset file '{value}'. Searched:\n  - {searched}\n"
        "Pass the dataset folder explicitly: --data-root /absolute/path/to/datasetsss"
    )


def truth(path, source):
    arr=np.asarray(Image.open(path).convert("RGB").resize(SIZE,Image.Resampling.NEAREST))
    if source=="water_bodies": return arr.mean(2)>=127
    return (arr[...,2]>220)&(arr[...,0]<35)&(arr[...,1]<35)


def main():
    p=argparse.ArgumentParser();p.add_argument("--manifest",type=Path,default=Path("artifacts/water_split/test.csv"));p.add_argument("--data-root",type=Path,default=None);p.add_argument("--limit",type=int,default=0);args=p.parse_args()
    project_root=Path(__file__).resolve().parents[1]
    if not args.manifest.is_absolute(): args.manifest=project_root/args.manifest
    rows=list(csv.DictReader(args.manifest.open())); rows=rows[:args.limit] if args.limit else rows
    tp=fp=fn=tn=0
    for i,row in enumerate(rows,1):
        image_path=resolve_dataset_path(row["image"],args.data_root,project_root)
        mask_path=resolve_dataset_path(row["mask"],args.data_root,project_root)
        image=Image.open(image_path).convert("RGB").resize(SIZE,Image.Resampling.BILINEAR)
        predicted=predict_water(image)[0]; actual=truth(mask_path,row["source"])
        tp+=int((predicted&actual).sum());fp+=int((predicted&~actual).sum());fn+=int((~predicted&actual).sum());tn+=int((~predicted&~actual).sum())
        if i%250==0: print(f"evaluated {i}/{len(rows)}")
    precision=tp/max(1,tp+fp);recall=tp/max(1,tp+fn);iou=tp/max(1,tp+fp+fn);dice=2*tp/max(1,2*tp+fp+fn);accuracy=(tp+tn)/max(1,tp+tn+fp+fn)
    print(f"images={len(rows)} accuracy={accuracy:.4f} precision={precision:.4f} recall={recall:.4f} IoU={iou:.4f} Dice={dice:.4f}")


if __name__=="__main__":main()
