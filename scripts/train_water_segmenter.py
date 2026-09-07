#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, json, pickle, random, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from satquery.models.water_segmenter import pixel_features

SEED = 26167
SIZE = (224, 224)


def water_body_pairs(root: Path):
    images, masks = root / "Water Bodies Dataset" / "Images", root / "Water Bodies Dataset" / "Masks"
    return [(p, masks/p.name, "water_bodies") for p in sorted(images.iterdir()) if p.suffix.lower() in {".jpg", ".png", ".jpeg"} and (masks/p.name).exists()]


def landcover_pairs(root: Path):
    folder = root / "archive (3)" / "train"
    colours = {"urban":(0,255,255), "agriculture":(255,255,0), "rangeland":(255,0,255), "forest":(0,255,0), "water":(0,0,255), "barren":(255,255,255)}
    pairs = []
    for image in sorted(folder.glob("*_sat.*")):
        mask = folder / f"{image.stem.removesuffix('_sat')}_mask.png"
        if not mask.exists(): continue
        arr = np.asarray(Image.open(mask).convert("RGB").resize((96,96), Image.Resampling.NEAREST))
        counts = {name:int(np.all(arr==colour,2).sum()) for name,colour in colours.items()}
        pairs.append((image, mask, "landcover_"+max(counts,key=counts.get)))
    return pairs


def mask_array(path: Path, source: str):
    arr = np.asarray(Image.open(path).convert("RGB").resize(SIZE, Image.Resampling.NEAREST))
    if source == "water_bodies": return arr.mean(2) >= 127
    return (arr[...,2]>220) & (arr[...,0]<35) & (arr[...,1]<35)


def sampled(image_path, mask_path, source, rng, per_class=1800):
    image = Image.open(image_path).convert("RGB").resize(SIZE, Image.Resampling.BILINEAR)
    y = mask_array(mask_path, source).reshape(-1); x = pixel_features(image); chosen=[]
    for value in (False, True):
        candidates=np.flatnonzero(y==value)
        if len(candidates): chosen.extend(rng.choice(candidates,min(per_class,len(candidates)),replace=False))
    chosen=np.asarray(chosen)
    return x[chosen], y[chosen].astype(np.uint8)


def main():
    parser=argparse.ArgumentParser(description="Train water segmentation from a small balanced subset; reserve all other pairs for testing.")
    parser.add_argument("--data",type=Path,default=Path("datasetsss")); parser.add_argument("--output",type=Path,default=Path("artifacts/water_segmenter.pkl"))
    parser.add_argument("--water-train",type=int,default=180); parser.add_argument("--per-landcover-class",type=int,default=20)
    args=parser.parse_args(); rng_py=random.Random(SEED); rng=np.random.default_rng(SEED)
    water=water_body_pairs(args.data); rng_py.shuffle(water); train=water[:args.water_train]; test=water[args.water_train:]
    groups=defaultdict(list)
    for pair in landcover_pairs(args.data): groups[pair[2]].append(pair)
    for label in sorted(groups):
        rng_py.shuffle(groups[label]); train.extend(groups[label][:args.per_landcover_class]); test.extend(groups[label][args.per_landcover_class:])
    xs,ys=[],[]
    for image,mask,source in train:
        x,y=sampled(image,mask,"water_bodies" if source=="water_bodies" else "landcover",rng); xs.append(x);ys.append(y)
    x_train,y_train=np.concatenate(xs),np.concatenate(ys)
    model=HistGradientBoostingClassifier(max_iter=140,learning_rate=.08,max_leaf_nodes=31,l2_regularization=1.2,random_state=SEED).fit(x_train,y_train)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    metadata={"version":"1.0","seed":SEED,"training_images":len(train),"reserved_test_images":len(test),"training_pixels":len(y_train),"water_pixel_fraction":round(float(y_train.mean()),4),"train_by_source":dict(sorted(Counter(p[2] for p in train).items()))}
    with args.output.open("wb") as handle: pickle.dump({"model":model,"metadata":metadata},handle)
    manifests=args.output.parent/"water_split"; manifests.mkdir(exist_ok=True)
    for name,rows in (("train.csv",train),("test.csv",test)):
        with (manifests/name).open("w",newline="") as handle:
            writer=csv.writer(handle);writer.writerow(["image","mask","source"]);writer.writerows((str(a),str(b),c) for a,b,c in rows)
    (manifests/"summary.json").write_text(json.dumps(metadata,indent=2));print(json.dumps(metadata,indent=2))


if __name__=="__main__": main()
