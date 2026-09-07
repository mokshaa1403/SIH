#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, random, sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from satquery.models.unet_water import build_unet

SEED = 26167


def resolve(value, data_root):
    p = Path(value)
    if p.exists(): return p
    parts = Path(*p.parts[1:]) if p.parts and p.parts[0] == "datasetsss" else p
    return data_root / parts


def mask(path, source, size):
    a = np.asarray(Image.open(path).convert("RGB").resize((size, size), Image.Resampling.NEAREST))
    if source == "water_bodies": return a.mean(2) >= 127
    return (a[..., 2] > 220) & (a[..., 0] < 35) & (a[..., 1] < 35)


class WaterDataset(Dataset):
    def __init__(self, rows, root, size, augment=False): self.rows, self.root, self.size, self.augment = rows, root, size, augment
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        row = self.rows[i]; image = Image.open(resolve(row["image"], self.root)).convert("RGB").resize((self.size, self.size), Image.Resampling.BILINEAR)
        target = Image.fromarray(mask(resolve(row["mask"], self.root), "water_bodies" if row["source"] == "water_bodies" else "landcover", self.size))
        if self.augment:
            if random.random() < .5: image, target = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT), target.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if random.random() < .5: image, target = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM), target.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            image = ImageEnhance.Brightness(image).enhance(random.uniform(.72, 1.25))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(.8, 1.2))
        x = torch.from_numpy(np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0)
        y = torch.from_numpy(np.asarray(target, dtype=np.float32)).unsqueeze(0)
        return x, y


def score(model, loader, device, threshold=.5):
    tp=fp=fn=tn=0
    model.eval()
    with torch.no_grad():
        for x,y in loader:
            p=torch.sigmoid(model(x.to(device))).cpu()>=threshold; y=y>=.5
            tp+=(p&y).sum().item(); fp+=(p&~y).sum().item(); fn+=(~p&y).sum().item(); tn+=(~p&~y).sum().item()
    return {"accuracy":(tp+tn)/max(1,tp+tn+fp+fn),"precision":tp/max(1,tp+fp),"recall":tp/max(1,tp+fn),"iou":tp/max(1,tp+fp+fn),"dice":2*tp/max(1,2*tp+fp+fn)}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--manifest",type=Path,default=Path("artifacts/water_split/train.csv")); p.add_argument("--data-root",type=Path,default=Path("datasetsss")); p.add_argument("--output",type=Path,default=Path("artifacts/water_unet.pt")); p.add_argument("--epochs",type=int,default=16); p.add_argument("--size",type=int,default=256); p.add_argument("--batch",type=int,default=8); args=p.parse_args()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    rows=list(csv.DictReader(args.manifest.open())); random.shuffle(rows); cut=max(1,int(len(rows)*.8)); train,val=rows[:cut],rows[cut:]
    device=torch.device("mps" if torch.backends.mps.is_available() else "cpu"); print(f"device={device} train={len(train)} validation={len(val)}",flush=True)
    train_loader=DataLoader(WaterDataset(train,args.data_root,args.size,True),batch_size=args.batch,shuffle=True,num_workers=0)
    val_loader=DataLoader(WaterDataset(val,args.data_root,args.size),batch_size=args.batch,num_workers=0)
    model=build_unet().to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=1e-4); bce=nn.BCEWithLogitsLoss()
    best=None
    for epoch in range(1,args.epochs+1):
        model.train(); total=0
        for x,y in train_loader:
            x,y=x.to(device),y.to(device); optimizer.zero_grad(); logits=model(x); probs=torch.sigmoid(logits)
            dice_loss=1-(2*(probs*y).sum((2,3))+1)/((probs+y).sum((2,3))+1); loss=bce(logits,y)+dice_loss.mean(); loss.backward(); optimizer.step(); total+=loss.item()
        metrics=score(model,val_loader,device); print(f"epoch={epoch} loss={total/len(train_loader):.4f} val={metrics}",flush=True)
        if best is None or metrics["dice"]>best[0]: best=(metrics["dice"],{k:v.detach().cpu() for k,v in model.state_dict().items()},metrics)
    model.load_state_dict(best[1]); threshold_scores=[]
    for threshold in np.arange(.30,.76,.05): threshold_scores.append((score(model,val_loader,device,float(threshold))["dice"],float(threshold)))
    _,threshold=max(threshold_scores); metrics=score(model,val_loader,device,threshold)
    args.output.parent.mkdir(parents=True,exist_ok=True); torch.save({"state_dict":best[1],"threshold":threshold,"input_size":args.size,"training_images":len(train),"validation_images":len(val),"validation_metrics":metrics},args.output)
    print(f"saved={args.output} threshold={threshold:.2f} metrics={metrics}",flush=True)

if __name__ == "__main__": main()
