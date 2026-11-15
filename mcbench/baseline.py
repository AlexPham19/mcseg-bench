"""Small plain U-Net reference, independently implemented for this starter.

Not the historical PlainUNet or DinoUNet checkpoint graph. CPU smoke results
are software checks, not evidence of medical performance.
"""
from functools import lru_cache
from pathlib import Path
import random
import time

import numpy as np

from .io import array, relative, reference, sha, write_json


def normalize(image):
    x = np.asarray(image, dtype=np.float32)
    values = x[x > 0]
    if not len(values):
        return np.zeros_like(x)
    lo, hi = np.percentile(values, (1, 99.5))
    if hi <= lo:
        return np.zeros_like(x)
    return np.clip((x-lo)/(hi-lo), 0, 1).astype(np.float32)


def make_model(width=16):
    import torch
    from torch import nn
    class Block(nn.Sequential):
        def __init__(self, a, b):
            super().__init__(nn.Conv2d(a,b,3,padding=1), nn.GroupNorm(4,b), nn.ReLU(),
                             nn.Conv2d(b,b,3,padding=1), nn.GroupNorm(4,b), nn.ReLU())
    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.a, self.b, self.c = Block(1,width), Block(width,width*2), Block(width*2,width*4)
            self.d, self.e = Block(width*6,width*2), Block(width*3,width)
            self.head = nn.Conv2d(width,1,1)
        def forward(self,x):
            a = self.a(x)
            b = self.b(nn.functional.max_pool2d(a,2))
            c = self.c(nn.functional.max_pool2d(b,2))
            d = self.d(torch.cat((nn.functional.interpolate(c,size=b.shape[-2:],mode="bilinear",align_corners=False),b),1))
            return self.head(self.e(torch.cat((nn.functional.interpolate(d,size=a.shape[-2:],mode="bilinear",align_corners=False),a),1)))
    return UNet()


def positions(length, patch, stride):
    return sorted(set(list(range(0, max(1, length-patch+1), stride)) + [max(0,length-patch)]))


def tiled(model, image, device, patch=256, stride=128, batch=8):
    import torch
    if patch < 8 or not 0 < stride <= patch:
        raise ValueError("Require patch>=8 and 0<stride<=patch")
    x = normalize(image)
    h,w = x.shape
    padded = np.pad(x, ((0,max(0,patch-h)),(0,max(0,patch-w))),mode="edge")
    coords = [(y,z) for y in positions(padded.shape[0],patch,stride) for z in positions(padded.shape[1],patch,stride)]
    # Positive window floor ensures edge pixels never receive zero total weight.
    window = np.maximum(np.outer(np.hanning(patch),np.hanning(patch)), 1e-3).astype(np.float32)
    total, weight = np.zeros_like(padded), np.zeros_like(padded)
    model.eval()
    with torch.inference_mode():
        for start in range(0,len(coords),batch):
            c = coords[start:start+batch]
            xb = torch.from_numpy(np.stack([padded[y:y+patch,z:z+patch] for y,z in c])[:,None]).to(device)
            probs = model(xb).sigmoid()[:,0].cpu().numpy()
            for (y,z),p in zip(c,probs):
                total[y:y+patch,z:z+patch] += p*window
                weight[y:y+patch,z:z+patch] += window
    return (total/weight)[:h,:w].astype(np.float32)


def train(rows, root, output, seed=42, epochs=50, steps=128, batch=4, patch=256, width=16, lr=.001, device="cpu"):
    import torch
    from .metrics import measure, aggregate
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Training output must be new or empty")
    if min(epochs,steps,batch) < 1 or patch < 8 or width % 4:
        raise ValueError("Positive epochs/steps/batch, patch>=8, width divisible by4 required")
    fitting = [r for r in rows if r["split"] == "train"]
    dev = [r for r in rows if r["split"] == "dev"]
    if not fitting or not dev:
        raise ValueError("Need train and dev splits")
    if any(r["label_status"] != "dense_verified" for r in fitting+dev):
        raise ValueError("Reference trainer requires audited dense_verified train/dev labels")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    rng = np.random.default_rng(seed)
    model = make_model(width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    out.mkdir(parents=True,exist_ok=True)
    @lru_cache(maxsize=4)
    def load(cid):
        row = next(r for r in fitting if r["case_id"] == cid)
        im, gt = normalize(array(relative(root,row["image"]))), reference(row,root)
        if im.shape != gt.shape:
            raise ValueError("Image/reference geometry differs")
        return im,gt,np.argwhere(gt)
    def sample():
        row = fitting[int(rng.integers(len(fitting)))]
        im,gt,points = load(row["case_id"])
        h,w = im.shape
        if len(points) and rng.random() < .5:
            cy,cx = points[int(rng.integers(len(points)))]
            y,x = cy-patch//2+int(rng.integers(-patch//4,patch//4+1)),cx-patch//2+int(rng.integers(-patch//4,patch//4+1))
        else:
            y,x = int(rng.integers(max(1,h-patch+1))),int(rng.integers(max(1,w-patch+1)))
        y,x = int(np.clip(y,0,max(0,h-patch))),int(np.clip(x,0,max(0,w-patch)))
        image,mask = im[y:y+patch,x:x+patch],gt[y:y+patch,x:x+patch]
        pads = ((0,patch-image.shape[0]),(0,patch-image.shape[1]))
        image,mask = np.pad(image,pads,mode="edge"),np.pad(mask,pads)
        if rng.random() < .5:
            image,mask = image[:,::-1],mask[:,::-1]
        if rng.random() < .5:
            image,mask = image[::-1],mask[::-1]
        return image.copy(),mask.astype(np.float32).copy()
    history, best = [], -1.
    started = time.time()
    for epoch in range(1,epochs+1):
        model.train()
        losses = []
        for _ in range(steps):
            pairs = [sample() for _ in range(batch)]
            images = torch.from_numpy(np.stack([p[0] for p in pairs])[:,None]).to(device)
            masks = torch.from_numpy(np.stack([p[1] for p in pairs])[:,None]).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            p = logits.sigmoid()
            dice = (2*(p*masks).sum((1,2,3))+1)/(p.sum((1,2,3))+masks.sum((1,2,3))+1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits,masks) + (1-dice).mean()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        evaluated = []
        for row in dev:
            im,gt = array(relative(root,row["image"])),reference(row,root)
            prob = tiled(model,im,device,patch,max(1,patch//2))
            evaluated.append(measure(gt,prob >= .5))
        score = aggregate(evaluated)["positive_macro_iou"]
        if score is None:
            raise ValueError("Need a positive dev reference for checkpoint selection")
        history.append(dict(epoch=epoch,train_loss=float(np.mean(losses)),dev_positive_macro_iou=score))
        if score > best:
            best = score
            torch.save(dict(state=model.state_dict(),width=width,patch=patch,epoch=epoch),out/"model.pt")
        print(f"epoch={epoch} loss={history[-1]['train_loss']:.5f} devIoU={score:.5f}",flush=True)
    write_json(out/"training.json",dict(seed=seed,epochs=epochs,steps=steps,batch=batch,patch=patch,width=width,lr=lr,
               preprocessing="per-image positive-pixel percentiles 1,99.5, no resizing",loss="BCE + soft Dice equal coefficients",
               checkpoint_selection="max dev foreground macro IoU at0.5; earliest tie",history=history,
               checkpoint_sha256=sha(out/"model.pt"),torch_version=torch.__version__,numpy_version=np.__version__,
               device=device,seconds=time.time()-started,train_images=len(fitting),dev_images=len(dev)))
    return out/"model.pt"


def predict(rows,root,checkpoint,out,split,device="cpu"):
    import torch
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Prediction output must be new or empty")
    state = torch.load(checkpoint,map_location=device,weights_only=True)
    model = make_model(state["width"]).to(device)
    model.load_state_dict(state["state"])
    out.mkdir(parents=True,exist_ok=True)
    for row in rows:
        if row["split"] != split:
            continue
        # Prediction never loads or examines a reference mask.
        p = tiled(model,array(relative(root,row["image"])),device,state["patch"],max(1,state["patch"]//2))
        np.save(out/(row["case_id"]+".npy"),p,allow_pickle=False)
    write_json(out/"run.json",dict(model_sha256=sha(checkpoint),pipeline_id="reference_unet_v1_perimage_p1_99.5_hann_halfstride",
                                  split=split,patch=state["patch"],device=device))
