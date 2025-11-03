import argparse, os, glob, math
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as T

to_tensor = T.ToTensor()
to_pil = T.ToPILImage()

def grid_partition(x, tile_h, tile_w):
    # x: [1,C,H,W]  →  [N, tile_h, tile_w, C]
    _, C, H, W = x.shape
    gh, gw = tile_h, tile_w
    x = x.view(1, C, gh, H//gh, gw, W//gw)
    x = x.permute(0, 3, 5, 2, 4, 1).contiguous().view(-1, gh, gw, C)
    return x

def pad_to_multiples(im, tile_w, tile_h, fill=(0,0,0)):
    w, h = im.size
    new_w = math.ceil(w / tile_w) * tile_w
    new_h = math.ceil(h / tile_h) * tile_h
    if new_w == w and new_h == h:
        return im, (0,0)
    canvas = Image.new(im.mode, (new_w, new_h), fill)
    canvas.paste(im, ((new_w - w)//2, (new_h - h)//2))
    return canvas, ((new_w - w)//2, (new_h - h)//2)

def tile_pair(img_path, lbl_path, out_img_dir, out_lbl_dir, tile_w, tile_h, overlap=0):
    img = Image.open(img_path).convert("RGB")
    lbl = Image.open(lbl_path).convert("RGB")  

    # لا نلفّ ولا نعمل Resize—Padding فقط
    img, _ = pad_to_multiples(img, tile_w, tile_h)
    lbl, _ = pad_to_multiples(lbl, tile_w, tile_h)

    it = to_tensor(img).unsqueeze(0)  # [1,C,H,W]
    lt = to_tensor(lbl).unsqueeze(0)


    tiles_i = grid_partition(it, tile_h, tile_w)  # [N, th, tw, C]
    tiles_l = grid_partition(lt, tile_h, tile_w)

    base = Path(img_path).stem
    N = tiles_i.shape[0]
    for n in range(N):
        pi = to_pil(tiles_i[n].permute(2,0,1))  # C,H,W
        pl = to_pil(tiles_l[n].permute(2,0,1))
        name = f"{base}_tile{n:04d}.png"
        pi.save(out_img_dir / name)
        pl.save(out_lbl_dir / name)

    return N

def process_split(root, out_root, split, tile_w, tile_h, overlap=0):
    img_dir = Path(root)/split/"cloud"
    lbl_dir = Path(root)/split/"label"
    out_img = Path(out_root)/split/"cloud"; out_img.mkdir(parents=True, exist_ok=True)
    out_lbl = Path(out_root)/split/"label"; out_lbl.mkdir(parents=True, exist_ok=True)

    total = 0
    exts = {".png",".jpg",".jpeg",".tif",".tiff",".bmp",".webp"}
    imgs = sorted([p for p in img_dir.glob("*") if p.is_file() and p.suffix.lower() in exts])

    for ip in imgs:
        lp = lbl_dir/ip.name
        if not lp.exists():
            print(f"[WARN]There is no equivalent for: {lp.name}")
            continue
        total += tile_pair(ip, lp, out_img, out_lbl, tile_w, tile_h, overlap)
    print(f"[{split}] batch has been created {total} batch.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dataset root containing train/test with cloud/label")
    ap.add_argument("--out_root", required=True, help="output root (e.g., dataset_grid)")
    ap.add_argument("--tile_w", type=int, default=128)
    ap.add_argument("--tile_h", type=int, default=128)
    ap.add_argument("--overlap", type=int, default=0)
    args = ap.parse_args()

    for sp in ["train","test"]:
        process_split(args.root, args.out_root, sp, args.tile_w, args.tile_h, args.overlap)
