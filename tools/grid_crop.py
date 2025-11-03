import argparse
import glob
import os
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T

transform = T.Compose([
    T.ToTensor(),
])

reverse_transforms = T.Compose([
    T.Lambda(lambda t: t * 255.),
    T.Lambda(lambda t: t.detach().cpu().numpy().astype(np.uint8)),
    T.ToPILImage(),
])


def grid_partition(x, grid_size=(8, 8)):
    # x: [B,C,H,W], grid_size=(tile_h, tile_w)
    B, C, H, W = x.shape
    gh, gw = grid_size
    x = x.view(B, C, gh, H // gh, gw, W // gw)
    x = x.permute(0, 3, 5, 2, 4, 1).contiguous().view(-1, gh, gw, C)
    return x


def cropped_data(glob_path, out_dir, tile_w, tile_h):
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for img_path in sorted(glob.glob(glob_path)):
        im = Image.open(img_path).convert('RGB')
        old_w, old_h = im.size
        new_w = int(np.ceil(old_w / tile_w)) * tile_w
        new_h = int(np.ceil(old_h / tile_h)) * tile_h
        canvas = Image.new('RGB', (new_w, new_h))
        canvas.paste(im, ((new_w - old_w) // 2, (new_h - old_h) // 2))
        image = transform(canvas)
        if image.shape[1] > image.shape[2]:
            # rotate to keep H<=W if desired
            image = torch.rot90(image, k=3, dims=[1, 2])
        t = image.unsqueeze(0)  # [1,C,H,W]
        tiles = grid_partition(t, (tile_h, tile_w))  # [N, th, tw, C]
        for tile in tiles:
            save_img = reverse_transforms(tile)
            save_img.save(os.path.join(out_dir, f"{n:08d}.png"))
            n += 1
        print(f"{img_path}: {tiles.shape[0]} tiles -> {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--glob', required=True, help="Glob for input images, e.g. '/data/HR/*.png'")
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--tile_w', type=int, default=256)
    ap.add_argument('--tile_h', type=int, default=256)
    args = ap.parse_args()
    cropped_data(args.glob, args.out_dir, args.tile_w, args.tile_h)
