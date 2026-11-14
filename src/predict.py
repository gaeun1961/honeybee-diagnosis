"""사진 1장 -> 응애·석고병·부저병 확률 + Grad-CAM 히트맵

실행: python src/predict.py 이미지경로 [--weights models/best.pt] [--out out/]
- 모델 구조·전처리는 train.py와 같음 (ResNet18, fc 3출력, 16:9 리사이즈)
- 확률은 클래스별 sigmoid (다중 라벨). 셋 다 0.5 미만이면 정상으로 판정
- Grad-CAM: layer4 출력 기준, 확률이 가장 높은 클래스. 원본 크기로 키워 겹친 그림을 out/{stem}_cam.jpg로 저장
"""
import argparse
from pathlib import Path

import matplotlib
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

from train import LABELS, MEAN, STD

ROOT = Path(__file__).resolve().parent.parent


def load_model(weights):
    ckpt = torch.load(weights, map_location="cpu")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(LABELS))
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), ckpt["size"]


def preprocess(size):
    """train.py 평가용과 같은 전처리: 16:9 리사이즈 (size x size*9/16) + ImageNet 정규화"""
    return transforms.Compose([transforms.Resize((round(size * 9 / 16), size)),
                               transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def grad_cam(model, x):
    """확률(=logit) 최고 클래스 기준 layer4 Grad-CAM. (probs, top, cam[h,w] 0~1) 반환"""
    feats = {}
    h = model.layer4.register_forward_hook(lambda m, i, o: feats.update(a=o))
    logits = model(x)[0]
    h.remove()
    top = int(logits.argmax())
    a = feats["a"]
    g, = torch.autograd.grad(logits[top], a)
    cam = torch.relu((g.mean((2, 3), keepdim=True) * a).sum(1))[0].detach()
    cam = cam / cam.max() if cam.max() > 0 else cam
    return torch.sigmoid(logits).detach().tolist(), top, cam.numpy()


def overlay(img, cam, alpha=0.45):
    cam = np.asarray(Image.fromarray(np.uint8(cam * 255)).resize(img.size, Image.BILINEAR)) / 255.0
    heat = Image.fromarray(np.uint8(matplotlib.colormaps["jet"](cam)[..., :3] * 255))
    return Image.blend(img, heat, alpha)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("--weights", type=Path, default=ROOT / "models/best.pt")
    ap.add_argument("--out", type=Path, default=Path("out"))
    a = ap.parse_args()

    model, size = load_model(a.weights)
    img = Image.open(a.image).convert("RGB")
    probs, top, cam = grad_cam(model, preprocess(size)(img).unsqueeze(0))

    for c, p in zip(LABELS, probs):
        print(f"{c}: {p:.3f}")
    found = [c for c, p in zip(LABELS, probs) if p >= 0.5]
    print("판정:", ", ".join(found) if found else "정상 (모두 0.5 미만)")

    a.out.mkdir(parents=True, exist_ok=True)
    path = a.out / f"{a.image.stem}_cam.jpg"
    overlay(img, cam).save(path, "JPEG", quality=90)
    print(f"Grad-CAM({LABELS[top]}) -> {path}")


if __name__ == "__main__":
    main()
