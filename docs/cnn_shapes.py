"""ResNet18(train.py와 같은 구성) 층별 출력 크기·파라미터 수를 손계산 공식과 비교 -> docs/cnn_shapes.md

실행: python docs/cnn_shapes.py
이미지·data/ 없이 랜덤 텐서 (1, 3, 512, 512)만 사용. 가중치는 크기 계산에 상관없어 weights=None.

출력 크기: W_out = floor((W - K + 2P) / S) + 1
- layer1~4는 첫 블록의 첫 3x3 conv(S=2면 다운샘플)가 크기를 정하고, 나머지 conv는 K=3, P=1, S=1이라 크기가 그대로
파라미터 수: K'(Kh^2 * Cin + 1)  (K' = 출력 채널 수, +1은 bias)
"""
from pathlib import Path

import torch
from torch import nn
from torchvision import models

N_CLASSES = 3  # train.py: 응애·석고병·부저병
H = W = 512

# 손계산에 쓰는 층 정의 (ResNet18 논문/구현 기준): 이름, K, S, P, 출력 채널
SPEC = [
    ("conv1", 7, 2, 3, 64),
    ("maxpool", 3, 2, 1, 64),
    ("layer1", 3, 1, 1, 64),
    ("layer2", 3, 2, 1, 128),
    ("layer3", 3, 2, 1, 256),
    ("layer4", 3, 2, 1, 512),
]


def out_size(w, k, s, p):
    return (w - k + 2 * p) // s + 1


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def main():
    torch.manual_seed(0)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, N_CLASSES)
    model.eval()

    shapes = {}
    hooks = [getattr(model, name).register_forward_hook(lambda m, i, o, name=name: shapes.__setitem__(name, tuple(o.shape)))
             for name, *_ in SPEC]
    with torch.no_grad():
        model(torch.randn(1, 3, H, W))
    for h in hooks:
        h.remove()

    L = ["# ResNet18 층별 출력 크기·파라미터 수", "",
         f"`python docs/cnn_shapes.py`로 생성. 입력: 랜덤 텐서 (1, 3, {H}, {W}), train.py와 같은 구성(ResNet18, fc {N_CLASSES}출력).", "",
         "## 출력 크기: hook 실측 vs 손계산", "",
         "손계산: W_out = ⌊(W − K + 2P) / S⌋ + 1. layer1~4는 첫 블록의 첫 3x3 conv 기준 (나머지 conv는 K=3, P=1, S=1이라 크기 유지).", "",
         "| 층 | K | S | P | 입력 W | 손계산 W_out | 손계산 (C, H, W) | 실측 (C, H, W) | 일치 |",
         "|---|---|---|---|---|---|---|---|---|"]
    w, mismatch = W, 0
    for name, k, s, p, c in SPEC:
        wo = out_size(w, k, s, p)
        calc, real = (c, wo, wo), shapes[name][1:]
        ok = calc == real
        mismatch += not ok
        L.append(f"| {name} | {k} | {s} | {p} | {w} | ⌊({w}−{k}+{2 * p})/{s}⌋+1 = {wo} | {calc} | {real} | {'✅' if ok else '❌ 불일치'} |")
        w = wo
    L += ["", f"불일치 {mismatch}개.", ""]

    L += ["## 층별 파라미터 수", "",
          "| 층 | 파라미터 수 |", "|---|---|"]
    names = ["conv1", "bn1", "maxpool", "layer1", "layer2", "layer3", "layer4", "fc"]
    for name in names:
        L.append(f"| {name} | {n_params(getattr(model, name)):,} |")
    L += [f"| **합계** | {n_params(model):,} |", ""]

    conv1 = model.conv1
    kp, kh, cin = conv1.out_channels, conv1.kernel_size[0], conv1.in_channels
    with_bias, no_bias = kp * (kh * kh * cin + 1), kp * kh * kh * cin
    real = n_params(conv1)
    L += ["## conv1 파라미터: 공식 vs 실제", "",
          f"K' = {kp}, Kh = {kh}, Cin = {cin}, bias = {conv1.bias is not None}", "",
          "| 계산 | 값 | 실제와 일치 |", "|---|---|---|",
          f"| 공식 K'(Kh²·Cin+1) = {kp}·({kh}²·{cin}+1) | {with_bias:,} | {'✅' if with_bias == real else '❌ 불일치'} |",
          f"| bias 없이 K'·Kh²·Cin = {kp}·{kh}²·{cin} | {no_bias:,} | {'✅' if no_bias == real else '❌ 불일치'} |",
          f"| 실제 (`conv1.parameters()`) | {real:,} | |", ""]
    if with_bias != real:
        L.append(f"공식의 +1은 bias 몫이다. ResNet의 conv는 바로 뒤 BatchNorm이 평행이동(β)을 맡아서 bias=False라 "
                 f"출력 채널당 1개씩, 총 {with_bias - real}개 차이가 난다.")

    md = "\n".join(L) + "\n"
    out = Path(__file__).resolve().parent / "cnn_shapes.md"
    out.write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
