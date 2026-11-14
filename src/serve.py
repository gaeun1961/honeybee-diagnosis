"""유충 사진 진단 API (FastAPI, CPU)

실행: python src/serve.py [--weights models/best.pt] [--size 512] [--port 8000]
- 시작할 때 모델을 한 번만 로드
- POST /predict (multipart, 이미지 1장) -> {probs, verdict, top, cam}
    probs: {응애, 석고병, 부저병} sigmoid 확률 / verdict: 0.5 이상인 병 이름들, 없으면 ["정상"]
    top: 확률 최고 병 / cam: Grad-CAM(top 기준)을 원본에 겹친 JPEG data URL
- GET /health -> {"ok": true}
- 업로드 이미지는 디스크에 저장하지 않고 메모리에서만 처리
- CORS: http://localhost:*, http://127.0.0.1:*, file://(Origin: null)만 허용
"""
import argparse
import base64
import io
import threading
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from predict import grad_cam, load_model, overlay, preprocess
from train import LABELS

ROOT = Path(__file__).resolve().parent.parent
MAX_BYTES = 20 * 1024 * 1024
ORIGIN_REGEX = r"http://(localhost|127\.0\.0\.1)(:\d+)?|null"


def create_app(weights, size):
    model, ckpt_size = load_model(weights)
    if ckpt_size != size:
        print(f"주의: 학습 입력 가로 {ckpt_size}, 서버 입력 가로 {size}")
    tf = preprocess(size)
    lock = threading.Lock()  # ponytail: 요청을 한 번에 하나씩 처리 (Grad-CAM 훅이 모델 공유), 동시 요청이 많아지면 모델 복제/큐로

    app = FastAPI(title="꿀벌 유충 진단")
    app.add_middleware(CORSMiddleware, allow_origin_regex=ORIGIN_REGEX, allow_methods=["GET", "POST"], allow_headers=["*"])

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.post("/predict")
    def predict(file: UploadFile = File(...)):
        data = file.file.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise HTTPException(413, "이미지는 20MB 이하만 받습니다")
        try:
            img = Image.open(io.BytesIO(data)).convert("RGB")
        except (UnidentifiedImageError, OSError):
            raise HTTPException(400, "이미지 파일이 아닙니다")
        with lock:
            probs, top, cam = grad_cam(model, tf(img).unsqueeze(0))
        buf = io.BytesIO()
        overlay(img, cam).save(buf, "JPEG", quality=85)
        return {
            "probs": {c: round(p, 4) for c, p in zip(LABELS, probs)},
            "verdict": [c for c, p in zip(LABELS, probs) if p >= 0.5] or ["정상"],
            "top": LABELS[top],
            "cam": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(),
        }

    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=ROOT / "models/best.pt")
    ap.add_argument("--size", type=int, default=512, help="입력 가로 (세로 = size*9/16)")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    uvicorn.run(create_app(a.weights, a.size), host="127.0.0.1", port=a.port)
