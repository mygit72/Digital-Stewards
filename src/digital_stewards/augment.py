"""Offline photometric degradation: makes clean game frames look like re-encoded broadcast / YouTube footage
(colour shifts, lower resolution, blur, noise, JPEG). Never moves pixels, so labels stay valid."""
import cv2
import numpy as np


def photometric_augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Broadcast/YouTube-style degradation. Never moves pixels, so labels stay valid."""
    out = img.astype(np.float32)
    # colour: brightness/contrast/saturation/gamma + slight hue rotation
    out = out * rng.uniform(0.75, 1.25) + rng.uniform(-20, 20)
    mean = out.mean(axis=(0, 1), keepdims=True)
    out = (out - mean) * rng.uniform(0.75, 1.25) + mean
    out = np.clip(out, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + rng.integers(-5, 6)) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.7, 1.3), 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    gamma = rng.uniform(0.8, 1.25)
    out = cv2.LUT(out, (np.linspace(0, 1, 256) ** gamma * 255).astype(np.uint8))
    h, w = out.shape[:2]
    # resolution loss (720p/480p re-encodes upscaled back)
    if rng.random() < 0.7:
        f = rng.uniform(0.35, 0.8)
        out = cv2.resize(cv2.resize(out, (max(8, int(w * f)), max(8, int(h * f))), interpolation=cv2.INTER_AREA),
                         (w, h), interpolation=cv2.INTER_LINEAR)
    # blur: defocus or motion (onboard vibration)
    r = rng.random()
    if r < 0.35:
        k = int(rng.choice([3, 5, 7]))
        out = cv2.GaussianBlur(out, (k, k), 0)
    elif r < 0.7:
        k = int(rng.choice([5, 7, 9, 11]))
        kern = np.zeros((k, k), np.float32)
        kern[k // 2, :] = 1.0 / k
        rot = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), float(rng.uniform(0, 180)), 1.0)
        out = cv2.filter2D(out, -1, cv2.warpAffine(kern, rot, (k, k)))
    # sensor noise
    if rng.random() < 0.5:
        out = np.clip(out.astype(np.float32) + rng.normal(0, rng.uniform(2, 7), out.shape), 0, 255).astype(np.uint8)
    # compression last, like the real pipeline
    q = int(rng.integers(15, 60))
    ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else out
