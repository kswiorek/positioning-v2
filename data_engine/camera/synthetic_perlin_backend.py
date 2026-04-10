"""Synthetic Perlin-noise depth backend for capture debugging."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import numpy as np

from .base import DepthCameraBackend, DepthFrame


class _PerlinNoise2D:
    """Small deterministic 2D Perlin noise helper."""

    def __init__(self, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        p = rng.permutation(256).astype(np.int32)
        self.p = np.concatenate([p, p])

    @staticmethod
    def _fade(t: np.ndarray) -> np.ndarray:
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

    @staticmethod
    def _lerp(t: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a + t * (b - a)

    @staticmethod
    def _grad(h: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        h = h & 3
        u = np.where(h < 2, x, y)
        v = np.where(h < 2, y, x)
        return np.where((h & 1) == 0, u, -u) + np.where((h & 2) == 0, v, -v)

    def noise_batch(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        xi = np.floor(x).astype(np.int32) & 255
        yi = np.floor(y).astype(np.int32) & 255

        xf = x - np.floor(x)
        yf = y - np.floor(y)

        u = self._fade(xf)
        v = self._fade(yf)

        a = self.p[xi] + yi
        aa = self.p[a]
        ab = self.p[a + 1]
        b = self.p[xi + 1] + yi
        ba = self.p[b]
        bb = self.p[b + 1]

        return self._lerp(
            v,
            self._lerp(u, self._grad(self.p[aa], xf, yf), self._grad(self.p[ba], xf - 1.0, yf)),
            self._lerp(u, self._grad(self.p[ab], xf, yf - 1.0), self._grad(self.p[bb], xf - 1.0, yf - 1.0)),
        )

    def octave_noise_batch(
        self,
        x: np.ndarray,
        y: np.ndarray,
        octaves: int = 4,
        persistence: float = 0.5,
    ) -> np.ndarray:
        total = np.zeros_like(x, dtype=np.float32)
        freq = 1.0
        amp = 1.0
        amp_sum = 0.0

        for _ in range(max(int(octaves), 1)):
            total += self.noise_batch(x * freq, y * freq).astype(np.float32) * amp
            amp_sum += amp
            amp *= persistence
            freq *= 2.0

        return total / max(amp_sum, 1e-8)


class SyntheticPerlinDepthBackend(DepthCameraBackend):
    """Procedural depth stream that mimics rough depth backgrounds."""

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        seed: int = 1234,
        base_depth_m: float = 2.0,
        amplitude_m: float = 0.15,
        noise_scale: float = 2.5,
        octaves: int = 4,
        temporal_speed: float = 0.35,
        dropout_prob: float = 0.0,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.fps = max(int(fps), 1)
        self.seed = int(seed)
        self.base_depth_m = float(base_depth_m)
        self.amplitude_m = float(amplitude_m)
        self.noise_scale = float(noise_scale)
        self.octaves = int(octaves)
        self.temporal_speed = float(temporal_speed)
        self.dropout_prob = float(dropout_prob)

        self._rng = np.random.default_rng(self.seed)
        self._noise = _PerlinNoise2D(self.seed)
        self._x = None
        self._y = None
        self._started = False
        self._t0 = 0.0
        self._last_emit = 0.0

    def start(self) -> None:
        xv = np.linspace(-1.0, 1.0, self.width, dtype=np.float32)
        yv = np.linspace(-1.0, 1.0, self.height, dtype=np.float32)
        xx, yy = np.meshgrid(xv, yv)
        self._x = xx * self.noise_scale
        self._y = yy * self.noise_scale

        self._t0 = time.time()
        self._last_emit = self._t0
        self._started = True

    def read(self) -> Optional[DepthFrame]:
        if not self._started:
            raise RuntimeError("SyntheticPerlinDepthBackend is not started.")

        target_dt = 1.0 / float(self.fps)
        now = time.time()
        elapsed_since_emit = now - self._last_emit
        if elapsed_since_emit < target_dt:
            time.sleep(target_dt - elapsed_since_emit)
            now = time.time()

        t = (now - self._t0) * self.temporal_speed
        depth_noise = self._noise.octave_noise_batch(self._x + t, self._y + 0.5 * t, octaves=self.octaves)
        depth_m = self.base_depth_m + self.amplitude_m * depth_noise
        depth_m = np.maximum(depth_m, 0.05).astype(np.float32)

        if self.dropout_prob > 0.0:
            mask = self._rng.random(size=depth_m.shape) < self.dropout_prob
            depth_m = depth_m.copy()
            depth_m[mask] = 0.0

        self._last_emit = now
        return DepthFrame(depth_m=depth_m, color_bgr=None, timestamp_s=now)

    def stop(self) -> None:
        self._started = False

    def metadata(self) -> Dict[str, Any]:
        return {
            "backend": "synthetic_perlin",
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "seed": self.seed,
            "base_depth_m": self.base_depth_m,
            "amplitude_m": self.amplitude_m,
            "noise_scale": self.noise_scale,
            "octaves": self.octaves,
            "temporal_speed": self.temporal_speed,
            "dropout_prob": self.dropout_prob,
        }
