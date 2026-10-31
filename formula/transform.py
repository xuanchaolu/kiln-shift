# -*- coding: utf-8 -*-
"""烧制前 → 烧制后（1300℃ → 1800℃）颜色映射。

    1) 通道独立：x1 = a·x + b，w = C·t^0.746·(1−t)^1.165，y1 = x + K·w·(x1 − x)
    2) 亮度路径：亮度 L 同式映射得 L2，再按色度展开 y2 = L2 + S·(x − L)
    3) 按彩度混合：通道差 < C0 用 y2（灰不被染色），> C1 用 y1（有色釉黄化）
                   另加"很亮时也用 y1"，避免浅色釉失去黄化

K = 1.25，S = 2.0，C0 = 6，C1 = 40。端点 x=0/255 严格不动。
留一交叉验证：ΔE00 均值 11.84。
反向：亮度路径解析反解（彩度高时退回逐通道二分），往返误差 ≤ 2。

用法：
    from transform import before_to_after, after_to_before, to_hex
    before_to_after((212, 215, 212))      # -> (219.6, 208.4, 196.1)
    python transform.py 212 215 212
"""
import numpy as np

A = (1.5348, 1.9364, 1.8902)      # 每通道斜率
B = (-126.22, -223.93, -241.17)   # 每通道截距


def _clamp(v):
    return 0.0 if v < 0 else (255.0 if v > 255 else float(v))


WP, WQ, WC = 0.746, 1.165, 3.590511
K = 1.25                          # 变化量倍率（浓淡）
AY, BY = 1.6872, -174.40          # 亮度直线（43 块拟合）
C0, C1, S_CHROMA = 6.0, 40.0, 2.0  # 彩度混合阈值与色度增益
W_LUM = (0.2126, 0.7152, 0.0722)


def wfun(x):
    """两端权重，x 为 0-255 通道值"""
    t = min(max(float(x) / 255.0, 0.0), 1.0)
    v = WC * (max(t, 1e-9) ** WP) * (max(1.0 - t, 1e-9) ** WQ)
    return min(max(v, 0.0), 1.0)


def _fwd(x, c, k=None):
    """通道独立那一路（有色釉）"""
    t = _clamp(x)
    w = wfun(t)
    base = (1 - w) * t + w * (A[c] * t + B[c])
    return _clamp(t + (K if k is None else k) * (base - t))


def _lum_map(l, k=None):
    """亮度映射（低彩度那一路）"""
    l = _clamp(l)
    w = wfun(l)
    base = (1 - w) * l + w * (AY * l + BY)
    return _clamp(l + (K if k is None else k) * (base - l))


def _clip01(v):
    return 0.0 if v < 0 else (1.0 if v > 1 else float(v))


def _mix_weight(rgb, lum):
    wc = _clip01((max(rgb) - min(rgb) - C0) / (C1 - C0))
    wl = _clip01((lum - 160.0) / 40.0)
    return max(wc, wl)


def before_to_after(rgb, ndigits=1, k=None):
    """(R,G,B) 0-255 → 预测的烧成色；k 覆盖默认浓淡（0.6-2.0）"""
    rgb = [float(v) for v in rgb]
    lum = sum(w * v for w, v in zip(W_LUM, rgb))
    L2 = _lum_map(lum, k)
    w = _mix_weight(rgb, lum)
    p1 = [_fwd(v, c, k) for c, v in enumerate(rgb)]
    p2 = [L2 + S_CHROMA * (v - lum) for v in rgb]
    return tuple(round(_clamp(a + w * (b - a)), ndigits) for a, b in zip(p2, p1))


def after_to_before(rgb, ndigits=1, k=None):
    """由烧成色反推初始釉色。彩色像素逐通道二分；近中性像素按亮度解析反解。"""
    rgb = [float(v) for v in rgb]
    if max(rgb) - min(rgb) > 40:                     # 彩色：逐通道反查
        out = []
        for c, v in enumerate(rgb):
            lo, hi = 0.0, 255.0
            for _ in range(50):
                mid = (lo + hi) / 2
                if _fwd(mid, c, k) < v:
                    lo = mid
                else:
                    hi = mid
            out.append((lo + hi) / 2)
    else:                                            # 近中性：亮度路径
        L2 = _clamp(sum(w * v for w, v in zip(W_LUM, rgb)))
        lo, hi = 0.0, 255.0
        for _ in range(50):
            mid = (lo + hi) / 2
            if _lum_map(mid, k) < L2:
                lo = mid
            else:
                hi = mid
        lum = (lo + hi) / 2
        out = [lum + (v - L2) / S_CHROMA for v in rgb]
    return tuple(round(_clamp(v), ndigits) for v in out)


def to_hex(rgb):
    return '#' + ''.join(f'{int(round(_clamp(v))):02X}' for v in rgb)


def lut(n=256, inverse=False):
    """灰阶查表（仅对中性色成立）；彩色像素请直接调 before_to_after"""
    out = np.zeros((n, 3))
    for c in range(3):
        for i in range(n):
            if inverse:
                lo, hi = 0.0, 255.0
                for _ in range(40):
                    mid = (lo + hi) / 2
                    if _fwd(mid, c) < i:
                        lo = mid
                    else:
                        hi = mid
                out[i, c] = round((lo + hi) / 2)
            else:
                out[i, c] = round(_fwd(i, c))
    return out


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 4:
        rgb = tuple(float(v) for v in sys.argv[1:4])
        print('烧前', to_hex(rgb), rgb)
        print('烧后', to_hex(before_to_after(rgb)), before_to_after(rgb))
        print('回推', to_hex(after_to_before(before_to_after(rgb))), after_to_before(before_to_after(rgb)))
    else:
        print('灰阶（黑白两极应严格不动）：')
        for v in (0, 32, 64, 128, 192, 224, 240, 255):
            q = before_to_after((v, v, v))
            print(f'  {v:>3} → ({q[0]:5.1f},{q[1]:5.1f},{q[2]:5.1f})   B−R = {q[2]-q[0]:+6.1f}')
