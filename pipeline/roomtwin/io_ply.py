"""Point cloud / mesh I/O. cv2.loadPointCloud first; a small numpy PLY reader as the fallback.

cv2.loadPointCloud (OpenCV 5) reads ASCII and binary PLY and OBJ, returning Nx1x3 float32 arrays and
RGB in [0, 1]. The fallback exists for PLY variants it rejects (double-precision coordinates, extra
per-vertex properties in odd orders, big-endian files) which some phone scanning apps export.
"""
from __future__ import annotations

import os
import re

import cv2
import numpy as np

_PLY_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1", "short": "i2", "int16": "i2",
    "ushort": "u2", "uint16": "u2", "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}


def _read_ply_numpy(path: str):
    """Minimal PLY reader: vertex x/y/z (+ red/green/blue), ascii or binary, any endianness.

    Faces are read only far enough to skip them; mesh vertices are used as the point cloud."""
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError("truncated PLY header")
            line = line.decode("ascii", "replace").strip()
            header.append(line)
            if line == "end_header":
                break
        fmt = next(h.split()[1] for h in header if h.startswith("format"))
        elements, cur = [], None
        for h in header:
            tok = h.split()
            if tok[:1] == ["element"]:
                cur = {"name": tok[1], "count": int(tok[2]), "props": []}
                elements.append(cur)
            elif tok[:1] == ["property"] and cur is not None:
                cur["props"].append(tok[1:])
        vert = next(e for e in elements if e["name"] == "vertex")
        if any(p[0] == "list" for p in vert["props"]):
            raise ValueError("list property in vertex element")
        names = [p[1] for p in vert["props"]]
        if fmt == "ascii":
            data = np.loadtxt(f, max_rows=vert["count"], dtype=np.float64, ndmin=2)
            cols = {n: data[:, i] for i, n in enumerate(names)}
        else:
            endian = "<" if fmt == "binary_little_endian" else ">"
            dt = np.dtype([(p[1], endian + _PLY_TYPES[p[0]]) for p in vert["props"]])
            # vertex element is normally first; skip any element before it (rare)
            for e in elements:
                if e is vert:
                    break
                raise ValueError("vertex element is not first")
            raw = np.frombuffer(f.read(dt.itemsize * vert["count"]), dtype=dt, count=vert["count"])
            cols = {n: raw[n].astype(np.float64) for n in names}
    pts = np.stack([cols["x"], cols["y"], cols["z"]], 1)
    rgb = None
    keys = [("red", "green", "blue"), ("r", "g", "b"), ("diffuse_red", "diffuse_green", "diffuse_blue")]
    for kr, kg, kb in keys:
        if kr in cols:
            rgb = np.stack([cols[kr], cols[kg], cols[kb]], 1)
            if rgb.max() > 1.0:
                rgb = rgb / 255.0
            break
    return pts, rgb


def _has_faces(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    with open(path, "rb") as f:
        head = f.read(1 << 20)
    if ext == ".ply":
        hdr = head.split(b"end_header")[0].decode("ascii", "replace")
        return any(l.startswith("element face") and int(l.split()[2]) > 0 for l in hdr.splitlines())
    return b"\nf " in head


def sample_mesh(V, F, rgb=None, spacing=0.007, max_points=3_000_000, seed=0):
    """Area-weighted uniform samples on triangles (phone meshes are coarse on flat walls, so their
    vertices alone would under-sample exactly the surfaces we measure). Vertices are kept too."""
    rng = np.random.default_rng(seed)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    n = int(min(max_points, area.sum() / spacing ** 2))
    if n <= 0:
        return V, rgb
    tri = rng.choice(len(F), n, p=area / area.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s1 = np.sqrt(r1)
    w = np.stack([1 - s1, s1 * (1 - r2), s1 * r2], 1)
    P = w[:, :1] * a[tri] + w[:, 1:2] * b[tri] + w[:, 2:] * c[tri]
    C = None
    if rgb is not None:
        C = w[:, :1] * rgb[F[tri, 0]] + w[:, 1:2] * rgb[F[tri, 1]] + w[:, 2:] * rgb[F[tri, 2]]
        C = np.concatenate([C, rgb])
    return np.concatenate([P, V]), C


def load_cloud(path: str):
    """Load .ply/.obj as (points Nx3 float64, rgb Nx3 in [0,1] or None, loader_name).

    Meshes are sampled uniformly over their surface (cv2.loadMesh); plain clouds use
    cv2.loadPointCloud, with a numpy PLY/OBJ reader as the fallback."""
    ext = os.path.splitext(path)[1].lower()
    if _has_faces(path):
        try:
            v, idx, _, col, _ = cv2.loadMesh(path)
            V = np.asarray(v, np.float64).reshape(-1, 3)
            F = np.concatenate([np.asarray(t).reshape(-1, 3) for t in idx]).astype(np.int64)
            rgb = np.asarray(col, np.float64).reshape(-1, 3) if col is not None and np.asarray(col).size == V.size else None
            if len(F):
                P, C = sample_mesh(V, F, rgb)
                return P, C, "cv2.loadMesh+surface-sampling"
        except Exception:
            pass
    try:
        v, _, rgb = cv2.loadPointCloud(path)
        pts = np.asarray(v, np.float64).reshape(-1, 3)
        col = None
        if rgb is not None and np.asarray(rgb).size == pts.size:
            col = np.asarray(rgb, np.float64).reshape(-1, 3)
        if len(pts) == 0:
            raise ValueError("empty")
        return pts, col, "cv2.loadPointCloud"
    except Exception:
        if ext != ".ply":
            # OBJ that cv2 rejects: read 'v x y z [r g b]' lines directly
            return (*_read_obj_vertices(path), "numpy-obj")
        pts, col = _read_ply_numpy(path)
        return pts, col, "numpy-ply"


def _read_obj_vertices(path):
    pts, cols = [], []
    with open(path, "r", errors="replace") as f:
        for line in f:
            if line.startswith("v "):
                vals = [float(x) for x in line.split()[1:]]
                pts.append(vals[:3])
                if len(vals) >= 6:
                    cols.append(vals[3:6])
    pts = np.asarray(pts, np.float64)
    col = np.asarray(cols, np.float64) if len(cols) == len(pts) and cols else None
    return pts, col


def write_ply(path: str, points: np.ndarray, rgb01: np.ndarray | None = None):
    """Binary little-endian PLY (float xyz + uchar rgb). Compact for the web preview."""
    n = len(points)
    props = "property float x\nproperty float y\nproperty float z\n"
    dt = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    if rgb01 is not None:
        props += "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        dt += [("r", "u1"), ("g", "u1"), ("b", "u1")]
    arr = np.zeros(n, dtype=dt)
    arr["x"], arr["y"], arr["z"] = points[:, 0], points[:, 1], points[:, 2]
    if rgb01 is not None:
        c = np.clip(np.round(rgb01 * 255), 0, 255).astype(np.uint8)
        arr["r"], arr["g"], arr["b"] = c[:, 0], c[:, 1], c[:, 2]
    header = f"ply\nformat binary_little_endian 1.0\nelement vertex {n}\n{props}end_header\n"
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(arr.tobytes())


def trailing_int(name: str):
    """Last integer in a file name (frame index), or None."""
    m = re.findall(r"(\d+)", os.path.splitext(os.path.basename(name))[0])
    return int(m[-1]) if m else None
