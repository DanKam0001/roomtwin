"""Synthetic rooms with exact ground truth: walls within 2 cm, openings within 3 cm."""
import cv2
import numpy as np
import pytest

from roomtwin.evaluate import compare
from roomtwin.export import room_mesh
from roomtwin.io_ply import load_cloud
from roomtwin.process import process_cloud
from roomtwin.synth import DATASET, ROOMS, ground_truth, make_room, random_pose

WALL_TOL = 0.02
OPENING_TOL = 0.03


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    out = tmp_path_factory.mktemp("synth")
    res = {}
    for name, seed, up in DATASET:
        ply, gt = make_room(ROOMS[name], str(out), seed, up)
        P, C, _ = load_cloud(ply)
        room, _ = process_cloud(P, C, log=lambda *a: None)
        res[name] = (room, gt, compare(room, gt))
    return res


@pytest.mark.parametrize("name", [d[0] for d in DATASET])
def test_topology(results, name):
    room, gt, (rows, extra) = results[name]
    missing = [r["item"] for r in rows if r["measured"] is None]
    assert not missing, f"not detected: {missing}"
    assert not extra, f"spurious walls: {extra}"
    assert len(room["walls"]) == len(gt["walls"])


@pytest.mark.parametrize("name", [d[0] for d in DATASET])
def test_gravity_and_height(results, name):
    room, gt, (rows, _) = results[name]
    grav = next(r for r in rows if r["item"] == "gravity")
    assert grav["error"] < 0.5                      # degrees
    assert abs(room["ceiling_height"]["value"] - gt["ceiling_height"]) < 0.01


@pytest.mark.parametrize("name", [d[0] for d in DATASET])
def test_walls_within_2cm(results, name):
    _, _, (rows, _) = results[name]
    for r in rows:
        if r["item"].startswith("wall"):
            assert abs(r["error"]) <= WALL_TOL, r


@pytest.mark.parametrize("name", [d[0] for d in DATASET])
def test_openings_within_3cm(results, name):
    room, gt, (rows, _) = results[name]
    assert len(room["openings"]) == len(gt["openings"])
    for r in rows:
        if "(door)" in r["item"] or "(window)" in r["item"]:
            assert abs(r["error"]) <= OPENING_TOL, r


@pytest.mark.parametrize("name", [d[0] for d in DATASET])
def test_obstacles_found(results, name):
    _, _, (rows, _) = results[name]
    for r in rows:
        if r["item"].startswith("obstacle"):
            assert r["measured"] > 0.8, r            # top-view IoU with the furniture box


def test_qa_flags_gap_behind_door(results):
    room = results["rect"][0]
    codes = [q["code"] for q in room["qa"]]
    assert "GAP_BEHIND_DOOR" in codes or "WALL_OCCLUDED" in codes
    for q in room["qa"]:
        assert {"severity", "code", "message", "action", "target"} <= set(q)
        assert q["action"] in ("rescan", "marker", "ok")


def test_room_json_contract(results):
    room = results["lshape"][0]
    for key in ("version", "source", "floor", "ceiling_height", "walls", "openings", "obstacles", "coverage", "qa"):
        assert key in room
    poly = np.asarray(room["floor"]["polygon"])
    x, z = poly[:, 0], poly[:, 1]
    assert 0.5 * (np.dot(x, np.roll(z, -1)) - np.dot(z, np.roll(x, -1))) > 0   # counter-clockwise
    for w in room["walls"]:
        assert {"id", "a", "b", "length", "sigma", "height", "normal", "inlier_rms"} <= set(w)
    for o in room["openings"]:
        assert o["type"] in ("door", "window", "opening")


def test_mesh_input(tmp_path):
    """A triangle mesh (no noise) goes through cv2.loadMesh + surface sampling."""
    spec = ROOMS["angled"]
    gt = ground_truth(spec)
    fake = {"ceiling_height": {"value": spec.height},
            "floor": {"polygon": [list(c) for c in spec.corners]},
            "walls": [dict(id=w["id"], a=w["a"], b=w["b"], length=w["length"]) for w in gt["walls"]],
            "openings": [dict(o, wall=f"w{o['wall']}") for o in spec.openings]}
    prims = room_mesh(fake)
    T = random_pose(np.random.default_rng(3), "z")
    V, F = [], []
    for v, f, _ in prims.values():
        F.append(np.asarray(f) + sum(len(x) for x in V))
        V.append(np.asarray(v))
    V = np.concatenate(V) @ T[:3, :3].T + T[:3, 3]
    F = np.concatenate(F).astype(np.int32)
    path = str(tmp_path / "mesh.ply")
    cv2.saveMesh(path, V.astype(np.float32).reshape(-1, 1, 3), [f.reshape(1, 3) for f in F])
    P, C, loader = load_cloud(path)
    assert "loadMesh" in loader
    room, _ = process_cloud(P, C, log=lambda *a: None)
    gt.update(gt_to_scan=T.tolist())
    rows, extra = compare(room, gt)
    for r in rows:
        if r["item"].startswith("wall"):
            assert r["measured"] is not None and abs(r["error"]) <= WALL_TOL, r
