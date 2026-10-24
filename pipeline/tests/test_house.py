"""Synthetic two-room scan (bedroom + corridor behind its door): room segmentation, per-room walls
within 2 cm, and the shared doorway found on both sides."""
import numpy as np
import pytest

from roomtwin.evaluate import compare
from roomtwin.geometry import points_in_polygon
from roomtwin.io_ply import load_cloud
from roomtwin.process import process_cloud
from roomtwin.synth import make_house

WALL_TOL = 0.02
OPENING_TOL = 0.03


@pytest.fixture(scope="module")
def house_result(tmp_path_factory):
    out = tmp_path_factory.mktemp("house")
    ply, gt = make_house("bed_corridor", str(out), seed=7, up="z")
    P, C, _ = load_cloud(ply)
    room, extra = process_cloud(P, C, log=lambda *a: None)
    house = extra["house"]
    # match each ground-truth room to the result room containing its centre (mapped into the result frame)
    matched = {}
    for k, g in enumerate(gt["rooms"]):
        corners = np.asarray([w["a"] for w in g["walls"]], float)
        c = corners.mean(0)
        for r in house["rooms"]:
            rm = r["room"]
            M = np.asarray(rm["transform"]) @ np.asarray(g["gt_to_scan"])
            p = (M @ np.array([c[0], 0.0, c[1], 1.0]))[[0, 2]]
            if points_in_polygon(p[None], np.asarray(rm["floor"]["polygon"], float))[0]:
                matched[k] = r["id"]
    return room, house, gt, matched


def test_two_rooms_found(house_result):
    room, house, gt, matched = house_result
    assert house["type"] == "house" and len(house["rooms"]) == 2
    assert sorted(matched) == [0, 1] and len(set(matched.values())) == 2
    # the primary room (room.json) is the bedroom, the corridor is classified as a hallway
    assert matched[0] == room["id"] == house["rooms"][0]["id"]
    corridor = next(r["room"] for r in house["rooms"] if r["id"] == matched[1])
    assert corridor["segmentation"]["kind"] == "hallway"
    for r in house["rooms"]:
        assert r["transform"] == {"x": 0, "z": 0, "rot": 0}


@pytest.mark.parametrize("k", [0, 1])
def test_room_walls_within_2cm(house_result, k):
    _, house, gt, matched = house_result
    rm = next(r["room"] for r in house["rooms"] if r["id"] == matched[k])
    rows, extra = compare(rm, gt["rooms"][k])
    missing = [r["item"] for r in rows if r["measured"] is None and r["item"].startswith("wall")]
    assert not missing, f"not detected: {missing}"
    assert not extra, f"spurious walls: {extra}"
    for r in rows:
        if r["item"].startswith("wall"):
            assert abs(r["error"]) <= WALL_TOL, r
    assert abs(rm["ceiling_height"]["value"] - gt["rooms"][k]["ceiling_height"]) < 0.01


def test_shared_doorway(house_result):
    _, house, gt, matched = house_result
    assert len(house["doorways"]) == 1
    d = house["doorways"][0]
    assert {d["a"]["room"], d["b"]["room"]} == {matched[0], matched[1]}
    ends = []
    for side in ("a", "b"):
        rid = d[side]["room"]
        k = next(k for k, v in matched.items() if v == rid)
        rm = next(r["room"] for r in house["rooms"] if r["id"] == rid)
        o = next(o for o in rm["openings"] if o["id"] == d[side]["opening"])
        assert o["type"] == "door" and not o.get("inferred"), o      # seen from both sides
        rows, _ = compare(rm, gt["rooms"][k])
        door = [r for r in rows if r["item"].startswith(f"{o['id']} (door)")]
        assert door, rows
        for r in door:
            assert abs(r["error"]) <= OPENING_TOL, r
        w = next(w for w in rm["walls"] if w["id"] == o["wall"])
        a, b = np.asarray(w["a"]), np.asarray(w["b"])
        u = (b - a) / np.linalg.norm(b - a)
        ends.append(a + u * (o["offset"] + 0.5 * o["width"]))
    # the two sides line up: centres one wall thickness (12 cm) apart, not shifted along the wall
    assert np.linalg.norm(ends[0] - ends[1]) < 0.12 + 0.03
    assert house.get("entry") is None or house["entry"]["room"] in matched.values()
