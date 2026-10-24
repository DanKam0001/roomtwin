"""RGB-D fusion (cv2.Volume HashTSDF + cv2.Odometry) and the ArUco scale check."""
import json
import os

import cv2
import numpy as np
import pytest

from roomtwin.evaluate import compare
from roomtwin.process import process_cloud
from roomtwin.rgbd import RAYCAST_OFFSET_VOX, fuse_folder
from roomtwin.scalecheck import check_folder
from roomtwin.synth import ROOMS, make_aruco_frames, make_rgbd


@pytest.fixture(scope="module")
def rgbd_room(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("rgbd") / "rect")
    make_rgbd(ROOMS["rect"], d, seed=5, n_frames=200)
    P, C, src = fuse_folder(d, log=lambda *a: None)
    room, _ = process_cloud(P, C, src, log=lambda *a: None)
    gt = json.load(open(os.path.join(d, "gt.json")))
    return room, gt, compare(room, gt)


def test_rgbd_given_poses_walls(rgbd_room):
    room, gt, (rows, extra) = rgbd_room
    assert room["source"]["type"] == "rgbd"
    assert not extra
    for r in rows:
        if r["item"].startswith("wall"):
            assert r["measured"] is not None and abs(r["error"]) <= 0.02, r


def test_rgbd_given_poses_openings(rgbd_room):
    _, _, (rows, _) = rgbd_room
    for r in rows:
        if "(door)" in r["item"] or "(window)" in r["item"]:
            # the TSDF (4 cm truncation) blends the open door leaf into the wall near its hinge, so
            # the hinge-side jamb is the least certain edge; everything else is within 3 cm
            assert abs(r["error"]) <= 0.05, r


def _scale_case(tmp_path, s):
    d = make_aruco_frames(str(tmp_path / f"aruco_{s}"), depth_scale=s)
    return check_folder(d, 0.15)


def test_scale_check_ok(tmp_path):
    r = _scale_case(tmp_path, 1.0)
    assert r["status"] == "ok" and abs(r["scale"] - 1.0) < 0.005
    assert "matches" in r["verdict"]


def test_scale_check_detects_error(tmp_path):
    r = _scale_case(tmp_path, 1.02)
    assert abs(r["scale"] - 1.02) < 0.005
    assert "off by" in r["verdict"]


def _plane_volume(voxel, cam_t, depth_m):
    W, H = 256, 192
    K = np.array([[200.0, 0, 127.5], [0, 200.0, 95.5], [0, 0, 1]])
    kind = cv2.VolumeType_TSDF
    s = cv2.VolumeSettings(kind)
    for f in (s.setIntegrateWidth, s.setRaycastWidth):
        f(W)
    for f in (s.setIntegrateHeight, s.setRaycastHeight):
        f(H)
    s.setCameraIntegrateIntrinsics(K)
    s.setCameraRaycastIntrinsics(K)
    s.setVolumeResolution(np.array([160, 160, 160], np.int32))
    s.setVolumePose(np.eye(4, dtype=np.float32))
    s.setVoxelSize(voxel)
    s.setTsdfTruncateDistance(4 * voxel)
    s.setMaxDepth(5.0)
    s.setDepthFactor(1.0)
    v = cv2.Volume(kind, s)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = cam_t
    v.integrate(np.full((H, W), depth_m, np.float32), pose)
    pts, _ = v.fetchPointsNormals()
    return np.asarray(pts).reshape(-1, 4)[:, :3]


def test_opencv5_dense_tsdf_z_translation_behaviour():
    """Documents the OpenCV 5.0.0 dense-TSDF behaviour that made us use HashTSDF: a camera at z=0.3
    looking at a plane 1.7 m ahead (world z=2.0) yields a surface near z=2.3. If this test starts
    failing, OpenCV fixed it and the dense TSDF becomes usable again."""
    pts = _plane_volume(0.02, [1.6, 1.6, 0.3], 1.7)
    assert abs(np.median(pts[:, 2]) - 2.29) < 0.03


def test_raycast_offset_constant_is_documented():
    assert np.allclose(np.abs(RAYCAST_OFFSET_VOX), 0.5)
