# RoomTwin scan pipeline

Turns a phone LiDAR scan of a room into `room.json` (the contract in `../docs/SPEC.md`): floor
polygon, walls with lengths and uncertainties, doors and windows, furniture boxes, coverage and
scan-QA actions for the agent. Pure OpenCV 5 + numpy.

```bash
pip install -r requirements.txt            # opencv-python-headless==5.0.0.93, numpy==2.4.6
python -m roomtwin.process scan.ply -o out/                       # point cloud or mesh (.ply/.obj)
python -m roomtwin.process frames/ -o out/                        # RGB-D frame folder
python -m roomtwin.process scan.ply -o out/ --aruco photos/ --marker-size 0.15   # + scale check
python -m roomtwin.synth --out D:/roomtwin_data/synth --rgbd rect # synthetic rooms with ground truth
python -m roomtwin.evaluate                                       # measured vs truth table
python -m roomtwin.evaluate --sweep                               # 48-scan robustness sweep
python -m pytest -q tests                                         # 29 tests, ~50 s
```

Outputs in `out/`: `room.json`, `room_preview.ply` (decimated coloured cloud in the room frame),
`room_mesh.glb` (walls with openings cut out, floor, ceiling), `floorplan.png` (dimensioned top
view), `walls/<id>.png` (each wall's occupancy grid with openings and gaps). Exit code 2 still
writes a `room.json` with `error` and a `PIPELINE_FAILED` rescan action.

Frame: metres, y up, floor at y = 0, the longest-supported wall along +x, footprint starting at
(0, 0); `floor.polygon` is counter-clockwise in (x, z); wall normals point into the room;
`transform` is the 4x4 scan-to-room matrix. Openings: `offset` from the wall's `a` end, `sill` and
`height` of the clear opening (`height` = top - sill).

## Inputs

* `.ply` / `.obj` point clouds: `cv2.loadPointCloud`; a numpy PLY reader covers variants it rejects.
* Meshes (any face element): `cv2.loadMesh`, then area-weighted surface sampling (7 mm), because
  phone meshes are coarse exactly on the flat walls we measure.
* RGB-D folders, auto-detected (see `roomtwin/rgbd.py` docstring for keys):
  * **A. ARKit-style** `frame_N.jpg` + `depth_N.png|exr` + `frame_N.json` (`intrinsics`,
    `imageWidth/Height`, `cameraPoseARFrame` | `pose` | `transform`, `depthScale`); also inside
    `rgb/` and `depth/` subfolders, matched by the last integer in the name; ARKit camera axes.
  * **B. ScanNet-style** `color/`, `depth/`, `pose/*.txt`, `intrinsic/intrinsic_depth.txt`.
  * **C. TUM RGB-D** `rgb.txt`, `depth.txt`, `groundtruth.txt` (depth scale 5000).
  * optional `roomtwin_frames.json` overrides `convention`, `depth_scale`, `intrinsics`.
* `--aruco`: ARKit-style photos (RGB + depth + intrinsics) of a printed marker of known size.

Gravity is never assumed: y-up and z-up exports (and arbitrary orientations) are detected.

## Stages and the OpenCV 5 APIs used

| stage | what | OpenCV |
|---|---|---|
| load | clouds, meshes | `cv2.loadPointCloud`, `cv2.loadMesh` |
| RGB-D fusion | TSDF integration, surface by raycasting from the input poses | `cv2.VolumeSettings`, `cv2.Volume(VolumeType_HashTSDF)`, `Volume.integrate`, `Volume.raycast` |
| poses missing | frame-to-model odometry against the TSDF raycast, constant-velocity gating; with given poses it runs as a tracking check | `cv2.Odometry(OdometryType_RGB_DEPTH)`, `cv2.OdometrySettings`, `cv2.OdometryFrame`, `Odometry.prepareFrame/compute` |
| clean | 1 cm voxel centroids, grid radius-outlier removal, 2 cm working cloud | numpy |
| normals | local PCA on two staggered 10 cm grids, planarity test | numpy |
| gravity | normal clusters -> scored up candidates -> floor MSAC -> PCA normal | `cv2.PCACompute2`, `cv2.Rodrigues` |
| planes | MSAC plane model (`SAC_MODEL_PLANE` parametrisation) with local optimisation | `cv2.PCACompute2` |
| walls | one-point MSAC lines on oriented points, Huber refit, merge, outline ordering | `cv2.fitLine(DIST_HUBER)`, `cv2.findContours`, `cv2.morphologyEx`, `cv2.distanceTransform`, `cv2.line` |
| obstacles | top-view raster, components, robust boxes, oriented footprint | `cv2.connectedComponentsWithStats`, `cv2.morphologyEx`, `cv2.minAreaRect` |
| openings / gaps | per-wall 2 cm occupancy image, close/open, components, sub-cell edges | `cv2.morphologyEx`, `cv2.connectedComponentsWithStats`, `cv2.dilate`, `cv2.boxFilter` |
| coverage | floor / ceiling rasters | `cv2.fillPoly`, `cv2.erode`, `cv2.morphologyEx` |
| scale check | marker detection, camera-only distance, LiDAR plane | `cv2.aruco.ArucoDetector` (sub-pixel), `cv2.solvePnP(SOLVEPNP_IPPE_SQUARE)`, `cv2.fillConvexPoly`, `cv2.PCACompute2` |
| outputs | floor plan, wall elevations | `cv2.fillPoly`, `cv2.line`, `cv2.ellipse`, `cv2.rectangle`, `cv2.putText`, `cv2.imwrite` |
| synthetic data | marker photos | `cv2.aruco.generateImageMarker`, `cv2.getPerspectiveTransform`, `cv2.warpPerspective` |

### OpenCV 5.0.0 findings (verified, each reproduced by a script or test)

1. `cv::SACSegmentation` and `cv::voxelGridSampling` are in the C++ 3d module but are **not wrapped**
   in the Python bindings of opencv-python 5.0.0.93 (only the `SAC_MODEL_*` / `SAC_METHOD_*`
   constants are). We implement MSAC with the same plane model and polish with OpenCV least squares.
2. Dense `VolumeType_TSDF` (CPU) applies the camera's **world-z translation twice**: a camera at
   z = 0.3 m seeing a plane 1.7 m ahead puts it at z = 2.29 instead of 2.00. HashTSDF is correct.
   (`tests/test_rgbd_and_scale.py::test_opencv5_dense_tsdf_z_translation_behaviour` will start
   failing when this is fixed.)
3. `HashTSDF.fetchPointsNormals` returns every voxel in the truncation band (7 layers), not the zero
   crossing, so we extract the surface with `raycast` from the input viewpoints.
4. `HashTSDF.raycast` surfaces are offset by **half a voxel along (+x, +y, -z)** of the volume frame
   (least-squares fit over all pixels: (0.50, 0.54, -0.50) voxels at 1 cm and (0.49, 0.55, -0.50) at
   2 cm). Uncorrected, frame-to-model odometry drifts by that offset every frame (0.40 m after 200
   frames); corrected and re-projected, drift is 4.5 cm after 200 frames.
5. `OdometrySettings.setCameraMatrix` requires CV_32F; the OpenCL TSDF kernel fails with
   `CL_OUT_OF_RESOURCES` on this laptop, so the package disables OpenCL.

## Uncertainty model (per wall `sigma`, per opening `sigma`, ceiling `sigma`)

A wall line fitted to N_eff independent points with perpendicular residual RMS `s` (the reported
`inlier_rms`) and along-wall spread var_t has a perpendicular position error at along-wall
coordinate t of

    sigma_perp(t)^2 = s^2 / N_eff * (1 + (t - t_mean)^2 / var_t)

(the prediction variance of a least-squares line). N_eff is the number of occupied 2 cm x 2 cm
cells of the wall (`n_eff`): denser sampling than that is correlated and is not counted. A wall's
length runs between the intersections with its neighbours; moving neighbour j by sigma_perp moves
the corner along the wall by sigma_perp / sin(phi), so

    sigma_L^2 = (sigma_perp,prev(a) / sin phi_a)^2 + (sigma_perp,next(b) / sin phi_b)^2
                + (sigma_perp,own * (|cot phi_a| + |cot phi_b|))^2 + (k_scale * L)^2

k_scale (`--scale-sigma`, default 0.3 %) is the device scale uncertainty, an assumption the ArUco
check measures; on the synthetic data (exact scale) it dominates, so |error| / sigma is ~0.1.
An inferred wall (never observed, bridged between two parallel neighbours) gets 5 cm.
Ceiling: sigma_H^2 = sigma_floor^2 + sigma_ceiling^2 + (k_scale H)^2 with sigma = std / sqrt(cells).
Opening edges: sigma_edge^2 = spacing^2 / 12 + s^2 (sampling quantisation plus along-surface noise),
width sigma = sqrt(2) sigma_edge, +1 cell^2/3 per edge that could not be refined, +2 cm when the
hinge-side jamb is hidden by the open leaf (`hinge_side`).

Edges are refined below the 2 cm grid by counting: with locally uniform wall density rho (points per
metre of the profile, from the well-filled 2 cm bins next to the edge) the edge sits at
`t_start + count(points in [t_start, t_mid]) / rho`, unbiased under symmetric noise.

## Coverage and QA (consumed by the agent)

`coverage.score = 0.6 * walls + 0.25 * floor + 0.15 * ceiling` (observed fraction of 2 cm wall
cells excluding openings, 5 cm floor/ceiling cells; furniture footprints count as explained floor).
`coverage.gaps`: `{wall, from, to, y_from, y_to, reason: "no points" | "low density" | "occluded by oK", area_m2}`.
Every `qa` entry has `severity` (info|warn|error), a stable `code`, a concrete `message`, an
`action` (rescan|marker|ok) and a `target` (wall + range, corner + xz, or surface). Codes:
`GAP_BEHIND_DOOR`, `WALL_GAP`, `WALL_LOW_DENSITY`, `WALL_OCCLUDED`, `WALL_FEW_POINTS`,
`WALL_LOW_COVERAGE`, `WALL_INFERRED`, `CORNER_UNSEEN`, `NO_CEILING`, `CEILING_PARTIAL`,
`FLOOR_LOW_COVERAGE`, `GRAVITY_AMBIGUOUS`, `SCALE_MISMATCH`, `MARKER_NOT_FOUND`, `PIPELINE_FAILED`, `OK`.

## Robustness measures

Voxel-centroid downsampling (unbiased), grid radius-outlier removal, walls must reach the ceiling
band (rejects door leaves, beds, desks), candidates behind another wall within its extent are
outside the room (the hallway seen through a door), parallel candidates in front of a longer wall
are furniture, full-height gaps split a wall (alcove mouths) unless nothing is behind the gap
(tall cupboards), the outline is closed across door/window gaps so an unscanned floor patch cannot
leak, floor/ceiling heights use horizontal-surface points only, door gaps are separated from the
wall hidden behind the open leaf, and RGB-D odometry results are gated by a constant-velocity model.

## Evaluation (synthetic, exact ground truth)

`roomtwin.synth` builds rectangular, L-shaped, 30-degree-angled and alcove rooms with a door and a
window each, furniture, an open door leaf hiding the wall behind it, a hallway visible through the
door, partial ceiling coverage, a low-density patch, 7 mm normal noise, 0.2 % outliers, and a random
yaw in y-up or z-up convention. Full measured-vs-truth table: `python -m roomtwin.evaluate`
(also written to `D:/roomtwin_data/eval/synthetic_table.md`). Summary:

| test | result |
|---|---|
| 4 rooms (rect, L, angled, alcove), 0.28-0.36 M points | all walls found, no spurious walls; max wall-length error 0.41 cm (median per room 0.07-0.41 cm); ceiling within 0.1 cm; floor area within 0.003 m2; gravity exact |
| openings (2-3 per room) | all found and typed; max error over offset/width/sill/height 1.39 cm |
| furniture boxes | top-view IoU 0.92-0.98 |
| robustness sweep: 4 rooms x 6 seeds x noise 5 / 10 mm, random yaw, y-up/z-up | **48/48 pass** (walls <= 2 cm, opening dims <= 3 cm); worst wall 0.62 cm, worst opening 2.29 cm |
| 2.3 M point L-room (0.6 cm spacing) | 10.6-11.8 s end to end on a laptop (load 0.7 s, clean 4.4 s, walls 3.2 s); walls within 0.12 cm |
| RGB-D, 360 frames with ARKit poses -> HashTSDF | walls within 0.12 cm, ceiling 0.08 cm, openings within 2.2 cm |
| RGB-D, 360 frames, no poses -> cv2.Odometry frame-to-model | walls within 3.4 cm (odometry drift), openings within 7 cm, ceiling -4.7 cm |
| mesh input (coarse triangle mesh, z-up) | walls within 2 cm (test) |
| ArUco scale check, 4 photos | true scale 1.000 -> 0.998; depth scaled by 1.020 -> 1.018, reported as "off by +1.8%" |
| real data: TUM RGB-D fr1/room (Kinect, mocap poses, 454 frames) | qualitative: 4 main walls, a window and a doorway found, ceiling 3.06 m; the right wall is never seen and is inferred; cluttered corner gives 3 short spurious segments; 97 s (fusion dominates) |

Raw outputs: `D:/roomtwin_data/eval/` (synthetic table, sweep JSON) and `D:/roomtwin_data/out/<case>/`.

## Known weaknesses

* Synthetic noise is Gaussian along the surface normal; real LiDAR has range-dependent, correlated
  noise, multipath at corners and mesh smoothing, so real errors will be larger than the table.
  The 0.3 % scale term covers device scale; it is an assumption until the ArUco check is run.
* Ground truth for real data is still missing (no tape-measured TUM room); the iPhone bedroom with
  tape measurements is the real test.
* Walls must be vertical planes and reach the ceiling band; sloped ceilings, curved walls, and
  walls fully hidden by floor-to-ceiling furniture are not modelled (the latter become an inferred
  wall with a 5 cm sigma and a rescan action).
* A tall cupboard in front of a wall is treated as furniture only if part of the real wall above or
  beside it is visible; heavy clutter in corners can create short spurious wall segments (seen on TUM).
* The hinge-side jamb of an open door is hidden by the leaf; it is estimated from the leaf and
  carries +2 cm sigma. Closed doors are found from the recess (2-8 cm behind the wall plane); a
  door that is flush with the wall is invisible to geometry.
* Floor-to-ceiling openings are reported as `opening`, windows need a sill >= 0.3 m; curtains in
  front of a window make it a coverage gap rather than a window.
* Obstacles are axis-aligned boxes in the room frame (plus an oriented `footprint`); things seen
  only from one side (a wardrobe from the front) are as deep as what was seen. Label is "unknown".
* Without poses the RGB-D path drifts (frame-to-frame odometry has no loop closure); ARKit poses
  should always be exported. Gravity in that mode leans on the mean camera "up" hint.
* One room per run; `house.json` (multi-room) is not produced yet.
* RGB-D fusion of 640x480 Kinect frames is slow (~0.2 s/frame); iPhone depth (256x192) is ~5x faster.
