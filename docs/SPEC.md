# RoomTwin: spec (shared contract for all components)

Entry for the OpenCV AI Competition 2026 (AWS). Deadline 26 Oct 11:59pm PT = Tue 27 Oct 5:59pm AEDT.
Internal target: everything working Mon 26 Oct.

**DONE WHEN:** one real room scanned with an iPhone 14 Pro (LiDAR) comes out within ±5 cm of
tape-measure readings on 3-4 walls; 3 furniture items added from links or typed dimensions;
at least one "won't fit through the door, here's why" result; a build order; a live link on AWS;
a video under 5 minutes.

## Judges and rubric (from Downloads/ai-page-saves/0003, 0004)
Gary Bradski (OpenCV founder, robotics/3D) and Phil Nelson (OpenCV content). Technical execution
30% = correctness and depth of the **OpenCV 5** implementation. Innovation 20, impact 20, UX 10,
docs 10, AWS/reproducibility/responsible-use 10. Agentic Vision award: visual results must change
the agent's next action (e.g. "rescan this corner"), with a trace and human control.
Required: technical report (incl. responsible use), pinned deps, architecture diagram, working web
endpoint, video <=5 min showing the team, evaluation evidence incl. failure cases.

## Components
1. `pipeline/` (Python, **opencv-python-headless==5.0.0.93**, numpy): scan to `room.json`.
2. `web/` (static three.js app, no build step, three@0.160.0 from a CDN): loads `room.json`,
   floor plan + 3D walk mode, furniture placement, fit/route checks, build order.
3. `server/` (FastAPI): upload scan, run the pipeline, furniture-link dimension extraction, the
   scan-QA agent, serving web/. Deployed on AWS.

## Inputs the pipeline must accept
- A point cloud or mesh: `.ply` / `.obj` (load with `cv2.loadPointCloud` where possible), in metres,
  y-up or z-up (detect gravity from the dominant horizontal plane, not assumed).
- Optionally RGB-D frames: a folder of RGB images + 16-bit or float depth + per-frame intrinsics and
  pose JSON (ARKit-style 4x4 camera-to-world). Fuse with `cv2.Volume` (TSDF / HashTSDF) when
  present; refine with `cv2.Odometry` if poses are missing or noisy.
- Optional ArUco/ChArUco photos for an independent scale check.

## room.json (units: metres; world frame: y up, floor at y = 0)
```json
{
  "version": 1,
  "source": {"type": "pointcloud|rgbd", "file": "...", "points": 123456},
  "floor": {"polygon": [[x, z], ...], "area_m2": 0.0},
  "ceiling_height": {"value": 2.42, "sigma": 0.01},
  "walls": [
    {"id": "w0", "a": [x, z], "b": [x, z], "length": 3.91, "sigma": 0.02,
     "height": 2.42, "normal": [nx, nz], "inlier_rms": 0.006}
  ],
  "openings": [
    {"id": "d0", "type": "door|window|opening", "wall": "w2",
     "offset": 0.35, "width": 0.82, "sill": 0.0, "height": 2.04, "sigma": 0.02}
  ],
  "obstacles": [ {"id": "o0", "label": "unknown", "min": [x, y, z], "max": [x, y, z]} ],
  "coverage": {"score": 0.93, "gaps": [ {"wall": "w1", "from": 1.2, "to": 1.9, "reason": "no points"} ]},
  "qa": [ {"severity": "warn", "message": "...", "action": "rescan|marker|ok"} ],
  "mesh": "room_mesh.glb (optional, decimated, for the 3D view)"
}
```
Rooms need not be rectangular: `floor.polygon` is the true footprint (L-shapes, angled walls).
Multiple rooms: a `house.json` lists rooms plus a 2D transform per room and the shared doorways.

## Furniture item (web + server)
```json
{"id": "f1", "name": "KIVIK sofa", "url": "...", "image": "...",
 "assembled": {"w": 2.28, "d": 0.95, "h": 0.83},
 "packages": [{"w": 1.2, "d": 0.6, "h": 0.3, "count": 3}],
 "flatpack": true, "needs_assembly_space": 0.6}
```

## Logistics rules (web)
- Fit through an opening: try all orientations of the box (6 axis permutations) plus tilting in the
  door plane; it passes if some orientation fits within width x height with clearance (default
  2 cm), and report the best orientation ("tilt on its side").
- Route: 2D configuration-space search (x, z, yaw) of the item's footprint from the entry door
  through rooms/hallways to its target pose, avoiding walls and already-placed items. Use the
  tightest-corner test for hallway turns. Report where it gets stuck.
- Flat-pack: check packages for the route and the assembled size for the room.
- Build order: place items farthest from the entry along the route first; an item must not block
  the route or the assembly space of any later item; big before small; explain each step.
