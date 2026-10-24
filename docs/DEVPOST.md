# Devpost submission fields (copy-paste)

## Project name

RoomTwin

## Tagline (under 200 characters)

Scan a room with a LiDAR phone; OpenCV 5 on AWS measures every wall with an error bar, an agent asks for a rescan where needed, and you check furniture fits through the door.

## Categories / prizes

* Overall
* Agentic Vision Award

COOL (Cloud-Optimized OpenCV Library) is not used in this entry, so it is not entered for the Best Use of COOL Award.

## Links

* Live app: https://54-226-102-179.sslip.io/
* Code: (repository link)
* Video: (link, under 5 minutes)
* Technical report: docs/REPORT.md in the repository

## Inspiration

Every move ends with the same questions: will the sofa fit against that wall, will it go through the bedroom door, can it get round the hallway corner, and what goes in first. People answer them with a tape measure and guesswork, and getting it wrong means a return trip or a sofa stuck on the landing. LiDAR phones can capture a room in a minute, but the exports are just point clouds or meshes. Nothing tells you how long a wall is, how wide the door really is, how sure that number is, or which corner the phone never saw. We wanted a measured model with honest error bars, plus something that tells you what to rescan before you rely on it.

## What it does

1. You scan a room (or a room and its hallway) with a LiDAR phone app and upload the .ply / .obj / .zip.
2. An OpenCV 5 pipeline on an AWS Graviton instance finds gravity, separates rooms at doorways, fits the walls, finds doors and windows (including a closed door flush with the wall and a window behind a blind), boxes the furniture, and scores coverage. Every wall and opening comes with a 1-sigma uncertainty, and every gap comes with a concrete QA action ("rescan the corner behind the door on w1, 0.34 to 1.06 m along the wall").
3. A scan coach agent inspects that evidence: the summary, the per-wall occupancy images and the floor plan the pipeline drew. It then accepts the scan, requests one specific physical rescan, or asks you a question. You decide what to do; a rescan becomes attempt 2.
4. In the web app (plan, 3D and walk views) you add furniture from a product link or typed sizes. RoomTwin checks door fit over all 6 box orientations plus tilting in the door plane, searches a route (x, z, yaw) from the entry door to the item's spot, explains where and why it gets stuck, and builds an order that never blocks a later item's route or assembly space.

## How we built it

* **Pipeline (Python, opencv-python-headless 5.0.0.93, numpy):** `cv2.loadPointCloud` / `cv2.loadMesh` for input; `cv2.Volume` (HashTSDF) and `cv2.Odometry` for RGB-D frames; `cv2.PCACompute2` and `cv2.Rodrigues` for gravity and plane fits; room segmentation in a top-view image with `connectedComponentsWithStats`, `morphologyEx`, `findContours`, `distanceTransform`, `watershed` and `distanceTransformWithLabels`; walls with MSAC and `cv2.fitLine(DIST_HUBER)`; doors, windows and gaps from per-wall occupancy images with morphology and components, with edges refined below the 2 cm grid; furniture with `minAreaRect`; an optional ArUco scale check with `ArucoDetector` and `solvePnP(IPPE_SQUARE)`. Wall uncertainty comes from the least-squares line prediction variance, propagated through the corner angles.
* **Agent:** Claude with strict tool schemas. Three read-only perception tools (summary, view wall image, view floor plan) and three action tools (accept, request rescan, ask). Max 8 turns, a fallback when no action is chosen, and every call written to `coach_trace.json`.
* **Server:** FastAPI on an EC2 t4g.medium (Graviton, arm64), uvicorn under systemd, Caddy for automatic HTTPS on an sslip.io name, SSH open to one IP only, the API key kept in a root-only env file outside the repo, and a daily cap on model calls.
* **Web:** static ES modules and three.js with no build step; door fit, configuration-space A* and build order run in a Web Worker.
* **Evaluation:** a synthetic room generator with exact ground truth (rectangular, L-shaped, angled and alcove rooms with an open door hiding a wall, a hallway seen through the door, noise, outliers, random yaw, y-up or z-up), a 48-scan sweep, a two-room test, and a real iPhone 14 Pro scan.

## Challenges we ran into

* **OpenCV 5.0.0 binding gaps and bugs.** `SACSegmentation` and `voxelGridSampling` are not exposed in Python. Dense `VolumeType_TSDF` applies the camera's world-z translation twice, so we used HashTSDF. `HashTSDF.fetchPointsNormals` returns the whole truncation band rather than the surface. `HashTSDF.raycast` is offset by half a voxel, which made odometry drift 0.40 m over 200 frames; after correcting it, drift was 4.5 cm. Each one is reproduced by a script or test and written up.
* **Real scans are not one room.** The phone also captures the hallway seen through the door. We had to separate rooms before fitting walls (door lintels close the doorway in a top view), level each room on its own floor because the scan drifts about 1 degree between rooms, and pair doors seen from one side with an inferred partner.
* **Doors are hard.** An open leaf hides the wall behind it, and a closed door flush with the wall shows up only as thin gaps around its frame. Both needed dedicated handling.
* **Honest uncertainty.** On real data, drift between scanning passes doubles surfaces, so the textbook sigma is optimistic. We report inlier RMS alongside it and say so.

## Accomplishments that we're proud of

* Synthetic rooms with exact ground truth: every wall found, wall-length errors at most 0.4 cm, ceilings within 0.1 cm, openings within 1.4 cm, top-view furniture IoU 0.93-0.98.
* 48 of 48 sweep scans pass (4 room shapes x 6 seeds x 5 / 10 mm noise, random yaw, y-up / z-up): worst wall 0.62 cm, worst opening 2.29 cm.
* Two-room house test: both rooms found every time, the door detected from both sides and paired within 0.95 cm.
* A real iPhone 14 Pro scan (2.0 M points) processed in 19 s on a laptop. The bedroom comes out as 6 walls, ceiling 2.399 m, a closed flush door of 0.77 x 2.07 m and a 1.69 m window behind a blind, with the hallway separated and the doorway paired.
* A recorded agent trace where the pipeline's gap output chose which wall image the agent inspected next, and the agent turned two separate QA findings and an unknown door hinge side into one instruction: close the door and sweep the strip beside it.

## What we learned

* OpenCV 5's 3D module (point cloud I/O, TSDF volumes, odometry) is usable from Python today, but test it against ground truth: two of the bugs we found shift geometry silently instead of raising an error.
* Classic 2D image operations (morphology, connected components, distance transforms, watershed) are the right tools for 3D rooms once you project to the right image: a top view for rooms, one occupancy image per wall for openings and gaps.
* An agent is most useful when its perception is someone else's measured output. The coach adds judgement and a plain instruction; it never measures.
* Synthetic ground truth catches geometry bugs. It does not predict real accuracy: real LiDAR noise is correlated and scans drift.

## What's next

* Fill in the tape-measure comparison on the real bedroom and publish it, error bars included.
* Measure the coach's success rate on labelled good and bad scans, with and without injected defects.
* Register rooms to each other (shared-wall constraints) to remove the inter-room drift.
* Use the ARKit RGB-D export with poses as the default input, and run the ArUco scale check on real scans.
* A 3D route search that plans tilting and lifting along the route, not only at the door.
* Accounts, scan expiry and per-client rate limits on the server.

## Built with

opencv, opencv-5, python, numpy, fastapi, uvicorn, aws, amazon-ec2, aws-graviton, caddy, systemd, ubuntu, claude, anthropic-api, three.js, javascript, web-workers, lidar, iphone
