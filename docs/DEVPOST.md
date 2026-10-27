# Devpost submission fields (copy-paste)

## Project name

RoomTwin

## Tagline (under 200 characters)

Will it fit? Scan your room with an iPhone and RoomTwin tells you before you buy: through the door, round the corner, in the right order. OpenCV 5 on AWS, measured with error bars.

## Categories / prizes

* Overall
* Agentic Vision Award

COOL (Cloud-Optimized OpenCV Library) is not used in this entry, so it is not entered for the Best Use of COOL Award.

## Links

* Live app: https://54-226-102-179.sslip.io/
* Code: (repository link)
* Video: (link, under 5 minutes)
* Technical report: docs/REPORT.md in the repository

## The problem, and who pays for it

Furniture that doesn't fit is a common, expensive failure:

* **How common:** in a 2025 survey of 1,500 UK furniture buyers, **49% had ordered furniture that didn't fit through the door or couldn't get up the stairs**, and 60% had bought furniture online that turned out to be the wrong size ([Fixtuur, "Rise of Returniture", 2025](https://furniturenews.net/resources/more-shoppers-now-try-buy-and-return-furniture-reports-fixtuur)). That is a self-reported survey, not a measured return rate.
* **What it costs:** restocking fees typically run **10-20% of the price** ([Consumer Reports, via Money](https://money.com/restocking-fee-furniture-problem-solved/)). On a $1,000 sofa that is $100-200 for the customer before return freight. The retailer still pays for two delivery trips and a damaged-returns risk.

The buyers are the people who pay for that failed delivery:

* **Furniture retailers**, with a "will it fit your home?" check next to the Buy button. RoomTwin already reads IKEA product pages, including the flat-pack box sizes.
* **Removalists and delivery companies**, who can quote the right crew and plan which items to carry in boxed or tilted, before the truck leaves.
* The customer themselves, before spending the money.

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

* Checked against a tape measure in a real bedroom: the 4.81 m wall came out as 4.809 m, the 60 cm step as 60.3 cm, and the ceiling as 2.399 m (tape about 2.38-2.40 m). We also report the misses: the door was 9 cm off, and the window 7.6 cm off because it was measured on the blind (see the report).
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

* Re-measure the door jamb to jamb with the door open, and scan with the door open, to settle the 9 cm door discrepancy.
* Measure the coach's success rate on labelled good and bad scans, with and without injected defects.
* Register rooms to each other (shared-wall constraints) to remove the inter-room drift.
* Use the ARKit RGB-D export with poses as the default input, and run the ArUco scale check on real scans.
* A 3D route search that plans tilting and lifting along the route, not only at the door.
* Accounts, scan expiry and per-client rate limits on the server.

## Built with

opencv, opencv-5, python, numpy, fastapi, uvicorn, aws, amazon-ec2, aws-graviton, caddy, systemd, ubuntu, claude, anthropic-api, three.js, javascript, web-workers, lidar, iphone
