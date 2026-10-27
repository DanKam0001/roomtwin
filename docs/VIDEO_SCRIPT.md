# RoomTwin demo video: script (target 3:45, hard limit 5:00)

The rules require the video to show the team, the application working, its architecture and its
main results. **[YOU]** marks the clips Danish films on his phone; everything else is screen capture
or existing footage.

| # | time | on screen | voice (Danish) |
|---|---|---|---|
| 1 | 0:00-0:20 | **[YOU]** facing the camera, in the bedroom | "Half of furniture buyers in a recent UK survey have ordered something that didn't fit through the door or up the stairs. That's a return, a restocking fee and a wasted delivery trip. RoomTwin answers it before you buy: will it fit through the door, round the corner, in the right order?" |
| 2 | 0:20-0:40 | scan screen recording (`raw/scan_screenrec.mp4`), sped up 6x: the hallway, then the bedroom, the mesh building | "I scan the room with my iPhone's LiDAR in about two minutes, using a free scanner app, and export the raw point cloud: two million points." |
| 3 | 0:40-1:05 | clip 01 (phone, upload/first run), then clip 02 (plan with ± values) | "RoomTwin uploads it to AWS, where my OpenCV 5 pipeline measures it in 13 seconds. It finds the floor, splits the scan into rooms at the doorways, fits every wall, and finds the door and window. Every measurement shows its uncertainty: 4.81 metres, plus or minus 1 centimetre." |
| 4 | 1:05-1:30 | **[YOU]** tape on the wall, cut to the comparison table | "Does it hold up? Tape says ___, RoomTwin says ___." (Fill in the real numbers tonight.) |
| 5 | 1:30-2:05 | clip 08 (coach): Scan check, then Ask the coach, then the rescan decision with the wall pulsing, then the trace | "Scans have holes. Instead of dumping fifteen warnings on you, an agent inspects the pipeline's evidence. It reads the summary, looks at the wall images OpenCV drew, and decides. Here it found one metre of wall I never swept, and told me exactly where to stand. I rescanned that corner, and attempt two was accepted." |
| 6 | 2:05-2:25 | clip 03 (3D orbit) and clip 04 (walk, phone) | "Then it's a digital twin I can orbit or walk through, on my phone." |
| 7 | 2:25-3:05 | clip 05 (paste the IKEA link, sizes appear), then the door verdict, then clip 06 (route animation) | "Here's the question I actually care about. I paste an IKEA link and it reads the real dimensions, including the flat-pack boxes. Assembled, this sofa is 13 centimetres too wide for my bedroom door, even upright. But its 2-metre box gets in: carried on its side, with under 2 centimetres to spare at this hallway wall. So go slowly there." |
| 8 | 3:05-3:20 | clip 07 (build order playing) | "For an empty room it plans the build order: furthest from the door first, never blocking the way for the next piece." |
| 9 | 3:20-3:40 | `docs/img/architecture.png`, then `agent.png` | "Under the hood: OpenCV 5 for all the geometry, with TSDF fusion, odometry, plane and line fitting, and watershed room splitting. Along the way I found and worked around five OpenCV 5 binding bugs. It runs on an AWS Graviton server, and the agent sits between the pipeline and the person, advising, never acting on its own." |
| 10 | 3:40-3:50 | **[YOU]** outro | "RoomTwin: scan a room, and see what fits. Thanks for watching." |

## What Danish films tonight (phone, landscape, good light)
1. **Intro (clip 1):** 20 s, facing the camera, in the bedroom. Read the line above, or say it in your own words.
2. **Tape measure (clip 4):** measure one long wall with the tape. Hold the tape reading up to the camera for 2 seconds.
3. **Outro (clip 10):** 10 s.
4. **The rescan:** do the coach's corner rescan with screen recording on, and upload it in RoomTwin. This gives attempt 2 for clip 5.
