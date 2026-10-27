# RoomTwin

**Will it fit?** In a 2025 survey of 1,500 UK furniture buyers, 49% had ordered furniture that
didn't fit through the door or couldn't get up the stairs
([Fixtuur](https://furniturenews.net/resources/more-shoppers-now-try-buy-and-return-furniture-reports-fixtuur)).
Each failure means a return, a 10-20% restocking fee and a wasted delivery trip. RoomTwin answers
the question before anyone buys or loads a truck. It is built for furniture retailers (a "fits your
home?" check by the Buy button), removalists, and their customers.

Scan a room with a LiDAR phone and get a measured digital twin you can plan with. An OpenCV 5
pipeline on AWS Graviton turns the raw point cloud into walls with lengths and an uncertainty
each, doors and windows with clear width and height, furniture boxes and a coverage report. A scan
coach agent looks at that evidence (the numbers and the per-wall occupancy images the pipeline
draws) and either accepts the scan or asks for one specific rescan. In the web app you place
furniture from a product link or typed sizes, check whether it fits through each door and in which
orientation, search a route from the front door to where it goes, and get a build order.

**Live:** https://54-226-102-179.sslip.io/

Entry for the OpenCV AI Competition 2026, powered by AWS (Overall and Agentic Vision). Technical
report: [docs/REPORT.md](docs/REPORT.md). Devpost fields: [docs/DEVPOST.md](docs/DEVPOST.md).
Data contract: [docs/SPEC.md](docs/SPEC.md). Pipeline details and OpenCV 5 findings:
[pipeline/README.md](pipeline/README.md).

![Architecture](docs/img/architecture.png)

## Screenshots

Placeholders, to be captured from the live app:

1. `docs/img/screen_plan.png`: plan view of the real bedroom + hallway scan with wall lengths +/- sigma
2. `docs/img/screen_scancheck.png`: Scan check card with QA findings and the coach's rescan decision
3. `docs/img/screen_trace.png`: the coach's readable trace, wall pulsed on the plan
4. `docs/img/screen_doorfit.png`: "won't fit through the door, here's why" result for a sofa
5. `docs/img/screen_route.png`: route search from the entry door, with the stuck point marked
6. `docs/img/screen_buildorder.png`: build order steps
7. `docs/img/screen_walk.png`: 3D / walk view

Pipeline outputs already in the repo: [real scan floor plan](docs/img/real_bedroom_hallway_floorplan.png),
[synthetic L-room floor plan](docs/img/synthetic_lshape_floorplan.png),
[wall occupancy image](docs/img/synthetic_lshape_wall_w1.png).

## Quick start (local)

Requires Python 3.11+ and, for the web tests only, Node 20+. Dependencies are pinned in
[server/requirements.txt](server/requirements.txt) and [pipeline/requirements.txt](pipeline/requirements.txt)
(`opencv-python-headless==5.0.0.93`, `numpy==2.4.6`, `fastapi==0.141.1`, ...). The server runs the
pipeline with its own Python, so install both into one environment.

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r pipeline/requirements.txt -r server/requirements.txt

# optional: enables the scan coach and model-based link extraction
export ANTHROPIC_API_KEY=...           # Windows PowerShell: $env:ANTHROPIC_API_KEY="..."

cd server
uvicorn app:app --port 8000
# open http://127.0.0.1:8000/
```

Without an API key everything works except the scan coach (it needs the model) and link
extraction falls back to schema.org data and dimension patterns on the page.

Web app only, no server (bundled sample rooms, typed furniture sizes):

```bash
python -m http.server 8080 -d web      # open http://127.0.0.1:8080/
```

Pipeline only, from the command line:

```bash
cd pipeline
python -m roomtwin.process scan.ply -o out/          # .ply / .obj point cloud or mesh
python -m roomtwin.process frames/ -o out/           # RGB-D frame folder (ARKit / ScanNet / TUM layouts)
python -m roomtwin.process scan.ply -o out/ --aruco photos/ --marker-size 0.15   # + ArUco scale check
```

Outputs: `room.json`, `house.json`, `floorplan.png`, `walls/<id>.png`, `room_mesh.glb`,
`room_preview.ply` (formats in [docs/SPEC.md](docs/SPEC.md)).

Server environment variables: `ROOMTWIN_DATA` (scan storage, default `./data`),
`ROOMTWIN_MAX_UPLOAD_MB` (default 400), `ROOMTWIN_LLM_DAILY_CAP` (model calls per day, default 300),
`ANTHROPIC_API_KEY`.

## Deploy to AWS

The live instance is an EC2 `t4g.medium` (Graviton, arm64) running Ubuntu 24.04 arm64, with
Caddy for HTTPS and the app under systemd.

1. **Launch the instance** (done once by hand; not scripted in the repo). Equivalent AWS CLI:

   ```bash
   aws ec2 create-key-pair --key-name roomtwin --query KeyMaterial --output text > ~/.ssh/roomtwin.pem
   chmod 600 ~/.ssh/roomtwin.pem
   aws ec2 create-security-group --group-name roomtwin --description "RoomTwin web + SSH"
   aws ec2 authorize-security-group-ingress --group-name roomtwin --protocol tcp --port 22  --cidr <your-ip>/32
   aws ec2 authorize-security-group-ingress --group-name roomtwin --protocol tcp --port 80  --cidr 0.0.0.0/0
   aws ec2 authorize-security-group-ingress --group-name roomtwin --protocol tcp --port 443 --cidr 0.0.0.0/0
   aws ec2 run-instances --instance-type t4g.medium --image-id <ubuntu-24.04-arm64-ami> \
     --key-name roomtwin --security-groups roomtwin \
     --user-data file://infra/server-userdata.sh
   ```

   [infra/server-userdata.sh](infra/server-userdata.sh) runs on first boot: installs Python, Caddy
   and OpenCV's system libraries, creates the unprivileged `roomtwin` user and `/var/lib/roomtwin`,
   adds 2 GB swap, and writes the `roomtwin.service` systemd unit (uvicorn on 127.0.0.1:8000).

2. **Deploy the code** from a machine with bash, ssh and tar:

   ```bash
   export ANTHROPIC_API_KEY=...            # written to /etc/roomtwin.env (mode 600) on the server
   infra/deploy.sh <public-ip>
   # -> deployed: https://<a-b-c-d>.sslip.io/
   ```

   [infra/deploy.sh](infra/deploy.sh) waits for first boot to finish, copies `server/`, `pipeline/`
   and `web/`, creates a venv and installs the pinned requirements, writes the secret to
   `/etc/roomtwin.env`, writes the Caddyfile for `<ip-with-dashes>.sslip.io` (Caddy obtains the TLS
   certificate automatically), and restarts the service. Re-run it to update.

3. **Check:** `curl https://<ip-with-dashes>.sslip.io/api/health` returns `{"ok": true, ...}`.
   Logs: `journalctl -u roomtwin`; per-scan `events.jsonl`, `pipeline.log` and `coach_trace.json`
   in `/var/lib/roomtwin/scans/<id>/`.

No secrets are stored in the repository.

## Tests and evaluation

```bash
# pipeline: 33 tests, about 45 s (synthetic rooms, two-room house, RGB-D fusion, ArUco, mesh input,
# and a test that documents an OpenCV 5.0.0 TSDF bug)
cd pipeline && python -m pytest -q tests

# web: 20 tests (door fit, route search, house transforms, placement), node's built-in runner
cd web && node --test tests/*.test.js        # or: npm test

# evaluation against synthetic ground truth (measured vs truth table) and the 48-scan sweep
cd pipeline
python -m roomtwin.synth --out <data-dir>              # generate rooms with exact ground truth
python -m roomtwin.evaluate --data <data-dir> --md synthetic_table.md
python -m roomtwin.evaluate --data <data-dir> --sweep  # 4 rooms x 6 seeds x 2 noise levels
```

The default data folder is `D:/roomtwin_data/synth` (the author's machine); pass `--out` / `--data`.
Results are summarised in [docs/REPORT.md](docs/REPORT.md#8-evaluation): synthetic walls within
0.4 cm, 48/48 sweep scans pass, and a real iPhone 14 Pro bedroom + hallway scan.

## Repository layout

```
docs/
  SPEC.md            data contract: room.json, house.json, furniture item, logistics rules
  REPORT.md          technical report
  DEVPOST.md         submission form fields
  coach_trace_lshape.json   a real scan-coach trace
  img/               architecture, pipeline and agent diagrams (.mmd sources + .png), pipeline outputs
pipeline/            OpenCV 5 scan pipeline (Python)
  roomtwin/          process.py (entry), io_ply, rgbd (TSDF + odometry), gravity, rooms, walls,
                     openings, obstacles, qa, scalecheck, export, geometry, synth, evaluate
  tests/             pytest suite
  requirements.txt
server/              FastAPI app
  app.py             upload, run pipeline, serve outputs, daily model-call cap, static web app
  coach.py           scan coach agent (perception -> decision -> action, trace)
  extract.py         furniture dimensions from a product link
  requirements.txt
web/                 static three.js app, no build step
  index.html, css/, js/app.js, js/plan2d.js, js/scene3d.js, js/walk.js, js/scancheck.js
  js/logistics/      doorfit, route (configuration-space A*), plan (build order), worker
  js/core/           geometry, house/world model
  samples/           sample rooms and houses
  tests/             node tests
infra/
  server-userdata.sh EC2 first boot
  deploy.sh          copy, install, secrets, Caddy, systemd
```

## License

MIT, copyright (c) 2026 Danish Kamboh. See [LICENSE](LICENSE).
