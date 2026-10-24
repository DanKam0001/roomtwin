"""RoomTwin API + static web app.

  POST /api/extract   {url} -> furniture item (SPEC format)
  POST /api/scan      multipart file (.ply/.obj/.zip) -> {id, room_url, ...}; runs the OpenCV 5 pipeline
  GET  /api/scans/{id}/{file}  pipeline outputs (room.json, floorplan.png, ...)
  POST /api/coach     {id} -> scan-coach agent decision + trace
  GET  /api/image?url=  product-image proxy (for images without CORS headers)
  GET  /api/health
Everything else is served from ../web.
"""
import json
import os
import subprocess
import sys
import time
import uuid
import zipfile
from pathlib import Path

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import extract as extractor

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("ROOMTWIN_DATA", ROOT / "data")).resolve()
SCANS = DATA / "scans"
SCANS.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD = int(os.environ.get("ROOMTWIN_MAX_UPLOAD_MB", "400")) * 1024 * 1024
ALLOWED = {".ply", ".obj", ".zip"}

app = FastAPI(title="RoomTwin")


class ExtractReq(BaseModel):
    url: str


class CoachReq(BaseModel):
    id: str


def log_event(scan_dir: Path, event: dict):
    event["t"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(scan_dir / "events.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


@app.get("/api/health")
def health():
    return {"ok": True, "scans": len(list(SCANS.iterdir()))}


@app.post("/api/extract")
def api_extract(req: ExtractReq):
    if not req.url.startswith(("http://", "https://")):
        raise HTTPException(400, "Give a full product link starting with http.")
    try:
        return extractor.extract(req.url, use_llm=bool(os.environ.get("ANTHROPIC_API_KEY")))
    except LookupError as e:
        raise HTTPException(422, str(e))
    except httpx.HTTPError as e:
        raise HTTPException(502, f"Couldn't open that page ({type(e).__name__}). Type the sizes in instead.")


@app.get("/api/image")
def api_image(url: str):
    if not url.startswith(("http://", "https://")):
        raise HTTPException(400, "bad url")
    r = httpx.get(url, headers={"User-Agent": extractor.UA}, follow_redirects=True, timeout=15)
    ctype = r.headers.get("content-type", "")
    if r.status_code != 200 or not ctype.startswith("image/") or len(r.content) > 8_000_000:
        raise HTTPException(502, "image unavailable")
    return Response(r.content, media_type=ctype, headers={"Cache-Control": "public, max-age=86400"})


def _safe_extract_zip(zpath: Path, dest: Path):
    with zipfile.ZipFile(zpath) as z:
        for m in z.infolist():
            target = (dest / m.filename).resolve()
            if not str(target).startswith(str(dest.resolve())):
                raise HTTPException(400, "zip contains unsafe paths")
        z.extractall(dest)


def _pick_input(folder: Path) -> Path:
    """A raw RGB-D frame folder beats a point cloud beats a mesh."""
    for p in folder.rglob("*.json"):
        if p.name.startswith("frame_"):
            return p.parent
    for ext in (".ply", ".obj"):
        found = sorted(folder.rglob(f"*{ext}"), key=lambda p: p.stat().st_size, reverse=True)
        if found:
            return found[0]
    raise HTTPException(400, "No .ply/.obj or frame data found in the upload.")


def run_pipeline(inp: Path, out: Path) -> dict:
    cmd = [sys.executable, "-m", "roomtwin.process", str(inp), "-o", str(out)]
    t = time.time()
    proc = subprocess.run(cmd, cwd=ROOT / "pipeline", capture_output=True, text=True, timeout=900)
    (out / "pipeline.log").write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr, encoding="utf-8")
    if proc.returncode != 0 or not (out / "room.json").exists():
        raise HTTPException(500, "Processing failed: " + (proc.stderr.strip().splitlines() or ["unknown"])[-1])
    return {"seconds": round(time.time() - t, 1)}


@app.post("/api/scan")
async def api_scan(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED:
        raise HTTPException(400, "Upload a .ply, .obj or .zip from the scanner app.")
    sid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    d = SCANS / sid
    (d / "upload").mkdir(parents=True)
    raw = d / "upload" / f"scan{ext}"
    size = 0
    with open(raw, "wb") as f:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD:
                raise HTTPException(413, "File too large.")
            f.write(chunk)
    if ext == ".zip":
        _safe_extract_zip(raw, d / "upload")
        raw.unlink()
    inp = _pick_input(d / "upload")
    log_event(d, {"type": "upload", "file": file.filename, "bytes": size, "input": inp.name})
    stats = run_pipeline(inp, d)
    room = json.loads((d / "room.json").read_text(encoding="utf-8"))
    log_event(d, {"type": "processed", **stats, "walls": len(room.get("walls", [])),
                  "coverage": room.get("coverage", {}).get("score")})
    return {"id": sid, "room_url": f"/api/scans/{sid}/room.json", "processing_s": stats["seconds"],
            "qa": room.get("qa", []), "files": sorted(p.name for p in d.iterdir() if p.is_file())}


@app.get("/api/scans/{sid}/{name}")
def api_scan_file(sid: str, name: str):
    p = (SCANS / sid / name).resolve()
    if not str(p).startswith(str(SCANS.resolve())) or not p.is_file() or "upload" in p.parts:
        raise HTTPException(404, "not found")
    return FileResponse(p)


@app.post("/api/coach")
def api_coach(req: CoachReq):
    import coach  # imported lazily: needs the API key
    d = (SCANS / req.id).resolve()
    if not str(d).startswith(str(SCANS.resolve())) or not (d / "room.json").exists():
        raise HTTPException(404, "scan not found")
    result = coach.run(d)
    log_event(d, {"type": "coach", "decision": result.get("decision")})
    return JSONResponse(result)


app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")
