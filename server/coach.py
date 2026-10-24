"""Scan coach: an agent that inspects the OpenCV 5 pipeline's evidence and decides what
happens next.

Perception: the pipeline's room.json (walls with sigma, coverage gaps, QA codes) and the
per-wall occupancy images (walls/<id>.png) and floor plan it drew.
Decision: Claude, with tools to look at that evidence.
Action: exactly one of accept_scan / request_rescan / ask_user. A rescan request is shown
to the person scanning (human in the loop); nothing is changed automatically.
Every tool call and result summary is written to coach_trace.json for review.
"""
import base64
import json
import time
from pathlib import Path

import anthropic

MODEL = "claude-opus-5-5"
MAX_TURNS = 8

SYSTEM = """You coach someone scanning a room with an iPhone LiDAR app for RoomTwin, which turns
the scan into a measured floor plan for planning furniture and moving logistics.
The OpenCV pipeline has already measured the room. Your job is to decide whether the scan is
good enough to plan with, or what single most useful thing the person should rescan.

Look at the evidence before deciding: start with get_scan_summary, then view the walls or the
floor plan that look questionable (high sigma, low coverage, inferred walls, gaps, suspiciously
short walls, odd openings). Judge what matters for furniture planning: wall lengths, the door
(width and height decide what fits through), and big hidden regions. Small gaps behind existing
furniture are acceptable; a missing or inferred wall, an unmeasured door, or a sigma above 3 cm
on a long wall is not.

Finish by calling exactly one action tool:
- accept_scan when the plan is usable (list any caveats honestly),
- request_rescan with ONE concrete, physical instruction (where to stand, what to point at),
- ask_user when only the person can resolve it (e.g. "is the 0.4 m segment a wall or a wardrobe?").
Speak plainly to a non-expert. Never invent measurements that are not in the evidence."""

def _tool(name, desc, props, required):
    return {"name": name, "description": desc, "strict": True,
            "input_schema": {"type": "object", "properties": props,
                             "required": required, "additionalProperties": False}}


TOOLS = [
    _tool("get_scan_summary", "Measured walls (length, sigma, coverage, inferred), openings, "
          "ceiling, coverage gaps and the pipeline's QA findings.", {}, []),
    _tool("view_wall", "Look at one wall's occupancy image from the pipeline: dark = scanned "
          "surface, white = no points; openings and gaps are visible as holes.",
          {"wall_id": {"type": "string"}}, ["wall_id"]),
    _tool("view_floorplan", "Look at the top-down floor plan the pipeline drew, with dimensions.", {}, []),
    _tool("accept_scan", "The scan is good enough to plan furniture with.",
          {"summary": {"type": "string"}, "caveats": {"type": "array", "items": {"type": "string"}}},
          ["summary", "caveats"]),
    _tool("request_rescan", "Ask the person to rescan one specific region.",
          {"wall_id": {"type": "string"}, "instruction": {"type": "string"},
           "reason": {"type": "string"}}, ["wall_id", "instruction", "reason"]),
    _tool("ask_user", "Ask the person a question only they can answer.",
          {"question": {"type": "string"}, "reason": {"type": "string"}}, ["question", "reason"]),
]
ACTIONS = {"accept_scan": "accept", "request_rescan": "rescan", "ask_user": "ask"}


def _png_block(path: Path):
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                        "data": base64.b64encode(path.read_bytes()).decode()}}


def _summary(room: dict) -> str:
    keep = {
        "walls": [{k: w.get(k) for k in ("id", "length", "sigma", "height", "coverage", "inferred")}
                  for w in room.get("walls", [])],
        "openings": room.get("openings", []),
        "ceiling_height": room.get("ceiling_height"),
        "floor_area_m2": room.get("floor", {}).get("area_m2"),
        "coverage": room.get("coverage"),
        "qa": room.get("qa"),
        "obstacles": [{"id": o.get("id"), "size_m": [round(b - a, 2) for a, b in zip(o["min"], o["max"])]}
                      for o in room.get("obstacles", [])],
    }
    return json.dumps(keep)


def run(scan_dir: Path) -> dict:
    room = json.loads((scan_dir / "room.json").read_text(encoding="utf-8"))
    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": "A new scan has been processed. Inspect it and decide."}]
    trace, result = [], None
    t0 = time.time()

    for _ in range(MAX_TURNS):
        resp = client.beta.messages.create(
            model=MODEL, max_tokens=8000, system=SYSTEM, tools=TOOLS,
            betas=["server-side-fallback-2026-07-01"], fallbacks="default",
            output_config={"effort": "medium"}, messages=messages,
        )
        messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason != "tool_use":
            break
        results = []
        for block in resp.content:
            if block.type != "tool_use":
                continue
            name, args = block.name, block.input
            if name in ACTIONS:
                result = {"decision": ACTIONS[name], **args}
                trace.append({"tool": name, "input": args})
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": "recorded"})
                continue
            if name == "get_scan_summary":
                content, note = _summary(room), "summary"
            elif name == "view_wall":
                p = scan_dir / "walls" / f"{args['wall_id']}.png"
                if not p.exists() and room.get("id"):
                    p = scan_dir / "walls" / f"{room['id']}_{args['wall_id']}.png"
                content = [_png_block(p)] if p.exists() else f"No image for wall {args['wall_id']}."
                note = p.name if p.exists() else "missing"
            elif name == "view_floorplan":
                p = scan_dir / "floorplan.png"
                content, note = [_png_block(p)], "floorplan.png"
            else:
                content, note = f"Unknown tool {name}", "error"
            trace.append({"tool": name, "input": args, "result": note})
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": content})
        messages.append({"role": "user", "content": results})
        if result:
            break

    if result is None:
        text = " ".join(b.text for b in messages[-1]["content"] if getattr(b, "type", "") == "text") \
            if messages[-1]["role"] == "assistant" else ""
        result = {"decision": "ask", "question": text or "The coach could not decide; check the floor plan.",
                  "reason": "no action tool called"}

    out = {**result, "trace": trace, "model": MODEL, "seconds": round(time.time() - t0, 1),
           "coverage": room.get("coverage", {}).get("score")}
    (scan_dir / "coach_trace.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out
