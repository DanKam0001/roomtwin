"""Furniture dimensions from a product page.

Order of preference: schema.org JSON-LD on the page, then dimension patterns in the
visible text, then Claude reading the relevant part of the page. Every result says which
source it came from so the UI can show how much to trust it.
"""
import json
import re

import anthropic
import httpx
from bs4 import BeautifulSoup

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/141.0 Safari/537.36")
MODEL = "claude-opus-5-5"

ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "assembled_cm": {
            "type": "object",
            "properties": {"w": {"type": "number"}, "d": {"type": "number"}, "h": {"type": "number"}},
            "required": ["w", "d", "h"], "additionalProperties": False,
        },
        "packages_cm": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"w": {"type": "number"}, "d": {"type": "number"},
                               "h": {"type": "number"}, "count": {"type": "integer"}},
                "required": ["w", "d", "h", "count"], "additionalProperties": False,
            },
        },
        "flatpack": {"type": "boolean"},
        "diagonal_depth_cm": {"type": ["number", "null"]},
        "found": {"type": "boolean"},
        "notes": {"type": "string"},
    },
    "required": ["name", "assembled_cm", "packages_cm", "flatpack", "diagonal_depth_cm", "found", "notes"],
    "additionalProperties": False,
}

PROMPT = """You read furniture product pages. From the page text below, extract the product name,
its assembled outer dimensions in centimetres (w = width, side to side; d = depth, front to back;
h = height), and the shipping package dimensions if listed (IKEA calls these "Packaging" and lists
each package). Set flatpack true if it ships in boxes for self-assembly. diagonal_depth_cm is the
diagonal depth some sofa listings give (for tipping through doors), else null.
Only use numbers that appear on the page; convert inches or millimetres to cm. If the page has no
usable dimensions set found=false and put zeros. Mention any assumption in notes.

URL: {url}

PAGE TEXT:
{text}"""


def _num(v):
    if isinstance(v, dict):
        unit = (v.get("unitCode") or v.get("unitText") or "").upper()
        val = _num(v.get("value"))
        if val is None:
            return None
        return val * {"MMT": 0.1, "MM": 0.1, "INH": 2.54, "IN": 2.54, "MTR": 100, "M": 100}.get(unit, 1)
    try:
        return float(str(v).replace(",", ".").split()[0])
    except (TypeError, ValueError, IndexError):
        return None


def _walk(obj):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v)


def from_jsonld(soup):
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "")
        except json.JSONDecodeError:
            continue
        for node in _walk(data):
            types = node.get("@type")
            types = types if isinstance(types, list) else [types]
            if "Product" not in types:
                continue
            w, d, h = (_num(node.get(k)) for k in ("width", "depth", "height"))
            image = node.get("image")
            if isinstance(image, list):
                image = image[0] if image else None
            if isinstance(image, dict):
                image = image.get("url")
            if w and d and h:
                return {"name": node.get("name") or "", "assembled_cm": {"w": w, "d": d, "h": h},
                        "image": image, "source": "schema.org"}
            return {"name": node.get("name") or "", "image": image}
    return None


DIM_RE = re.compile(
    r"(\d{2,4}(?:[.,]\d+)?)\s*(?:cm)?\s*[x×]\s*(\d{2,4}(?:[.,]\d+)?)\s*(?:cm)?\s*[x×]\s*"
    r"(\d{2,4}(?:[.,]\d+)?)\s*(cm|mm|in|\")", re.I)


def from_text(text):
    m = DIM_RE.search(text)
    if not m:
        return None
    scale = {"cm": 1, "mm": 0.1, "in": 2.54, '"': 2.54}[m.group(4).lower()]
    w, d, h = (float(g.replace(",", ".")) * scale for g in m.group(1, 2, 3))
    return {"assembled_cm": {"w": w, "d": d, "h": h}, "source": "text pattern",
            "notes": f"matched '{m.group(0)}'; order assumed W x D x H"}


def relevant_text(soup, limit=14000):
    """Visible text, keeping the lines near dimension words so long pages still fit."""
    for t in soup(["script", "style", "noscript", "svg"]):
        t.decompose()
    lines = [ln.strip() for ln in soup.get_text("\n").splitlines() if ln.strip()]
    keys = re.compile(r"dimension|measure|width|depth|height|length|packag|cm\b|mm\b|inch", re.I)
    keep = set()
    for i, ln in enumerate(lines):
        if keys.search(ln):
            keep.update(range(max(0, i - 3), i + 4))
    head = lines[:40]
    body = [lines[i] for i in sorted(keep)]
    return "\n".join(head + ["..."] + body)[:limit]


def from_claude(url, text):
    client = anthropic.Anthropic()
    resp = client.beta.messages.create(
        model=MODEL,
        max_tokens=4000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": ITEM_SCHEMA}},
        messages=[{"role": "user", "content": PROMPT.format(url=url, text=text)}],
    )
    if resp.stop_reason in ("refusal", "max_tokens"):
        return None
    raw = next((b.text for b in resp.content if b.type == "text"), None)
    if not raw:
        return None
    data = json.loads(raw)
    if not data.get("found"):
        return None
    data["source"] = "page read by Claude"
    return data


def to_item(url, data, image=None):
    """Convert to the SPEC furniture item (metres)."""
    a = data["assembled_cm"]
    item = {
        "name": data.get("name") or url.rstrip("/").split("/")[-1].replace("-", " ")[:60],
        "url": url,
        "image": image or data.get("image"),
        "assembled": {k: round(a[k] / 100, 3) for k in ("w", "d", "h")},
        "packages": [{"w": round(p["w"] / 100, 3), "d": round(p["d"] / 100, 3),
                      "h": round(p["h"] / 100, 3), "count": p.get("count", 1)}
                     for p in data.get("packages_cm") or []],
        "flatpack": bool(data.get("flatpack") or data.get("packages_cm")),
        "source": data.get("source"),
        "notes": data.get("notes", ""),
    }
    if data.get("diagonal_depth_cm"):
        item["diagonal_depth"] = round(data["diagonal_depth_cm"] / 100, 3)
    return item


def extract(url, use_llm=True):
    r = httpx.get(url, headers={"User-Agent": UA, "Accept-Language": "en-AU,en;q=0.9"},
                  follow_redirects=True, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    og = soup.find("meta", property="og:image")
    image = og["content"] if og and og.get("content") else None
    ld = from_jsonld(soup) or {}
    image = image or ld.get("image")
    title = ld.get("name") or (soup.title.string.strip() if soup.title and soup.title.string else "")

    if "assembled_cm" in ld and not use_llm:
        return to_item(url, ld, image)
    text = relevant_text(soup)
    if use_llm:
        data = from_claude(url, text)
        if data:
            return to_item(url, data, image)
    if "assembled_cm" in ld:
        return to_item(url, ld, image)
    found = from_text(text)
    if found:
        found["name"] = title
        return to_item(url, found, image)
    raise LookupError("No dimensions found on that page. Type them in instead.")
