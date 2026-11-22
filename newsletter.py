import csv
import datetime
import html
import io
import os
import re
import smtplib
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from google import genai
from google.genai import types

# ==============================================================================
# 1. CONFIGURATION & EDITORIAL FEEDS
# ==============================================================================
RSS_FEEDS = [
    ("NYT Arts & Culture", "https://rss.nytimes.com/services/xml/rss/nyt/Arts.xml"),
    ("The Guardian Culture", "https://www.theguardian.com/culture/rss"),
    ("The Guardian Art & Design", "https://www.theguardian.com/artanddesign/rss"),
    ("Aeon Magazine", "https://aeon.co/feed.rss"),
    ("The Verge", "https://www.theverge.com/rss/index.xml"),
    ("Wired", "https://www.wired.com/feed/rss"),
    ("Ars Technica", "https://feeds.arstechnica.com/arstechnica/index"),
    ("MIT Technology Review", "https://www.technologyreview.com/feed/"),
]

MODEL_ID = "gemini-2.5-flash"


# ==============================================================================
# 2. SUBSCRIBER LOADER (GOOGLE SHEET CSV + GMAIL FALLBACK)
# ==============================================================================
def get_subscribers(default_email: str = "") -> list[str]:
    """Fetches active subscriber emails from the published Google Sheet CSV."""
    subscribers: set[str] = set()

    if default_email and "@" in default_email:
        subscribers.add(default_email.strip().lower())

    fallback = os.environ.get("RECIPIENT_EMAIL", "")
    for addr in fallback.split(","):
        if "@" in addr:
            subscribers.add(addr.strip().lower())

    csv_url = os.environ.get("SUBSCRIBERS_CSV_URL", "").strip()
    if not csv_url:
        return sorted(subscribers)

    try:
        req = urllib.request.Request(csv_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw_text = resp.read().decode("utf-8-sig")

        reader = csv.reader(io.StringIO(raw_text))
        rows = list(reader)
        if not rows:
            return sorted(subscribers)

        headers = [h.strip().lower() for h in rows[0]]
        email_idx = next((i for i, h in enumerate(headers) if "email" in h), None)
        status_idx = next((i for i, h in enumerate(headers) if "status" in h), None)

        for row in rows[1:]:
            if not row:
                continue

            # Skip only if explicitly marked inactive/unsubscribed
            if status_idx is not None and len(row) > status_idx:
                status = row[status_idx].strip().lower()
                if status in {"unsubscribed", "inactive", "paused", "no"}:
                    continue

            candidate = ""
            if email_idx is not None and len(row) > email_idx:
                candidate = row[email_idx].strip()
            else:
                candidate = next((cell.strip() for cell in row if "@" in cell), "")

            if "@" in candidate and "." in candidate:
                subscribers.add(candidate.lower())

        print(f"Loaded {len(subscribers)} active subscribers from Google Sheet.")
    except Exception as exc:
        print(f"Warning: Could not fetch Google Sheet CSV ({exc}). Using fallback.")

    return sorted(subscribers)


# ==============================================================================
# 3. RSS FETCHER & IMAGE EXTRACTOR
# ==============================================================================
def _extract_image_url(item: ET.Element) -> str:
    """Extracts the primary article image URL from common RSS media tags."""
    for elem in item.iter():
        tag = elem.tag.lower()
        if tag.endswith("content") or tag.endswith("thumbnail"):
            url = elem.attrib.get("url", "").strip()
            medium = elem.attrib.get("medium", "").lower()
            mime = elem.attrib.get("type", "").lower()
            if url.startswith("http") and (
                medium == "image"
                or mime.startswith("image/")
                or any(url.lower().split("?")[0].endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp"))
            ):
                return url
        elif tag.endswith("enclosure"):
            url = elem.attrib.get("url", "").strip()
            mime = elem.attrib.get("type", "").lower()
            if url.startswith("http") and mime.startswith("image/"):
                return url

    # Fallback: check for <img> inside description or content:encoded
    for elem in item.iter():
        if elem.text and "<img" in elem.text:
            match = re.search(r'<img[^>]+src=["\'](https?://[^"\']+)["\']', elem.text, re.IGNORECASE)
            if match:
                return match.group(1).strip()

    return ""


def is_valid_image_url(url: str) -> bool:
    """Verifies that an extracted image URL responds cleanly before embedding."""
    if not url or not url.startswith("http"):
        return False
    try:
        req = urllib.request.Request(
            url,
            method="HEAD",
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            content_type = resp.headers.get("Content-Type", "").lower()
            return resp.status < 400 and ("image" in content_type or content_type == "")
    except Exception:
        return False


def fetch_rss_news(max_per_feed: int = 6) -> str:
    """Fetches recent headlines, summaries, links, and image URLs across feeds."""
    entries = []
    for source_name, feed_url in RSS_FEEDS:
        try:
            req = urllib.request.Request(feed_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=12) as resp:
                xml_bytes = resp.read()
            root = ET.fromstring(xml_bytes)

            items = root.findall(".//item")[:max_per_feed]
            for item in items:
                title = (item.findtext("title") or "").strip()
                link = (item.findtext("link") or "").strip()
                raw_desc = (item.findtext("description") or "").strip()
                clean_desc = re.sub(r"<[^>]+>", " ", html.unescape(raw_desc))
                clean_desc = re.sub(r"\s+", " ", clean_desc).strip()[:360]
                img_url = _extract_image_url(item)

                if title and link:
                    entries.append(
                        f"SOURCE: {source_name}\n"
                        f"TITLE: {title}\n"
                        f"URL: {link}\n"
                        f"IMAGE_URL: {img_url or 'NONE'}\n"
                        f"SUMMARY: {clean_desc}\n"
                    )
        except Exception as exc:
            print(f"Warning: Skipping {source_name} ({exc})")

    return "\n---\n".join(entries)


# ==============================================================================
# 4. MULTI-AGENT EDITORIAL PIPELINE
# ==============================================================================
def run_editorial_pipeline(client: genai.Client, raw_feed_digest: str, today_str: str) -> dict:
    """Runs Scout -> Editor-in-Chief -> Writer to produce structured edition data."""

    # --- AGENT 1: THE SCOUT ---
    scout_prompt = f"""
You are the Cultural & Tech Scout for "Morning AI & Culture Dispatch" ({today_str}).
Review the RSS digest below and shortlist the 8 strongest stories that sit at the intersection of:
- Human creative craft (design, architecture, books, visual art, filmmaking, music, physical objects)
- Cultural shifts (how people live, read, pay attention, gather, and make meaning)
- New technologies & AI (creative tools, synthetic media, human-machine collaboration, digital preservation)

STRICT EXCLUSIONS:
- Skip dry corporate earnings, stock movements, enterprise SaaS funding, or routine chip announcements unless they carry a deep cultural or artistic consequence.
- Preserve exact URLs and IMAGE_URLs from the feed.

RSS DIGEST:
{raw_feed_digest}
"""
    scout_resp = client.models.generate_content(
        model=MODEL_ID,
        contents=scout_prompt,
        config=types.GenerateContentConfig(temperature=0.4),
    )

    # --- AGENT 2: THE EDITOR-IN-CHIEF ---
    editor_prompt = f"""
You are the Editor-in-Chief of "Morning AI & Culture Dispatch" ({today_str}).
From the Scout's shortlist below, select exactly 3 stories that weave Craft, Culture, and Technology/AI together into a cohesive, contemplative morning edition:
- Story 1 lens: 01 / CRAFT & AI
- Story 2 lens: 02 / CULTURE & TECH
- Story 3 lens: 03 / DESIGN & FUTURES

Also compose:
1. TODAY'S SIGNAL: A crisp 2-sentence editorial thesis connecting the thread across today's three stories.
2. ONE OBJECT / ONE TAB: A curated cultural sign-off pair outside the breaking news cycle:
   - ONE OBJECT: One tangible design artifact, book, architectural work, or tactile object worth noticing (no URLs).
   - ONE TAB: One timeless essay, exhibition archive, or cultural curiosity to open later today (no URLs).

Return ONLY valid structured plain text using this exact delimiter format:

SIGNAL: <2-sentence thesis>

STORY_1_TAG: 01 / CRAFT & AI
STORY_1_TITLE: <Sentence-case editorial headline — capitalize first word and proper nouns only>
STORY_1_P1: <First paragraph: concrete, vivid reporting on what happened and why it matters>
STORY_1_P2: <Second paragraph: deeper cultural or
