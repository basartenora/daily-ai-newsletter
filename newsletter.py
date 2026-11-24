import csv
import datetime
import html
import io
import os
import random
import re
import smtplib
import time
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


def get_available_models(client: genai.Client) -> list[str]:
    """Discovers active Gemini/Gemma models enabled for this API key, prioritizing gemini-3.8-flash."""
    discovered = ["gemini-3.8-flash"]
    try:
        for m in client.models.list():
            name = (m.name or "").replace("models/", "")
            methods = getattr(m, "supported_actions", None) or getattr(m, "supported_generation_methods", None) or []
            if any(k in name for k in ("gemini-", "gemma-3")) and "embed" not in name and "tts" not in name and "image" not in name:
                if not methods or "generateContent" in methods:
                    if name not in discovered:
                        discovered.append(name)
    except Exception as exc:
        print(f"Warning: Could not list models dynamically ({exc}).")
    print(f"Active model pool: {discovered[:5]}")
    return discovered[:5]


def generate_with_retry(client: genai.Client, prompt: str, temperature: float, model_pool: list[str]):
    """Calls Gemini with patient jittered backoff across verified active models."""
    last_error = None
    for model_name in model_pool:
        max_attempts = 8 if model_name == "gemini-3.8-flash" else 3
        for attempt in range(max_attempts):
            try:
                return client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=temperature),
                )
            except Exception as exc:
                last_error = exc
                err_str = str(exc)
                if "503" in err_str or "429" in err_str or "UNAVAILABLE" in err_str:
                    wait_sec = min(45.0, (5 * (1.6 ** attempt)) + random.uniform(1.0, 4.0))
                    print(f"Model {model_name} busy (attempt {attempt + 1}/{max_attempts}). Retrying in {wait_sec:.1f}s...")
                    time.sleep(wait_sec)
                else:
                    print(f"Skipping {model_name} ({err_str[:80]}...)")
                    break
    raise RuntimeError(f"All active model attempts failed: {last_error}")


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


def fetch_rss_news(max_per_feed: int = 4) -> str:
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
                clean_desc = re.sub(r"\s+", " ", clean_desc).strip()[:220]
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
    model_pool = get_available_models(client)

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
    scout_resp = generate_with_retry(client, scout_prompt, temperature=0.4, model_pool=model_pool)
    time.sleep(4)

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
STORY_1_P2: <Second paragraph: deeper cultural or craft reflection>
STORY_1_SOURCE: <Publication name>
STORY_1_URL: <Exact URL from feed>
STORY_1_IMAGE: <Exact IMAGE_URL from feed or NONE>

STORY_2_TAG: 02 / CULTURE & TECH
STORY_2_TITLE: <Sentence-case editorial headline>
STORY_2_P1: <First paragraph>
STORY_2_P2: <Second paragraph>
STORY_2_SOURCE: <Publication name>
STORY_2_URL: <Exact URL from feed>
STORY_2_IMAGE: <Exact IMAGE_URL from feed or NONE>

STORY_3_TAG: 03 / DESIGN & FUTURES
STORY_3_TITLE: <Sentence-case editorial headline>
STORY_3_P1: <First paragraph>
STORY_3_P2: <Second paragraph>
STORY_3_SOURCE: <Publication name>
STORY_3_URL: <Exact URL from feed>
STORY_3_IMAGE: <Exact IMAGE_URL from feed or NONE>

ONE_OBJECT: <Title/Artifact — 1-sentence note on why it endures>
ONE_TAB: <Essay/Archive — 1-sentence note on why it is worth reading later>

SHORTLIST:
{scout_resp.text}
"""
    editor_resp = generate_with_retry(client, editor_prompt, temperature=0.55, model_pool=model_pool)
    time.sleep(4)

    writer_prompt = f"""
You are the Lead Copy Editor for "Morning AI & Culture Dispatch".
Polish the prose below so it reads like an independent print journal—calm, intelligent, unhurried, and completely free of breathless AI hype or corporate jargon.
- Ensure every STORY_X_TITLE uses clean sentence case (capitalizing the first word and proper nouns, never ALL CAPS or forced lowercase).
- Keep all delimiter keys, URLs, and IMAGE URLs completely intact.
- Do NOT add any URLs to ONE_OBJECT or ONE_TAB.

DRAFT:
{editor_resp.text}
"""
    writer_resp = generate_with_retry(client, writer_prompt, temperature=0.35, model_pool=model_pool)

    return parse_edition(writer_resp.text)


def parse_edition(raw_text: str) -> dict:
    """Parses the structured delimiter output from the Writer agent."""
    data = {}
    for line in raw_text.splitlines():
        if ":" in line:
            key, val = line.split(":", 1)
            key = key.strip()
            if key in {
                "SIGNAL",
                "ONE_OBJECT",
                "ONE_TAB",
                "STORY_1_TAG", "STORY_1_TITLE", "STORY_1_P1", "STORY_1_P2", "STORY_1_SOURCE", "STORY_1_URL", "STORY_1_IMAGE",
                "STORY_2_TAG", "STORY_2_TITLE", "STORY_2_P1", "STORY_2_P2", "STORY_2_SOURCE", "STORY_2_URL", "STORY_2_IMAGE",
                "STORY_3_TAG", "STORY_3_TITLE", "STORY_3_P1", "STORY_3_P2", "STORY_3_SOURCE", "STORY_3_URL", "STORY_3_IMAGE",
            }:
                data[key] = val.strip()
    return data


# ==============================================================================
# 5. HTML RENDERER (LIGHT + DARK MODE + RESPONSIVE MOBILE)
# ==============================================================================
def build_html_email(edition: dict, date_label: str) -> str:
    """Renders the finished responsive HTML email with Dark Mode adaptation."""
    stories_html = []

    for idx in (1, 2, 3):
        tag = html.escape(edition.get(f"STORY_{idx}_TAG", f"0{idx} / DISPATCH"))
        title = html.escape(edition.get(f"STORY_{idx}_TITLE", ""))
        p1 = html.escape(edition.get(f"STORY_{idx}_P1", ""))
        p2 = html.escape(edition.get(f"STORY_{idx}_P2", ""))
        source = html.escape(edition.get(f"STORY_{idx}_SOURCE", "Source"))
        url = edition.get(f"STORY_{idx}_URL", "#")
        img_url = edition.get(f"STORY_{idx}_IMAGE", "")

        media_block = ""
        if img_url and img_url != "NONE" and is_valid_image_url(img_url):
            safe_img = html.escape(img_url, quote=True)
            media_block = f"""
            <div class="story-media" style="margin: 22px 0 18px 0; border: 1px solid #1B38B8; background-color: #EBEEF6;">
              <img src="{safe_img}" alt="{title}" style="display: block; width: 100%; max-height: 290px; object-fit: cover;" />
              <div class="media-caption" style="padding: 8px 12px; font-family: 'IBM Plex Mono', monospace; font-size: 10px; letter-spacing: 0.08em; text-transform: uppercase; color: #1B38B8; border-top: 1px solid rgba(27,56,184,0.25);">
                FIG. 0{idx} // ARCHIVE VISUAL VIA {source}
              </div>
            </div>
            """

        stories_html.append(f"""
        <div class="story-block" style="padding: 32px 30px; border-bottom: 1px solid rgba(27, 56, 184, 0.22);">
          <div style="margin-bottom: 14px;">
            <span class="pill-tag" style="display: inline-block; font-family: 'IBM Plex Mono', monospace; font-size: 11px; font-weight: 500; letter-spacing: 0.08em; color: #1B38B8; border: 1px solid #1B38B8; padding: 3px 10px; border-radius: 999px;">
              {tag}
            </span>
          </div>
          <h2 class="story-title" style="margin: 0 0 16px 0; font-family: 'Jost', 'Helvetica Neue', Arial, sans-serif; font-size: 27px; line-height: 1.24; font-weight: 600; color: #0E238C; letter-spacing: -0.01em;">
            {title}
          </h2>
          <p class="body-copy" style="margin: 0 0 14px 0; font-family: 'IBM Plex Mono', monospace; font-size: 14px; line-height: 1.72; color: #1A2340;">
            {p1}
          </p>
          <p class="body-copy" style="margin: 0 0 18px 0; font-family: 'IBM Plex Mono', monospace; font-size: 14px; line-height: 1.72; color: #1A2340;">
            {p2}
          </p>
          {media_block}
          <div class="story-meta" style="margin-top: 14px; font-family: 'IBM Plex Mono', monospace; font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase; color: #1B38B8;">
            <span>SOURCE // {source}</span>
            <span style="margin: 0 8px;">&bull;</span>
            <a href="{html.escape(url, quote=True)}" style="color: #1B38B8; font-weight: 600; text-decoration: underline;">READ ARTICLE &rarr;</a>
          </div>
        </div>
        """)

    signal_text = html.escape(edition.get("SIGNAL", ""))
    one_object = html.escape(edition.get("ONE_OBJECT", ""))
    one_tab = html.escape(edition.get("ONE_TAB", ""))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<meta name="color-scheme" content="light dark" />
<meta name="supported-color-schemes" content="light dark" />
<title>Morning AI &amp; Culture Dispatch</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:ital,wght@0,400;0,500;0,600;1,400&family=Jost:wght@500;600;700&display=swap');

  @media only screen and (max-width: 600px) {{
    .container {{
      width: 100% !important;
      border-left: none !important;
      border-right: none !important;
    }}
    .pad-mobile {{
      padding-left: 18px !important;
      padding-right: 18px !important;
    }}
    .hero-title {{
      font-size: 32px !important;
      line-height: 1.08 !important;
    }}
    .story-block {{
      padding: 26px 18px !important;
    }}
    .story-title {{
      font-size: 23px !important;
      line-height: 1.26 !important;
    }}
    .body-copy {{
      font-size: 13.5px !important;
      line-height: 1.68 !important;
    }}
    .signal-cell {{
      display: block !important;
      width: 100% !important;
      border-left: none !important;
      padding-left: 0 !important;
      padding-top: 10px !important;
    }}
    .signoff-wrap {{
      padding: 28px 18px 48px 18px !important;
    }}
  }}

  @media (prefers-color-scheme: dark) {{
    body, .outer-bg {{
      background-color: #090E1A !important;
    }}
    .container {{
      background-color: #0E1528 !important;
      border-color: #3655D6 !important;
    }}
    .top-bar, .signal-band, .footer-bar {{
      background-color: #111A33 !important;
      border-color: #2A43A8 !important;
      color: #A9BEFF !important;
    }}
    .hero-box {{
      background: linear-gradient(125deg, #101A38 0%, #162866 55%, #1D3AA8 100%) !important;
      border-color: #3655D6 !important;
    }}
    .hero-title {{
      color: #DCE5FF !important;
    }}
    .hero-sub {{
      color: #9EB5FF !important;
    }}
    .story-block {{
      border-color: #243885 !important;
    }}
    .pill-tag {{
      color: #AFC3FF !important;
      border-color: #4A6CF7 !important;
    }}
    .story-title {{
      color: #C9D8FF !important;
    }}
    .body-copy {{
      color: #DCE3F5 !important;
    }}
    .story-meta, .story-meta a {{
      color: #9BB3FF !important;
    }}
    .story-media {{
      background-color: #141E3B !important;
      border-color: #3655D6 !important;
    }}
    .media-caption {{
      color: #9BB3FF !important;
      border-color: #2A43A8 !important;
    }}
  }}
</style>
</head>
<body class="outer-bg" style="margin: 0; padding: 0; background-color: #ECEEF2; -webkit-text-size-adjust: 100%;">
  <table role="presentation" width="100%" cellspacing="0" cellpadding="0" class="outer-bg" style="background-color: #ECEEF2; padding: 20px 0;">
    <tr>
      <td align="center">
        <table role="presentation" width="640" cellspacing="0" cellpadding="0" class="container" style="max-width: 640px; width: 100%; background-color: #F6F7F9; border: 1px solid #1B38B8;">
          <tr>
            <td class="top-bar pad-mobile" style="padding: 12px 30px; border-bottom: 1px solid #1B38B8; font-family: 'IBM Plex Mono', monospace; font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase; color: #1B38B8;">
              <span>EDITION // {html.escape(date_label)}</span>
              <span style="float: right;">CRAFT &bull; CULTURE &bull; TECH</span>
            </td>
          </tr>
          <tr>
            <td class="hero-box pad-mobile" style="padding: 44px 30px 40px 30px; border-bottom: 1px solid #1B38B8; background: linear-gradient(125deg, #F6F7F9 0%, #DCE4FA 48%, #3456D1 86%, #132A9C 100%);">
              <div class="hero-sub" style="font-family: 'IBM Plex Mono', monospace; font-size: 11px; letter-spacing: 0.12em; text-transform: uppercase; color: #1B38B8; margin-bottom: 12px;">
                MORNING FIELD NOTES
              </div>
              <h1 class="hero-title" style="margin: 0; font-family: 'Jost', 'Helvetica Neue', Arial, sans-serif; font-size: 44px; line-height: 1.04; font-weight: 700; color: #0E238C; letter-spacing: -0.02em;">
                Morning AI &amp;<br />Culture Dispatch
              </h1>
            </td>
          </tr>
          <tr>
            <td class="signal-band pad-mobile" style="padding: 20px 30px; background-color: #EEF1F8; border-bottom: 1px solid #1B38B8;">
              <table role="presentation" width="100%" cellspacing="0" cellpadding="0">
                <tr>
                  <td width="135" valign="top" style="font-family: 'IBM Plex Mono', monospace; font-size: 11px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: #1B38B8; padding-right: 14px;">
                    TODAY'S SIGNAL //
                  </td>
                  <td class="signal-cell body-copy" valign="top" style="font-family: 'IBM Plex Mono', monospace; font-size: 13px; line-height: 1.62; color: #1A2340; border-left: 1px solid rgba(27,56,184,0.35); padding-left: 16px;">
                    {signal_text}
                  </td>
                </tr>
              </table>
            </td>
          </tr>
          <tr>
            <td>
              {''.join(stories_html)}
            </td>
          </tr>
          <tr>
            <td class="signoff-wrap" style="padding: 36px 30px 64px 30px;">
              <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="background-color: #10248A; border: 1px solid #0B1963; color: #F4F6FC;">
                <tr>
                  <td style="padding: 26px 26px 28px 26px;">
                    <div style="font-family: 'IBM Plex Mono', monospace; font-size: 10.5px; letter-spacing: 0.12em; text-transform: uppercase; color: #A6BCFF; margin-bottom: 16px;">
                      CURATED PAIR // FOR LATER TODAY
                    </div>
                    <p style="margin: 0 0 14px 0; font-family: 'IBM Plex Mono', monospace; font-size: 13.5px; line-height: 1.65; color: #F4F6FC;">
                      <strong style="color: #A6BCFF; letter-spacing: 0.06em;">ONE OBJECT //</strong> {one_object}
                    </p>
                    <p style="margin: 0; font-family: 'IBM Plex Mono', monospace; font-size: 13.5px; line-height: 1.65; color: #F4F6FC;">
                      <strong style="color: #A6BCFF; letter-spacing: 0.06em;">ONE TAB //</strong> {one_tab}
                    </p>
                  </td>
                </tr>
              </table>
            </td>
          </tr>
          <tr>
            <td class="footer-bar pad-mobile" style="padding: 18px 30px; border-top: 1px solid #1B38B8; font-family: 'IBM Plex Mono', monospace; font-size: 11px; line-height: 1.6; letter-spacing: 0.06em; color: #1B38B8;">
              <div>CURATED BY MULTI-AGENT EDITORIAL PIPELINE &bull; CRAFT, CULTURE &amp; AI</div>
              <div style="margin-top: 6px; font-size: 10px; opacity: 0.85;">
                NO LONGER WANT THESE NOTES? REPLY WITH &ldquo;UNSUBSCRIBE&rdquo; ANYTIME.
              </div>
            </td>
          </tr>
        </table>
      </td>
    </tr>
  </table>
</body>
</html>"""


# ==============================================================================
# 6. MAIN EXECUTION & GMAIL SMTP DISPATCH (PRIVATE BCC)
# ==============================================================================
def main():
    api_key = os.environ["GEMINI_API_KEY"]
    sender_email = os.environ["GMAIL_ADDRESS"]
    app_password = os.environ["GMAIL_APP_PASSWORD"]

    recipients = get_subscribers(default_email=sender_email)
    if not recipients:
        raise ValueError("No valid recipients found.")

    now = datetime.datetime.now(datetime.timezone.utc)
    today_iso = now.strftime("%Y-%m-%d")
    date_label = now.strftime("%a, %d %b %Y").upper()

    print("1. Fetching RSS feeds...")
    raw_digest = fetch_rss_news()

    print("2. Running Scout -> Editor-in-Chief -> Writer agents...")
    client = genai.Client(api_key=api_key)
    edition = run_editorial_pipeline(client, raw_digest, today_iso)

    print("3. Rendering responsive HTML email...")
    html_body = build_html_email(edition, date_label)

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"Morning AI & Culture Dispatch — {now.strftime('%b %d')}"
    msg["From"] = f"Morning AI & Culture Dispatch <{sender_email}>"
    msg["To"] = sender_email
    msg["Bcc"] = ", ".join(recipients)
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    print(f"4. Sending edition via Bcc to {len(recipients)} subscriber(s)...")
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender_email, app_password)
        server.sendmail(sender_email, recipients, msg.as_string())

    print("Done! Edition dispatched successfully.")


if __name__ == "__main__":
    main()
