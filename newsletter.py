import csv
import html
import io
import os
import re
import smtplib
import time
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from google import genai
from google.genai import types

# 1. CLIENTE CONFIGURADO EN V1BETA
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(api_version="v1beta")
)

# FUENTES RSS OFICIALES DE PRIMER NIVEL (INCLUYENDO THE NEW YORK TIMES)
RSS_FEEDS = [
    ("New York Times Arts & Culture", "https://rss.nytimes.com/services/xml/rss/nyt/Arts.xml"),
    ("New York Times Tech", "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml"),
    ("The Guardian Culture", "https://www.theguardian.com/culture/rss"),
    ("The Guardian Art & Design", "https://www.theguardian.com/artanddesign/rss"),
    ("Wired", "https://www.wired.com/feed/rss"),
    ("MIT Technology Review", "https://www.technologyreview.com/feed/"),
    ("Aeon Magazine", "https://aeon.co/feed.rss"),
    ("The Verge", "https://www.theverge.com/rss/index.xml"),
    ("Ars Technica", "https://feeds.arstechnica.com/arstechnica/index"),
    ("The Guardian Tech", "https://www.theguardian.com/uk/technology/rss")
]

def get_available_models() -> list[str]:
    """Devuelve únicamente los 3 modelos principales de texto para evitar esperas largas."""
    return [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-2.0-flash"
    ]

def clean_xml_text(raw_text: str) -> str:
    if not raw_text:
        return ""
    text = re.sub(r"<[^>]+>", " ", raw_text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()

def fetch_rss_news(max_per_feed: int = 5) -> str:
    """Descarga titulares, resúmenes y URLs reales directamente de los RSS oficiales."""
    collected_items = []
    headers = {"User-Agent": "Mozilla/5.0 (compatible; MorningDispatchBot/1.0)"}

    for source_name, feed_url in RSS_FEEDS:
        try:
            req = urllib.request.Request(feed_url, headers=headers)
            with urllib.request.urlopen(req, timeout=6) as resp:
                xml_data = resp.read()
            
            root = ET.fromstring(xml_data)
            items = root.findall(".//item")
            is_atom = False
            if not items:
                items = root.findall(".//{http://www.w3.org/2005/Atom}entry")
                is_atom = True

            count = 0
            for item in items:
                if count >= max_per_feed:
                    break

                image_url = ""
                if not is_atom:
                    title = clean_xml_text(item.findtext("title", ""))
                    link = (item.findtext("link") or "").strip()
                    desc = clean_xml_text(item.findtext("description", ""))
                    pub_date = (item.findtext("pubDate") or "").strip()
                    media = item.find("{http://search.yahoo.com/mrss/}content")
                    if media is None:
                        media = item.find("{http://search.yahoo.com/mrss/}thumbnail")
                    if media is None:
                        media = item.find("enclosure")
                    if media is not None:
                        cand_url = (media.attrib.get("url") or "").strip()
                        if cand_url.startswith("http"):
                            image_url = cand_url
                else:
                    title = clean_xml_text(item.findtext("{http://www.w3.org/2005/Atom}title", ""))
                    link_el = item.find("{http://www.w3.org/2005/Atom}link")
                    link = (link_el.attrib.get("href", "") if link_el is not None else "").strip()
                    desc = clean_xml_text(
                        item.findtext("{http://www.w3.org/2005/Atom}summary", "")
                        or item.findtext("{http://www.w3.org/2005/Atom}content", "")
                    )
                    pub_date = (item.findtext("{http://www.w3.org/2005/Atom}published") or "").strip()
                    media = item.find("{http://search.yahoo.com/mrss/}content") or item.find("{http://search.yahoo.com/mrss/}thumbnail")
                    if media is not None:
                        cand_url = (media.attrib.get("url") or "").strip()
                        if cand_url.startswith("http"):
                            image_url = cand_url

                if title and link.startswith("http"):
                    collected_items.append(
                        f"- SOURCE: {source_name} | DATE: {pub_date}\n"
                        f"  TITLE: {title}\n"
                        f"  URL: {link}\n"
                        f"  IMAGE_URL: {image_url or 'NONE'}\n"
                        f"  SUMMARY: {desc[:380]}"
                    )
                    count += 1
        except Exception as e:
            print(f"Aviso leyendo RSS de {source_name}: {e}")

    if not collected_items:
        raise RuntimeError("No se pudieron descargar artículos de los feeds RSS.")

    print(f"Recopilados {len(collected_items)} artículos reales desde RSS oficiales.")
    return "\n\n".join(collected_items)

def verify_url(url: str) -> bool:
    """Comprueba que la URL sea accesible y no devuelva un 404."""
    if not url or not url.startswith("http"):
        return False
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; DispatchBot/1.0)"}
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status < 400
    except urllib.error.HTTPError as e:
        return e.code in (401, 403)
    except Exception:
        return False

def generate_resilient(prompt: str, models: list[str], temp: float = 0.3) -> str:
    """Ejecución de texto puro (sin Grounding) compatible con cuota gratuita."""
    config = types.GenerateContentConfig(temperature=temp)

    last_error = ""
    for model_name in models:
        for attempt in range(1, 3):
            try:
                print(f"-> Calling {model_name} (attempt {attempt}/2)...")
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=config
                )
                
                if response and response.text:
                    return response.text
            except Exception as e:
                last_error = str(e)
                print(f"Notice on {model_name}: {last_error}")

                if "429" in last_error or "resource_exhausted" in last_error.lower():
                    print(f"Limit hit on {model_name}. Switching model...")
                    time.sleep(1)
                    break

                if "503" in last_error or "high demand" in last_error.lower() or "disconnected" in last_error.lower():
                    wait_sec = attempt * 5
                    print(f"Server busy. Waiting {wait_sec}s...")
                    time.sleep(wait_sec)
                else:
                    break

    raise RuntimeError(f"Todos los modelos fallaron. Último error de Google: {last_error}")

def get_subscribers() -> list[str]:
    """Combina tu correo principal, posibles contactos fijos y los inscritos en Google Sheets."""
    subscribers = set()
    owner_email = os.environ.get("GMAIL_ADDRESS", "").strip()
    if owner_email:
        subscribers.add(owner_email.lower())

    extra = os.environ.get("EXTRA_SUBSCRIBERS", "")
    for email in extra.split(","):
        clean = email.strip().lower()
        if "@" in clean and "." in clean:
            subscribers.add(clean)

    csv_url = os.environ.get("SUBSCRIBERS_CSV_URL", "").strip()
    if csv_url:
        try:
            req = urllib.request.Request(csv_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                content = resp.read().decode("utf-8")
                reader = csv.reader(io.StringIO(content))
                next(reader, None)
                for row in reader:
                    for cell in row:
                        clean = cell.strip().lower()
                        if "@" in clean and "." in clean and " " not in clean:
                            subscribers.add(clean)
        except Exception as e:
            print(f"Aviso leyendo CSV de suscriptores ({e}). Enviando a lista local.")

    return sorted(list(subscribers))

def send_email_to_all(subject: str, plain_text: str, html_content: str):
    gmail_user = os.environ.get("GMAIL_ADDRESS")
    gmail_password = os.environ.get("GMAIL_APP_PASSWORD")
    recipients = get_subscribers()

    print(f"Preparando envío individual para {len(recipients)} suscriptor(es)...")

    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=20) as server:
        server.login(gmail_user, gmail_password)
        for recipient in recipients:
            try:
                msg = MIMEMultipart("alternative")
                msg["Subject"] = subject
                msg["From"] = f"Morning Dispatch <{gmail_user}>"
                msg["To"] = recipient

                msg.attach(MIMEText(plain_text, "plain"))
                msg.attach(MIMEText(html_content, "html"))

                server.send_message(msg)
                print(f"-> Enviado correctamente a: {recipient}")
                time.sleep(1.5)
            except Exception as send_err:
                print(f"Error enviando a {recipient}: {send_err}")

    print("Dispatch completed and sent successfully!")

# =========================================================================
# EJECUCIÓN PRINCIPAL
# =========================================================================
try:
    ACTIVE_MODELS = get_available_models()
    print(f"Detected active models: {ACTIVE_MODELS}")
    today_str = datetime.now().strftime("%Y // %m.%d").upper()
    issue_month_str = datetime.now().strftime("ISSUE %Y.%m").upper()
    readable_date = datetime.now().strftime("%B %d, %Y")

    # PASO 0: DESCARGA DE NOTICIAS REALES DESDE RSS
    print("0/3 -> Fetching live stories from tier-1 RSS feeds...")
    raw_rss_feed = fetch_rss_news(max_per_feed=5)

    # AGENTE 1: SCOUT (FILTRA Y ESTRUCTURA 6 CANDIDATOS DEL FEED RSS)
    scout_prompt = f"""
    You are a Senior Research Scout. Today is {readable_date}.
    Below is a live feed of real articles just published by top-tier outlets:

    {raw_rss_feed}

    Select 6 distinct, high-signal articles from the list above where CREATIVE CRAFT, HUMAN CULTURE, and NEW TECHNOLOGY / AI intersect:
    - Look for stories about how artists, designers, architects, writers, filmmakers, or everyday communities are shaping—or pushing back against—AI and emerging tools.
    - Avoid dry corporate earnings, stock news, or routine enterprise software updates.

    STRICT RULES:
    - Use ONLY articles present in the feed above.
    - Copy the EXACT URL and EXACT Source Name from the feed—never alter a single character of a URL.

    For each of the 6 candidates, provide:
    - Candidate Number (1 to 6)
    - Raw Title & Exact Publication Name
    - Exact URL
    - Exact IMAGE_URL (copy verbatim from feed, or NONE)
    - Concrete facts, figures, names, and core tension based on the summary (3 sentences).
    """

    print("1/3 -> Starting Scout step (selecting 6 candidates)...")
    scout_output = generate_resilient(scout_prompt, ACTIVE_MODELS, temp=0.2)

    # AGENTE 2: EDITOR JEFE (CURADURÍA, MACRO SIGNAL Y ONE OBJECT / ONE TAB)
    editor_prompt = f"""
    You are the Editor-in-Chief of "Morning AI & Culture Dispatch", an avant-garde minimalist daily briefing exploring the intersection of human craft, culture, and technology.
    Review the 6 candidate stories gathered by the Scout below:

    {scout_output}

    YOUR EDITORIAL MISSION:
    1. Discard weak stories (routine product updates, dry corporate news, or thin summaries).
    2. Select the BEST 3 stories—each one blending creative craft, cultural resonance, and new technology/AI from a distinct angle:
       - 01 / CRAFT & AI (How makers, designers, or artists are reshaping physical/digital craft with or against AI)
       - 02 / CULTURE & TECH (How new technologies are shifting shared rituals, taste, art, or human perception)
       - 03 / DESIGN & FUTURES (Emerging creative tools, material breakthroughs, or aesthetic counter-movements)
    3. Define TODAY'S MACRO SIGNAL:
       - A punchy 2-to-4 character stat or multiplier (e.g., "3X", "40%", "24H", "1:1") that captures the overarching pattern across today's edition (do NOT simply repeat a stat verbatim from Story 03).
       - A crisp 1-sentence explanation of what that signal reveals about today's edition.
    4. Define THE THREAD: 2 crisp sentences explaining the hidden tension connecting all three stories today.
    5. For each of the 3 selected stories, define:
       - Exact Category label (CRAFT & AI, CULTURE & TECH, or DESIGN & FUTURES)
       - Estimated reading time (e.g., "80 SEC READ", "90 SEC READ")
       - A punchy sentence-case headline (capitalize the first word and proper nouns, max 6 words)
       - Verified facts, names, and numbers to preserve
       - Exact Publication Name, exact URL, and exact IMAGE_URL (must match the Scout's URLs verbatim).
    6. Define a curated cultural sign-off pair ("ONE OBJECT / ONE TAB") inspired by the mood of today's stories:
       - OBJECT: One real, well-known physical design artifact, industrial object, book, or exhibition (8-14 words, no unverified URLs).
       - TAB: One real, enduring essay, historical archive, or cultural concept worth looking up later (8-14 words, no unverified URLs).
    """

    print("2/3 -> Starting Editor-in-Chief step (curating top 3)...")
    editor_brief = generate_resilient(editor_prompt, ACTIVE_MODELS, temp=0.3)

    # AGENTE 3: WRITER (REDACCIÓN CLARA Y MAQUETACIÓN HTML ATMOSFÉRICA)
    writer_prompt = f"""
    You are a sharp editorial writer. Turn the Editor-in-Chief's approved brief into a crisp, effortless morning read.

    STYLE & RIGOR RULES:
    - Clarity first: Short sentences, active verbs, concrete details (names, numbers, companies) from the brief.
    - Never invent facts, quotes, or URLs not present in the brief below.
    - Avoid convoluted metaphors, academic jargon, and AI clichés (never use delve, tapestry, landscape, paradigm, nuanced, labyrinth).
    - Keep paragraphs short (3 to 4 lines max).
    - DO NOT include takeaway boxes anywhere.
    - If a story has a valid IMAGE_URL (not NONE), include the <div class="story-media"> block right after the second paragraph and before <div class="source-row">. If IMAGE_URL is NONE, omit <div class="story-media"> completely for that story.

    Editor-in-Chief Brief:
    {editor_brief}

    Generate ONLY valid raw HTML (no markdown code blocks, no ```html) following this exact structure:
    1. Macro Signal Band:
       <div class="ticker-band">
         <div class="ticker-stat-wrap">
           <div class="ticker-tag">TODAY’S SIGNAL</div>
           <div class="ticker-number">STAT_VALUE</div>
         </div>
         <div class="ticker-desc">One crisp sentence explaining the macro pattern.</div>
       </div>

    2. Main Content Wrapper:
       <div class="content">
         <div class="thread-box">
           <div class="thread-label">THE THREAD // WHY THESE THREE STORIES CONNECT</div>
           <p class="intro-text"><span class="dropcap">FIRST_LETTER</span>REST_OF_2_SENTENCE_INTRO</p>
         </div>

         <!-- Repeat for each of the 3 stories (01 / CRAFT & AI, 02 / CULTURE & TECH, 03 / DESIGN & FUTURES) -->
         <div class="story-block">
           <div class="story-meta-row">
             <span class="pill-tag">01 / CATEGORY</span>
             <span class="read-badge">90 SEC READ</span>
           </div>
           <h2 class="story-title">Sentence-case headline</h2>
           <p class="subtitle">One plain, direct sentence summarizing the core development.</p>
           <p>First short factual paragraph.</p>
           <p>Second short factual paragraph.</p>
           <!-- Include ONLY if IMAGE_URL is not NONE -->
           <div class="story-media">
             <img src="EXACT_IMAGE_URL" alt="Editorial visual" class="story-img" />
             <div class="story-caption">FIG. 01 // PUBLICATION NAME ARCHIVE</div>
           </div>
           <div class="source-row">
             <span class="source-label">SOURCE // PUBLICATION NAME</span>
             <a href="EXACT_URL" class="source-link" target="_blank">READ ARTICLE &rarr;</a>
           </div>
         </div>

         <!-- Curated Cultural Pair Sign-Off Card -->
         <div class="verdict-box">
           <div class="verdict-header">CURATED PAIR // FOR LATER TODAY</div>
           <div class="verdict-row"><span class="verdict-tag">ONE OBJECT</span> Concise 8-14 word description of the object.</div>
           <div class="verdict-row"><span class="verdict-tag">ONE TAB</span> Concise 8-14 word description of the essay or archive.</div>
         </div>
       </div>
    """

    print("3/3 -> Starting Writer step (drafting HTML)...")
    writer_output = generate_resilient(writer_prompt, ACTIVE_MODELS, temp=0.4)
    newsletter_body_html = writer_output.replace("```html", "").replace("```", "").strip()

    # AUDITORÍA AUTOMÁTICA DE ENLACES Y ESTRUCTURA HTML
    print("Auditing links and HTML structure...")
    urls_found = re.findall(r'href="(https?://[^"]+)"', newsletter_body_html)
    for url in urls_found:
        if not verify_url(url):
            print(f"Removing broken or unreachable URL: {url}")
            newsletter_body_html = re.sub(
                rf'<a href="{re.escape(url)}"[^>]*>READ ARTICLE &rarr;</a>',
                '<span class="source-label">PRINT / ARCHIVE EDITION</span>',
                newsletter_body_html
            )

    imgs_found = re.findall(r'<img[^>]+src="([^"]+)"', newsletter_body_html)
    for img_url in imgs_found:
        if img_url == "NONE" or not verify_url(img_url):
            print(f"Removing broken or missing image: {img_url}")
            newsletter_body_html = re.sub(
                rf'<div class="story-media">\s*<img[^>]+src="{re.escape(img_url)}"[^>]*>\s*(?:<div class="story-caption">.*?</div>\s*)?</div>',
                '',
                newsletter_body_html,
                flags=re.DOTALL
            )

    if newsletter_body_html.count('class="story-block"') < 3:
        raise RuntimeError("Validación fallida: El HTML generado no contiene las 3 historias completas.")

    # PLANTILLA EDITORIAL ATMOSFÉRICA (COBALTO PROFUNDO + BLANCO GALERÍA)
    html_template = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=Inter:wght@400;500;600;700&family=Jost:wght@500;600;700&display=swap" rel="stylesheet">
  <style>
    body { margin: 0; padding: 0; background-color: #E5E8EE; font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Helvetica Neue', Helvetica, Arial, sans-serif; color: #0A2496; line-height: 1.65; -webkit-font-smoothing: antialiased; }
    .container { max-width: 620px; margin: 28px auto; background-color: #F5F6F8; border: 1px solid rgba(10, 36, 150, 0.24); border-radius: 0px; }
    
    /* Atmospheric Cobalt & Mist Hero Header */
    .hero-wrap {
      background-color: #E7ECF6;
      background-image:
        radial-gradient(circle at 82% 38%, rgba(6, 25, 118, 0.72) 0%, rgba(17, 50, 196, 0.42) 32%, rgba(231, 236, 246, 0) 68%),
        radial-gradient(circle at 62% 85%, rgba(10, 36, 150, 0.52) 0%, rgba(60, 99, 217, 0.22) 40%, rgba(231, 236, 246, 0) 72%),
        linear-gradient(155deg, #F4F6FA 0%, #E7ECF6 55%, #DCE4F5 100%);
      border-bottom: 1px solid rgba(10, 36, 150, 0.24);
    }
    .top-bar { padding: 13px 30px; border-bottom: 1px solid rgba(10, 36, 150, 0.22); font-family: 'IBM Plex Mono', monospace; font-size: 11px; font-weight: 700; letter-spacing: 1.3px; display: flex; justify-content: space-between; text-transform: uppercase; color: #071C7A; background-color: rgba(244, 246, 250, 0.55); }
    .header { padding: 44px 30px 44px 30px; }
    .header-title { font-family: 'Jost', 'Futura', 'Century Gothic', 'Trebuchet MS', sans-serif; margin: 0; font-size: 44px; font-weight: 600; line-height: 1.02; letter-spacing: -1.4px; color: #061A78; max-width: 390px; }
    .header-sub { margin: 18px 0 0 0; font-size: 14px; font-weight: 500; max-width: 350px; line-height: 1.52; color: #0A2496; }

    /* Macro Signal Ticker Band */
    .ticker-band { background-color: #EBF0FA; border-bottom: 1px solid rgba(10, 36, 150, 0.22); padding: 18px 30px; display: flex; align-items: center; gap: 20px; }
    .ticker-stat-wrap { flex-shrink: 0; }
    .ticker-tag { font-family: 'IBM Plex Mono', monospace; font-size: 9.5px; font-weight: 700; letter-spacing: 1.2px; text-transform: uppercase; color: #3A50AC; margin-bottom: 2px; }
    .ticker-number { font-family: 'Jost', 'Futura', sans-serif; font-size: 36px; font-weight: 700; line-height: 0.95; letter-spacing: -1px; color: #0A2496; }
    .ticker-desc { font-size: 13.5px; line-height: 1.48; font-weight: 500; color: #0A2496; border-left: 1px solid rgba(10, 36, 150, 0.22); padding-left: 18px; }

    /* Editorial Content & Unifying Thread */
    .content { padding: 34px 30px 56px 30px; }
    .thread-box { padding-bottom: 28px; margin: 0 0 34px 0; border-bottom: 1px dashed rgba(10, 36, 150, 0.28); }
    .thread-label { font-family: 'IBM Plex Mono', monospace; font-size: 10px; font-weight: 700; letter-spacing: 1.3px; text-transform: uppercase; color: #3A50AC; margin-bottom: 10px; }
    .intro-text { font-size: 15px; margin: 0; line-height: 1.68; color: #0A2496; }
    .dropcap { float: left; font-family: 'Jost', sans-serif; font-size: 42px; font-weight: 700; line-height: 0.85; padding: 4px 10px 4px 0; margin-top: 2px; color: #0A2496; }

    /* Story Blocks & Frosted Cobalt Pills */
    .story-block { margin-bottom: 40px; padding-bottom: 34px; border-bottom: 1px solid rgba(10, 36, 150, 0.22); }
    .story-block:last-of-type { border-bottom: none; margin-bottom: 26px; padding-bottom: 6px; }
    .story-meta-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; gap: 10px; }
    .pill-tag { font-family: 'IBM Plex Mono', monospace; font-size: 11px; font-weight: 600; letter-spacing: 0.9px; text-transform: uppercase; color: #FFFFFF; background-color: #0D2BB0; background-image: linear-gradient(135deg, #0D2BB0 0%, #2448DB 100%); border: 1px solid rgba(255, 255, 255, 0.35); padding: 4px 13px; border-radius: 999px; }
    .read-badge { font-family: 'IBM Plex Mono', monospace; font-size: 10px; font-weight: 600; letter-spacing: 0.8px; padding: 3px 9px; border: 1px solid rgba(10, 36, 150, 0.25); border-radius: 999px; color: #3A50AC; text-transform: uppercase; }
    @media (prefers-color-scheme: dark) {
      body { background-color: #060A14 !important; color: #C4D4FF !important; }
      .container { background-color: #0E1528 !important; border-color: rgba(168, 192, 255, 0.28) !important; }
      .hero-wrap { background-color: #091338 !important; background-image: linear-gradient(155deg, #091338 0%, #0E216B 55%, #16339E 100%) !important; border-bottom-color: rgba(168, 192, 255, 0.28) !important; }
      .top-bar { color: #DCE6FF !important; background-color: rgba(9, 19, 56, 0.55) !important; border-bottom-color: rgba(168, 192, 255, 0.28) !important; }
      .header-title { color: #FFFFFF !important; }
      .header-sub { color: #DCE6FF !important; }
      .ticker-band { background-color: #15213E !important; border-bottom-color: rgba(168, 192, 255, 0.28) !important; }
      .ticker-tag, .thread-label, .read-badge, .story-caption, .source-label, .footer { color: #93AEFF !important; }
      .ticker-number, .ticker-desc, .intro-text, .dropcap, .story-title, .subtitle, p, .source-link { color: #C4D4FF !important; }
      .ticker-desc, .thread-box, .story-block, .story-media, .story-caption, .footer, .read-badge { border-color: rgba(168, 192, 255, 0.28) !important; }
      .story-media { background-color: #15213E !important; }
    }

    @media only screen and (max-width: 600px) {
      .top-bar { padding: 11px 16px !important; font-size: 10px !important; }
      .footer { padding: 20px 16px !important; font-size: 10px !important; flex-wrap: wrap !important; gap: 8px !important; }
      .content { padding: 24px 16px 46px 16px !important; }
      .header { padding: 28px 16px !important; }
      .header-title { font-size: 30px !important; line-height: 1.06 !important; }
      .header-sub { font-size: 13.5px !important; }
      .story-title { font-size: 22px !important; line-height: 1.18 !important; }
      .story-img { max-height: 200px !important; }
      .verdict-box { padding: 20px 16px !important; }
      .ticker-band { padding: 16px !important; display: block !important; }
      .ticker-desc { border-left: none !important; border-top: 1px solid rgba(10, 36, 150, 0.22) !important; padding-left: 0 !important; padding-top: 10px !important; margin-top: 10px !important; }
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="hero-wrap">
      <div class="top-bar"><span>DAILY DISPATCH</span><span>{{TODAY_STR}}</span></div>
      <div class="header">
        <h1 class="header-title">Morning AI &amp;<br>Culture Dispatch</h1>
        <p class="header-sub">A sharp, curated 4-minute briefing on human friction, creative craft, and algorithmic culture.</p>
      </div>
    </div>
    {{NEWSLETTER_BODY}}
    <div class="footer"><span>CURATED WITH PRECISION</span><span>{{ISSUE_MONTH}}</span></div>
  </div>
  <div style="text-align:center; padding: 14px; font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #3A50AC;">
    Don’t want these daily briefings? <a href="mailto:{{UNSUB_EMAIL}}?subject=UNSUBSCRIBE" style="color: #0A2496; text-decoration: underline;">Unsubscribe</a>
  </div>
</body>
</html>"""

    full_email_html = (
        html_template
        .replace("{{TODAY_STR}}", today_str)
        .replace("{{ISSUE_MONTH}}", issue_month_str)
        .replace("{{NEWSLETTER_BODY}}", newsletter_body_html)
        .replace("{{UNSUB_EMAIL}}", os.environ.get("GMAIL_ADDRESS", ""))
    )

    send_email_to_all(
        subject=f"Morning AI & Culture Dispatch // {today_str}",
        plain_text=newsletter_body_html,
        html_content=full_email_html
    )

except Exception as fatal_err:
    print(f"CRITICAL ERROR: {fatal_err}")
    try:
        gmail_user = os.environ.get("GMAIL_ADDRESS")
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[ALERT] Morning Dispatch Failed // {datetime.now().strftime('%Y-%m-%d')}"
        msg["From"] = f"Morning Dispatch <{gmail_user}>"
        msg["To"] = gmail_user
        msg.attach(MIMEText(f"The automated newsletter run failed with error:\n\n{fatal_err}", "plain"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=15) as server:
            server.login(gmail_user, os.environ.get("GMAIL_APP_PASSWORD"))
            server.send_message(msg)
    except Exception:
        pass
    raise fatal_err
