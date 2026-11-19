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
    ("New York Times Tech", "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml"),
    ("New York Times Arts & Culture", "https://rss.nytimes.com/services/xml/rss/nyt/Arts.xml"),
    ("New York Times Business", "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml"),
    ("MIT Technology Review", "https://www.technologyreview.com/feed/"),
    ("Wired", "https://www.wired.com/feed/rss"),
    ("Ars Technica", "https://feeds.arstechnica.com/arstechnica/index"),
    ("The Verge", "https://www.theverge.com/rss/index.xml"),
    ("The Guardian Tech", "https://www.theguardian.com/uk/technology/rss"),
    ("The Guardian Culture", "https://www.theguardian.com/culture/rss"),
    ("Financial Times Tech", "https://www.ft.com/technology?format=rss")
]

def get_available_models() -> list[str]:
    discovered = []
    excluded_keywords = (
        "embedding", "vision", "live", "translate", "audio",
        "tts", "image", "robotics", "veo", "imagen"
    )
    try:
        for m in client.models.list():
            name = m.name.replace("models/", "")
            if "gemini" in name and not any(kw in name.lower() for kw in excluded_keywords):
                discovered.append(name)
    except Exception as e:
        print(f"No se pudo listar modelos dinámicamente: {e}")

    preferred = [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-2.0-flash",
        "gemini-3.8-flash"
    ]
    ordered = [m for m in preferred if m in discovered]
    others = [m for m in discovered if m not in ordered and ("flash" in m or "pro" in m)]
    
    final_list = ordered + others
    return final_list if final_list else ["gemini-2.5-flash", "gemini-2.5-pro"]

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
            with urllib.request.urlopen(req, timeout=8) as resp:
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

                if not is_atom:
                    title = clean_xml_text(item.findtext("title", ""))
                    link = (item.findtext("link") or "").strip()
                    desc = clean_xml_text(item.findtext("description", ""))
                    pub_date = (item.findtext("pubDate") or "").strip()
                else:
                    title = clean_xml_text(item.findtext("{http://www.w3.org/2005/Atom}title", ""))
                    link_el = item.find("{http://www.w3.org/2005/Atom}link")
                    link = (link_el.attrib.get("href", "") if link_el is not None else "").strip()
                    desc = clean_xml_text(
                        item.findtext("{http://www.w3.org/2005/Atom}summary", "")
                        or item.findtext("{http://www.w3.org/2005/Atom}content", "")
                    )
                    pub_date = (item.findtext("{http://www.w3.org/2005/Atom}published") or "").strip()

                if title and link.startswith("http"):
                    collected_items.append(
                        f"- SOURCE: {source_name} | DATE: {pub_date}\n"
                        f"  TITLE: {title}\n"
                        f"  URL: {link}\n"
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
        with urllib.request.urlopen(req, timeout=6) as resp:
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
        for attempt in range(1, 4):
            try:
                print(f"-> Calling {model_name} (attempt {attempt}/3)...")
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
                    print(f"Limit hit on {model_name}. Waiting 4s and switching model...")
                    time.sleep(4)
                    break

                if "503" in last_error or "high demand" in last_error.lower() or "disconnected" in last_error.lower():
                    wait_sec = attempt * 8
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
    readable_date = datetime.now().strftime("%B %d, %Y")

    # PASO 0: DESCARGA DE NOTICIAS REALES DESDE RSS
    print("0/3 -> Fetching live stories from tier-1 RSS feeds...")
    raw_rss_feed = fetch_rss_news(max_per_feed=5)

    # AGENTE 1: SCOUT (FILTRA Y ESTRUCTURA 6 CANDIDATOS DEL FEED RSS)
    scout_prompt = f"""
    You are a Senior Research Scout. Today is {readable_date}.
    Below is a live feed of real articles just published by top-tier outlets:

    {raw_rss_feed}

    Select 6 distinct, high-signal articles from the list above covering:
- 2 stories on AI's impact on creative craft, design, publishing, art, or media.
- 2 stories on workplace friction, corporate decisions, or regulation involving AI and human judgment.
- 2 stories on cultural shifts, consumer behavior, or social debates around modern technology.

    STRICT RULES:
- Use ONLY articles present in the feed above.
- Copy the EXACT URL and EXACT Source Name from the feed—never alter a single character of a URL.

    For each of the 6 candidates, provide:
- Candidate Number (1 to 6)
- Raw Title & Exact Publication Name
- Exact URL
- Concrete facts, figures, names, and core tension based on the summary (3 sentences).
    """

    print("1/3 -> Starting Scout step (selecting 6 candidates)...")
    scout_output = generate_resilient(scout_prompt, ACTIVE_MODELS, temp=0.2)

    # AGENTE 2: EDITOR JEFE (CURADURÍA Y ÁNGULO EDITORIAL)
    editor_prompt = f"""
    You are the Editor-in-Chief of "Morning AI & Culture Dispatch", an avant-garde minimalist daily briefing.
    Review the 6 candidate stories gathered by the Scout below:

    {scout_output}

    YOUR EDITORIAL MISSION:
1. Discard weak stories (routine product updates, repetitive tropes, or thin summaries).
2. Select the BEST 3 stories—one for each pillar:
  - 01 / CRAFT (Creativity, design, media, human skill)
  - 02 / WORK (Institutions, labor, governance, human judgment)
  - 03 / CULTURE (Behavior, psychology, society, daily life)
3. Define a unifying thread for today's edition (2 crisp sentences explaining what connects today's cultural mood).
4. For each of the 3 selected stories, define:
  - The exact Category label (CRAFT, WORK, or CULTURE)
  - A punchy lowercase headline (max 6 words)
  - The sharp editorial angle (what is the real human tension here?)
  - The verified facts, names, and numbers to preserve
  - The exact Publication Name and exact URL (must match the Scout's URLs verbatim).
    """

    print("2/3 -> Starting Editor-in-Chief step (curating top 3)...")
    editor_brief = generate_resilient(editor_prompt, ACTIVE_MODELS, temp=0.3)

    # AGENTE 3: WRITER (REDACCIÓN CLARA Y MAQUETACIÓN HTML)
    writer_prompt = f"""
    You are a sharp editorial writer. Turn the Editor-in-Chief's approved brief into a crisp, effortless morning read.

    STYLE & RIGOR RULES:
- Clarity first: Short sentences, active verbs, concrete details (names, numbers, companies) from the brief.
- Never invent facts, quotes, or URLs not present in the brief below.
- Avoid convoluted metaphors, academic jargon, and AI clichés (never use delve, tapestry, landscape, paradigm, nuanced, labyrinth).
- Keep paragraphs short (3 to 4 lines max).
- DO NOT include any <img> tags anywhere.
- DO NOT include any takeaway boxes.

    Editor-in-Chief Brief:
    {editor_brief}

    Generate ONLY valid raw HTML (no markdown code blocks, no ```html):
- Start with the unifying 2-sentence editorial intro inside <p class="intro-text">...</p>
- For each of the 3 stories, wrap in <div class="story-block">:
  - Index & Category header: <div class="story-meta">01 / CATEGORY</div> (use 01, 02, 03)
  - <h2 class="story-title">lowercase headline</h2>
  - <p class="subtitle">One plain, direct sentence summarizing the core development.</p>
  - 2 short, factual paragraphs (<p>...</p>).
  - Source link block using ONLY the exact verified URL from the brief:
        <div class="source-row"><span class="source-label">SOURCE // PUBLICATION NAME</span> <a href="EXACT_URL" class="source-link" target="_blank">READ ARTICLE &rarr;</a></div>
- End with 1 simple, grounded closing thought inside <div class="parting-thought">...</div>.
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

    if newsletter_body_html.count('class="story-block"') < 3:
        raise RuntimeError("Validación fallida: El HTML generado no contiene las 3 historias completas.")

    # PLANTILLA EDITORIAL
    html_template = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link href="[https://fonts.googleapis.com/css2?family=Jost:wght@500;600;700&display=swap](https://fonts.googleapis.com/css2?family=Jost:wght@500;600;700&display=swap)" rel="stylesheet">
  <style>
    body { margin: 0; padding: 0; background-color: #F4F4F4; font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Helvetica Neue', Helvetica, Arial, sans-serif; color: #2140D6; line-height: 1.65; -webkit-font-smoothing: antialiased; }
    .container { max-width: 620px; margin: 30px auto; background: #FFFFFF; border: 1px solid #2140D6; border-radius: 0px; }
    .top-bar { padding: 12px 30px; border-bottom: 1px solid #2140D6; font-size: 11px; font-weight: 600; letter-spacing: 1.2px; display: flex; justify-content: space-between; text-transform: uppercase; }
    .header { padding: 48px 30px 42px 30px; border-bottom: 1px solid #2140D6; }
    .header-title { font-family: 'Jost', 'Futura', 'Century Gothic', 'Trebuchet MS', sans-serif; margin: 0; font-size: 46px; font-weight: 600; line-height: 1.02; letter-spacing: -1.5px; color: #2140D6; }
    .header-sub { margin: 22px 0 0 0; font-size: 14px; font-weight: 400; max-width: 460px; line-height: 1.5; color: #2140D6; }
    .content { padding: 35px 30px; }
    .intro-text { font-size: 15px; padding-bottom: 28px; margin-bottom: 35px; border-bottom: 1px dashed #2140D6; }
    .story-block { margin-bottom: 42px; padding-bottom: 34px; border-bottom: 1px solid #2140D6; }
    .story-block:last-of-type { border-bottom: none; margin-bottom: 20px; padding-bottom: 10px; }
    .story-meta { font-size: 11px; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 8px; color: #2140D6; }
    .story-title { font-family: 'Jost', 'Futura', 'Century Gothic', 'Trebuchet MS', sans-serif; font-size: 28px; font-weight: 600; line-height: 1.12; letter-spacing: -0.6px; text-transform: lowercase; margin: 0 0 10px 0; color: #2140D6; }
    .subtitle { font-size: 15px; font-weight: 600; margin: 0 0 18px 0; line-height: 1.45; }
    p { margin: 0 0 16px 0; font-size: 15.5px; line-height: 1.65; font-weight: 400; }
    .source-row { margin-top: 14px; padding-top: 10px; display: flex; justify-content: space-between; align-items: center; font-size: 11.5px; font-weight: 700; letter-spacing: 0.8px; text-transform: uppercase; }
    .source-label { color: #2140D6; opacity: 0.8; }
    .source-link { color: #2140D6; text-decoration: underline; text-underline-offset: 3px; }
    .parting-thought { border-top: 1px dashed #2140D6; padding-top: 24px; margin-top: 20px; font-size: 14.5px; font-style: italic; }
    .footer { padding: 25px 30px; border-top: 1px solid #2140D6; font-size: 11px; font-weight: 600; display: flex; justify-content: space-between; text-transform: uppercase; letter-spacing: 1.2px; }
  </style>
</head>
<body>
  <div class="container">
    <div class="top-bar"><span>DAILY DISPATCH</span><span>{{TODAY_STR}}</span></div>
    <div class="header">
      <h1 class="header-title">Morning AI &amp;<br>Culture Dispatch</h1>
      <p class="header-sub">A sharp, curated 4-minute briefing on human friction, creative craft, and algorithmic culture.</p>
    </div>
    <div class="content">{{NEWSLETTER_BODY}}</div>
    <div class="footer"><span>CURATED WITH PRECISION</span></div>
  </div>
  <div style="text-align:center; padding: 14px; font-size: 11px; color: #2140D6; opacity: 0.7;">
    Don’t want these daily briefings? <a href="mailto:{{UNSUB_EMAIL}}?subject=UNSUBSCRIBE" style="color: #2140D6;">Unsubscribe</a>
  </div>
</body>
</html>"""

    full_email_html = (
        html_template
        .replace("{{TODAY_STR}}", today_str)
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
