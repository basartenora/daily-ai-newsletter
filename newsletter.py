import os
import re
import smtplib
import time
import urllib.request
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

def get_available_models() -> list[str]:
    discovered = []
    try:
        for m in client.models.list():
            name = m.name.replace("models/", "")
            if "gemini" in name and "embedding" not in name and "vision" not in name:
                discovered.append(name)
    except Exception as e:
        print(f"No se pudo listar modelos dinámicamente: {e}")

    preferred = ["gemini-3.8-flash", "gemini-3.1-pro-preview", "gemini-3.8-pro"]
    ordered = [m for m in preferred if m in discovered]
    others = [m for m in discovered if m not in ordered and ("3." in m or "flash" in m)]
    
    final_list = ordered + others
    return final_list if final_list else ["gemini-3.8-flash", "gemini-3.1-pro-preview"]

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

def generate_resilient(prompt: str, models: list[str], use_search: bool = False, temp: float = 0.3) -> str:
    config = types.GenerateContentConfig(
        tools=[types.Tool(google_search=types.GoogleSearch())] if use_search else None,
        temperature=temp
    )

    last_error = ""
    for model_name in models:
        for attempt in range(1, 4):
            try:
                print(f"-> Calling {model_name} (search={use_search}, attempt {attempt}/3)...")
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=config
                )
                
                if response and response.text:
                    text_output = response.text
                    
                    if use_search and response.candidates:
                        candidate = response.candidates[0]
                        if getattr(candidate, "grounding_metadata", None):
                            chunks = getattr(candidate.grounding_metadata, "grounding_chunks", []) or []
                            verified_links = []
                            for chunk in chunks:
                                web = getattr(chunk, "web", None)
                                if web and getattr(web, "uri", None):
                                    title = getattr(web, "title", "Source")
                                    verified_links.append(f"- {title}: {web.uri}")
                            
                            if verified_links:
                                text_output += "\n\nVERIFIED GROUNDING URLS:\n" + "\n".join(verified_links[:15])
                    
                    return text_output
            except Exception as e:
                last_error = str(e)
                print(f"Notice on {model_name}: {last_error}")

                if "429" in last_error or "resource_exhausted" in last_error.lower():
                    print(f"Limit hit on {model_name}. Switching to next model...")
                    time.sleep(2)
                    break

                if "503" in last_error or "high demand" in last_error.lower() or "disconnected" in last_error.lower():
                    wait_sec = attempt * 8
                    print(f"Server busy. Waiting {wait_sec}s...")
                    time.sleep(wait_sec)
                else:
                    break

    raise RuntimeError(f"Todos los modelos fallaron. Último error de Google: {last_error}")

def send_email(subject: str, plain_text: str, html_content: str):
    gmail_user = os.environ.get("GMAIL_ADDRESS")
    gmail_password = os.environ.get("GMAIL_APP_PASSWORD")

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"Morning Dispatch <{gmail_user}>"
    msg["To"] = gmail_user

    msg.attach(MIMEText(plain_text, "plain"))
    msg.attach(MIMEText(html_content, "html"))

    print("Sending email to Gmail...")
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=15) as server:
        server.login(gmail_user, gmail_password)
        server.send_message(msg)
    print("Dispatch completed and sent successfully!")

# EJECUCIÓN PRINCIPAL CON CAPTURA DE ERRORES
try:
    ACTIVE_MODELS = get_available_models()
    print(f"Detected active models: {ACTIVE_MODELS}")
    today_str = datetime.now().strftime("%Y // %m.%d").upper()
    readable_date = datetime.now().strftime("%B %d, %Y")

    # AGENTE 1: SCOUT
    scout_prompt = f"""
    You are a Senior Research Scout. Today is {readable_date}.
    Search the web and gather 6 real, distinct news stories or cultural essays published in the last 72 hours across these themes:
    - 2 stories on AI's impact on creative craft, design, publishing, art, or media.
    - 2 stories on workplace friction, corporate decisions, or regulation involving AI and human judgment.
    - 2 stories on cultural shifts, consumer behavior, or social debates around modern technology.

    STRICT SOURCE RULES:
    - Prioritize top-tier journalism: Financial Times, The New York Times, The Atlantic, Wired, MIT Technology Review, The Economist, Ars Technica, Bloomberg, Nature, The Verge, The Guardian.
    - Reject vendor press releases, self-promotional AI announcements, and listicles.

    For each of the 6 candidates, provide:
    - Candidate Number (1 to 6)
    - Raw Title & Publication Name
    - Exact URL from the search results
    - Concrete facts, figures, names, and core tension (3-4 sentences).
    """

    print("1/3 -> Starting Scout step (gathering 6 stories)...")
    scout_output = generate_resilient(scout_prompt, ACTIVE_MODELS, use_search=True, temp=0.2)

    # AGENTE 2: EDITOR JEFE
    editor_prompt = f"""
    You are the Editor-in-Chief of "Morning AI & Culture Dispatch", an avant-garde minimalist daily briefing.
    Review the 6 candidate stories gathered by the Scout below:

    {scout_output}

    YOUR EDITORIAL MISSION:
    1. Discard weak stories (corporate PR disguised as news, repetitive tropes, or thin reporting).
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
       - The exact Publication Name and exact URL (must match the Scout's verified URLs).
    """

    print("2/3 -> Starting Editor-in-Chief step (curating top 3)...")
    editor_brief = generate_resilient(editor_prompt, ACTIVE_MODELS, use_search=False, temp=0.3)

    # AGENTE 3: WRITER
    writer_prompt = f"""
    You are a sharp editorial writer. Turn the Editor-in-Chief's approved brief into a crisp, effortless morning read.

    STYLE & RIGOR RULES:
    - Clarity first: Short sentences, active verbs, concrete details (names, numbers, companies) from the brief.
    - Never invent facts, quotes, or URLs not present in the brief below.
    - Avoid convoluted metaphors, academic jargon, and AI clichés (never use delve, tapestry, landscape, paradigm, nuanced, labyrinth).
    - Keep paragraphs short (3 to 4 lines max).
    - DO NOT include any <img> tags anywhere.

    Editor-in-Chief Brief:
    {editor_brief}

    Generate ONLY valid raw HTML (no markdown code blocks, no ```html):
    - Start with the unifying 2-sentence editorial intro inside <p class="intro-text">...</p>
    - For each of the 3 stories, wrap in <div class="story-block">:
      * Index & Category header: <div class="story-meta">01 / CATEGORY</div> (use 01, 02, 03)
      * <h2 class="story-title">lowercase headline</h2>
      * <p class="subtitle">One plain, direct sentence summarizing the core development.</p>
      * 2 short, factual paragraphs (<p>...</p>).
      * Takeaway box: <div class="takeaway-box"><span class="takeaway-label">TAKEAWAY //</span> One clear, memorable sentence.</div>
      * Source link block using ONLY the exact verified URL from the brief:
        <div class="source-row"><span class="source-label">SOURCE // PUBLICATION NAME</span> <a href="EXACT_URL" class="source-link" target="_blank">READ ARTICLE &rarr;</a></div>
    - End with 1 simple, grounded closing thought inside <div class="parting-thought">...</div>.
    """

    print("3/3 -> Starting Writer step (drafting HTML)...")
    writer_output = generate_resilient(writer_prompt, ACTIVE_MODELS, use_search=False, temp=0.4)
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
    .header-year { font-family: 'Jost', 'Futura', 'Century Gothic', 'Trebuchet MS', sans-serif; margin: 6px 0 0 0; font-size: 42px; font-weight: 500; line-height: 1.0; letter-spacing: -1px; color: #2140D6; }
    .header-sub { margin: 22px 0 0 0; font-size: 14px; font-weight: 400; max-width: 460px; line-height: 1.5; color: #2140D6; }
    .content { padding: 35px 30px; }
    .intro-text { font-size: 15px; padding-bottom: 28px; margin-bottom: 35px; border-bottom: 1px dashed #2140D6; }
    .story-block { margin-bottom: 42px; padding-bottom: 34px; border-bottom: 1px solid #2140D6; }
    .story-block:last-of-type { border-bottom: none; margin-bottom: 20px; padding-bottom: 10px; }
    .story-meta { font-size: 11px; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 8px; color: #2140D6; }
    .story-title { font-family: 'Jost', 'Futura', 'Century Gothic', 'Trebuchet MS', sans-serif; font-size: 28px; font-weight: 600; line-height: 1.12; letter-spacing: -0.6px; text-transform: lowercase; margin: 0 0 10px 0; color: #2140D6; }
    .subtitle { font-size: 15px; font-weight: 600; margin: 0 0 18px 0; line-height: 1.45; }
    p { margin: 0 0 16px 0; font-size: 15.5px; line-height: 1.65; font-weight: 400; }
    .takeaway-box { border: 1px solid #2140D6; padding: 15px 18px; margin-top: 20px; font-size: 14.5px; line-height: 1.55; background-color: #F8F9FF; }
    .takeaway-label { font-weight: 800; font-size: 12px; letter-spacing: 0.8px; margin-right: 4px; }
    .source-row { margin-top: 14px; padding-top: 10px; display: flex; justify-content: space-between; align-items: center; font-size: 11.5px; font-weight: 700; letter-spacing: 0.8px; text-transform: uppercase; }
    .source-label { color: #2140D6; opacity: 0.8; }
    .source-link { color: #2140D6; text-decoration: underline; text-underline-offset: 3px; }
    .parting-thought { border-top: 1px dashed #2140D6; padding-top: 24px; margin-top: 20px; font-size: 14.5px; font-style: italic; }
    .footer { padding: 25px 30px; border-top: 1px solid #2140D6; font-size: 11px; font-weight: 600; display: flex; justify-content: space-between; text-transform: uppercase; letter-spacing: 1.2px; }
  </style>
</head>
<body>
  <div class="container">
    <div class="top-bar"><span>DISPATCH // ISSUE DAILY</span><span>{{TODAY_STR}}</span></div>
    <div class="header">
      <h1 class="header-title">Morning AI &amp;<br>Culture Dispatch</h1>
      <div class="header-year">2026</div>
      <p class="header-sub">A sharp, curated 4-minute briefing on human friction, creative craft, and algorithmic culture.</p>
    </div>
    <div class="content">{{NEWSLETTER_BODY}}</div>
    <div class="footer"><span>CURATED WITH PRECISION</span><span>END OF DISPATCH &rarr;</span></div>
  </div>
</body>
</html>"""

    full_email_html = html_template.replace("{{TODAY_STR}}", today_str).replace("{{NEWSLETTER_BODY}}", newsletter_body_html)

    send_email(
        subject=f"Morning AI & Culture Dispatch // {today_str}",
        plain_text=newsletter_body_html,
        html_content=full_email_html
    )

except Exception as fatal_err:
    print(f"CRITICAL ERROR: {fatal_err}")
    try:
        send_email(
            subject=f"[ALERT] Morning Dispatch Failed // {datetime.now().strftime('%Y-%m-%d')}",
            plain_text=f"The automated newsletter run failed with error:\n\n{fatal_err}",
            html_content=f"<p><strong>Automated Dispatch Error:</strong></p><pre>{fatal_err}</pre>"
        )
    except Exception:
        pass
    raise fatal_err
