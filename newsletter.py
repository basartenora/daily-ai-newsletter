import os
import smtplib
import time
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from google import genai
from google.genai import types

# 1. CLIENTE CONFIGURADO EN V1BETA (SOPORTA MODELOS ESTABLES Y PREVIEW)
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(api_version="v1beta")
)

def get_available_models() -> list[str]:
    """Descubre dinámicamente qué modelos activos tiene tu API Key."""
    discovered = []
    try:
        for m in client.models.list():
            name = m.name.replace("models/", "")
            if "gemini" in name and "embedding" not in name and "vision" not in name:
                discovered.append(name)
    except Exception as e:
        print(f"No se pudo listar modelos dinámicamente: {e}")

    # Priorizamos gemini-3.8-flash y cualquier variante gemini-3 disponible
    preferred = ["gemini-3.8-flash", "gemini-3.1-pro-preview", "gemini-3.8-pro"]
    ordered = [m for m in preferred if m in discovered]
    others = [m for m in discovered if m not in ordered and ("3." in m or "flash" in m)]
    
    final_list = ordered + others
    return final_list if final_list else ["gemini-3.8-flash", "gemini-3.1-pro-preview"]

def generate_resilient(prompt: str, models: list[str]) -> str:
    """Prueba los modelos disponibles con backoff progresivo para absorber picos 503."""
    for model_name in models:
        for attempt in range(1, 4):
            try:
                print(f"-> Calling {model_name} (attempt {attempt}/3)...")
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
                if response and response.text:
                    return response.text
            except Exception as e:
                err_msg = str(e)
                print(f"Notice on {model_name}: {err_msg}")

                if "429" in err_msg or "resource_exhausted" in err_msg.lower():
                    raise RuntimeError("Cuota diaria agotada (429). Espera al reseteo o usa otra clave.")

                if "503" in err_msg or "high demand" in err_msg.lower() or "disconnected" in err_msg.lower():
                    wait_sec = attempt * 8  # 8s, 16s, 24s
                    print(f"Server busy. Waiting {wait_sec}s...")
                    time.sleep(wait_sec)
                else:
                    break

    raise RuntimeError("Todos los modelos disponibles estuvieron saturados tras varios reintentos.")

ACTIVE_MODELS = get_available_models()
print(f"Detected active models for this key: {ACTIVE_MODELS}")

today_str = datetime.now().strftime("%A, %B %d, %Y")

# 2. AGENTE 1: CULTURE & AI SCOUT
scout_prompt = f"""
You are the Culture & AI Scout for a prestigious daily dispatch. Today is {today_str}.
Suggest 3 distinct cultural dilemmas emerging from AI adoption:
1. Story 1: Impact on human creative craft, writing, or habits.
2. Story 2: Workplace or institutional tension where human judgment meets automation.
3. Story 3: Modern digital habit or psychological shift.

For each, provide:
- Headline
- One-sentence subtitle on the core paradox
- Core tension in 2 sentences
- The takeaway insight in 1 sentence
"""

print("Starting Scout step...")
scout_output = generate_resilient(scout_prompt, ACTIVE_MODELS)

# 3. AGENTE 2: ESSAYIST & EDITORIAL DESIGNER
writer_prompt = f"""
You are an Essayist & Editorial Designer writing in the style of The New York Times culture desk and high-end Substack essays.
Tone: Witty, perceptive, concise. No corporate AI buzzwords (never use delve, tapestry, landscape).

Based on these notes:
{scout_output}

Generate ONLY valid raw HTML (no markdown code blocks, no ```html):
- A 2-sentence opening hook on today's cultural current.
- For each of the 3 stories:
  * <span class="pill">CATEGORY</span>
  * <h2>Headline</h2>
  * <p class="subtitle">Subtitle</p>
  * An editorial image:
    - Story 1: <img src="[https://images.unsplash.com/photo-1518770660439-4636190af475?auto=format&fit=crop&w=800&q=80](https://images.unsplash.com/photo-1518770660439-4636190af475?auto=format&fit=crop&w=800&q=80)" class="story-img" alt="Visual" />
    - Story 2: <img src="[https://images.unsplash.com/photo-1507679799987-c73779587ccf?auto=format&fit=crop&w=800&q=80](https://images.unsplash.com/photo-1507679799987-c73779587ccf?auto=format&fit=crop&w=800&q=80)" class="story-img" alt="Visual" />
    - Story 3: <img src="[https://images.unsplash.com/photo-1451187580459-43490279c0fa?auto=format&fit=crop&w=800&q=80](https://images.unsplash.com/photo-1451187580459-43490279c0fa?auto=format&fit=crop&w=800&q=80)" class="story-img" alt="Visual" />
  * 2 tight narrative paragraphs.
  * <div class="takeaway-box"><strong>The takeaway:</strong> Core insight.</div>
- A brief parting thought.
"""

print("Starting Writer step...")
writer_output = generate_resilient(writer_prompt, ACTIVE_MODELS)
newsletter_body_html = writer_output.replace("```html", "").replace("```", "").strip()

# 4. PLANTILLA HTML CON PALETA EDITORIAL
html_template = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    body {
      margin: 0;
      padding: 0;
      background-color: #FEFFE3;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
      color: #255A4B;
      line-height: 1.65;
    }
    .container {
      max-width: 620px;
      margin: 30px auto;
      background: #FFFFFF;
      border-radius: 18px;
      overflow: hidden;
      box-shadow: 0 4px 20px rgba(37, 90, 75, 0.06);
      border: 1px solid #EAEFD2;
    }
    .header {
      background: #255A4B;
      padding: 40px 30px;
      text-align: center;
      color: #FEFFE3;
    }
    .header-tag {
      display: inline-block;
      background-color: #C9FF46;
      color: #255A4B;
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 1.5px;
      padding: 5px 12px;
      border-radius: 20px;
      margin-bottom: 12px;
    }
    .header h1 {
      margin: 0 0 8px 0;
      font-size: 26px;
      font-weight: 800;
      letter-spacing: -0.5px;
      color: #FEFFE3;
    }
    .header p {
      margin: 0;
      font-size: 14px;
      color: #A6FFE8;
    }
    .content {
      padding: 35px 30px;
    }
    .pill {
      display: inline-block;
      background-color: #A6FFE8;
      color: #255A4B;
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 1px;
      padding: 4px 10px;
      border-radius: 6px;
      margin-bottom: 10px;
    }
    h2 {
      color: #255A4B;
      font-size: 21px;
      margin: 8px 0 4px 0;
      line-height: 1.3;
      font-weight: 800;
    }
    .subtitle {
      color: #4456FF;
      font-size: 14px;
      font-weight: 600;
      margin: 0 0 16px 0;
    }
    .story-img {
      width: 100%;
      height: 240px;
      object-fit: cover;
      border-radius: 12px;
      margin: 12px 0 18px 0;
      display: block;
    }
    p {
      margin: 0 0 16px 0;
      font-size: 15px;
    }
    .takeaway-box {
      background-color: #F6FAED;
      border-left: 4px solid #34C38F;
      padding: 12px 16px;
      border-radius: 0 8px 8px 0;
      margin: 20px 0 35px 0;
      font-size: 14px;
    }
    .takeaway-box strong {
      color: #255A4B;
    }
    .footer {
      background-color: #FEFFE3;
      padding: 25px;
      text-align: center;
      font-size: 13px;
      color: #255A4B;
      border-top: 1px solid #EAEFD2;
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <span class="header-tag">Daily Briefing</span>
      <h1>Your Morning AI &amp; Culture Dispatch</h1>
      <p>{{TODAY_STR}} &bull; A 4-minute curated read</p>
    </div>
    <div class="content">
      {{NEWSLETTER_BODY}}
    </div>
    <div class="footer">
      <p style="margin: 0;">Curated daily with curious precision. Have a great day.</p>
    </div>
  </div>
</body>
</html>
"""

full_email_html = html_template.replace("{{TODAY_STR}}", today_str).replace("{{NEWSLETTER_BODY}}", newsletter_body_html)

# 5. ENVÍO POR GMAIL SMTP
gmail_user = os.environ.get("GMAIL_ADDRESS")
gmail_password = os.environ.get("GMAIL_APP_PASSWORD")

msg = MIMEMultipart("alternative")
msg["Subject"] = f"☕ Your Morning AI & Culture Dispatch — {today_str}"
msg["From"] = f"Your Morning Dispatch <{gmail_user}>"
msg["To"] = gmail_user

msg.attach(MIMEText(newsletter_body_html, "plain"))
msg.attach(MIMEText(full_email_html, "html"))

print("Sending email to Gmail...")
with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=15) as server:
    server.login(gmail_user, gmail_password)
    server.send_message(msg)

print("Dispatch completed and sent successfully!")
