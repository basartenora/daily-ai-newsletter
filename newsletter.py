import os
import smtplib
import time
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from google import genai
from google.genai.errors import ServerError

# 1. CLIENTE OFICIAL DE GOOGLE GEMINI
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

PRIMARY_MODEL = "gemini-3.8-flash"
FALLBACK_MODELS = ["gemini-3-flash", "gemini-flash-latest"]

def generate_with_retry(prompt: str, max_retries: int = 3) -> str:
    """Envía un prompt a Gemini manejando saturación temporal (503) y fallbacks."""
    models_to_try = [PRIMARY_MODEL] + FALLBACK_MODELS

    for current_model in models_to_try:
        for attempt in range(1, max_retries + 1):
            try:
                print(f"-> Calling {current_model} (attempt {attempt}/{max_retries})...")
                response = client.models.generate_content(
                    model=current_model,
                    contents=prompt
                )
                return response.text
            except ServerError as e:
                # Error 503 / High demand: esperar antes de reintentar
                if "503" in str(e) or "high demand" in str(e).lower():
                    wait_time = attempt * 4
                    print(f"High demand on {current_model}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise e
            except Exception as e:
                print(f"Error on {current_model}: {e}")
                break

    raise RuntimeError("All models and retries were exhausted.")

today_str = datetime.now().strftime("%A, %B %d, %Y")

# 2. AGENTE 1: CULTURE & AI SCOUT
scout_prompt = f"""
You are the "Culture & AI Scout", an intellectual trend spotter and cultural curator for a prestigious daily dispatch.
Today is {today_str}.

Curate 3 distinct story concepts examining modern life and AI:
1. Story 1: AI's subtle psychological or behavioral impact on human creativity and personal craft.
2. Story 2: A workplace or institutional tension where human judgment clashes with automation.
3. Story 3: A cultural curiosity, modern digital habit, or intellectual shift.

For each story, provide:
- Provocative title and angle
- The core paradox or human conflict
- Visual theme keyword (e.g., 'writer at dusk', 'brutalist office', 'smartphones in darkness')
- A memorable, punchy takeaway insight
"""

print("Starting Agent 1 (Culture & AI Scout)...")
scout_output = generate_with_retry(scout_prompt)

# 3. AGENTE 2: ESSAYIST & EDITORIAL DESIGNER
writer_prompt = f"""
You are an "Essayist & Editorial Designer" writing in the style of The New York Times culture desk and high-end Substack essayists.
Tone: Witty, perceptive, concise, and elegant. Absolutely avoid corporate AI buzzwords (delve, revolutionize, tapestry, landscape).

Based on the research below, write today's newsletter body.

RESEARCH BRIEF:
{scout_output}

STRICT HTML REQUIREMENTS:
- Output ONLY valid raw HTML (DO NOT wrap in ```html markdown fences).
- Start with an observational 2-3 sentence introductory hook introducing today's theme.
- For each of the 3 stories, format as follows:
  * <span class="pill">Category Name</span>
  * <h2>Sharp Headline</h2>
  * <p class="subtitle">One elegant sentence explaining the core paradox.</p>
  * An editorial image:
    - Story 1: <img src="[https://images.unsplash.com/photo-1518770660439-4636190af475?auto=format&fit=crop&w=800&q=80](https://images.unsplash.com/photo-1518770660439-4636190af475?auto=format&fit=crop&w=800&q=80)" class="story-img" alt="Editorial visual" />
    - Story 2: <img src="[https://images.unsplash.com/photo-1507679799987-c73779587ccf?auto=format&fit=crop&w=800&q=80](https://images.unsplash.com/photo-1507679799987-c73779587ccf?auto=format&fit=crop&w=800&q=80)" class="story-img" alt="Editorial visual" />
    - Story 3: <img src="[https://images.unsplash.com/photo-1451187580459-43490279c0fa?auto=format&fit=crop&w=800&q=80](https://images.unsplash.com/photo-1451187580459-43490279c0fa?auto=format&fit=crop&w=800&q=80)" class="story-img" alt="Editorial visual" />
  * 2 tight narrative paragraphs exploring the cultural tension.
  * <div class="takeaway-box"><strong>The takeaway:</strong> The core insight in one sentence.</div>
- End with a brief, reflective parting thought.
"""

print("Starting Agent 2 (Essayist & Editorial Designer)...")
writer_output = generate_with_retry(writer_prompt)

newsletter_body_html = writer_output.replace("```html", "").replace("```", "").strip()

# 4. PLANTILLA DE EMAIL CON TU PALETA EDITORIAL
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

# 5. ENVÍO A GMAIL
gmail_user = os.environ.get("GMAIL_ADDRESS")
gmail_password = os.environ.get("GMAIL_APP_PASSWORD")

msg = MIMEMultipart("alternative")
msg["Subject"] = f"☕ Your Morning AI & Culture Dispatch — {today_str}"
msg["From"] = f"Your Morning Dispatch <{gmail_user}>"
msg["To"] = gmail_user

msg.attach(MIMEText(newsletter_body_html, "plain"))
msg.attach(MIMEText(full_email_html, "html"))

with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
    server.login(gmail_user, gmail_password)
    server.send_message(msg)

print("Dispatch sent successfully!")
