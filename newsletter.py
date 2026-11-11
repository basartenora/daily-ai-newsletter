import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from crewai import Agent, Crew, LLM, Process, Task

# 1. MODELO GEMINI (ÚNICA API NECESARIA)
gemini_llm = LLM(
    model="gemini/gemini-3.8-flash",
    api_key=os.environ.get("GEMINI_API_KEY")
)

# 2. AGENTES SIN HERRAMIENTAS EXTERNAS (RÁPIDOS Y SIN DEPENDENCIAS)
scout = Agent(
    role="Culture & AI Scout",
    goal="Identify 3 provocative cultural dilemmas and human-interest tensions emerging from AI adoption.",
    backstory=(
        "You are an intellectual trend spotter and cultural curator. "
        "You observe how AI is reshaping writing, human craft, psychology, workplace culture, and modern society. "
        "You formulate compelling story premises focused on irony, human friction, and ethical nuance."
    ),
    llm=gemini_llm,
    max_iter=1,
    verbose=True,
)

writer = Agent(
    role="Essayist & Editorial Designer",
    goal="Transform the cultural premises into a stylish, punchy morning newsletter formatted in clean HTML.",
    backstory=(
        "You write in the sharp, observant style of The New York Times culture desk and high-end Substack essayists. "
        "Your tone is witty, perceptive, and sophisticated. You ban corporate AI buzzwords (delve, revolutionize, landscape). "
        "You craft memorable titles, clear single-sentence subtitles, narrative paragraphs, and crisp takeaway boxes."
    ),
    llm=gemini_llm,
    max_iter=1,
    verbose=True,
)

# 3. TAREAS
scout_task = Task(
    description=(
        "Curate 3 distinct story concepts examining modern life and AI:\n"
        "1. Story 1: AI's subtle psychological or behavioral impact on human creativity and personal work.\n"
        "2. Story 2: A workplace or institutional tension where human judgment clashes with automation.\n"
        "3. Story 3: A cultural curiosity, modern digital habit, or intellectual shift.\n\n"
        "For each story, provide:\n"
        "- Provocative title and angle\n"
        "- The core paradox or human conflict\n"
        "- A curated royalty-free visual theme keyword (e.g., 'minimalist architecture', 'human hands typing', 'abstract geometric light')\n"
        "- A memorable, punchy takeaway insight"
    ),
    expected_output="Curated briefing with 3 distinct cultural tensions and visual themes.",
    agent=scout,
)

write_task = Task(
    description=(
        "Write today's edition of 'Your Morning AI & Culture Dispatch' in clean, raw HTML (DO NOT wrap in ```html markdown).\n"
        "Structure requirements:\n"
        "- An observational 2-3 sentence introductory hook reflecting on today's cultural current.\n"
        "- 3 Stories, each formatted with:\n"
        "  * Category pill: <span class='pill'>Category Name</span> (e.g., 'Mind & Craft', 'Digital Ecology', 'Culture')\n"
        "  * <h2>Sharp, Catchy Headline</h2>\n"
        "  * <p class='subtitle'>One elegant sentence summarizing the paradox.</p>\n"
        "  * An image using Unsplash Source matching the visual theme keyword: <img src='[https://images.unsplash.com/photo-](https://images.unsplash.com/photo-)[relevant-id]?auto=format&fit=crop&w=800&q=80' class='story-img' alt='Editorial Visual' /> (Use curated high-quality Unsplash image URLs: '[https://images.unsplash.com/photo-1518770660439-4636190af475?auto=format&fit=crop&w=800&q=80](https://images.unsplash.com/photo-1518770660439-4636190af475?auto=format&fit=crop&w=800&q=80)', '[https://images.unsplash.com/photo-1507679799987-c73779587ccf?auto=format&fit=crop&w=800&q=80](https://images.unsplash.com/photo-1507679799987-c73779587ccf?auto=format&fit=crop&w=800&q=80)', '[https://images.unsplash.com/photo-1451187580459-43490279c0fa?auto=format&fit=crop&w=800&q=80](https://images.unsplash.com/photo-1451187580459-43490279c0fa?auto=format&fit=crop&w=800&q=80)').\n"
        "  * 2 rich, engaging narrative paragraphs breaking down the idea.\n"
        "  * Takeaway box: <div class='takeaway-box'><strong>The takeaway:</strong> The core insight in one sentence.</div>\n"
        "- A closing parting thought to start the morning."
    ),
    expected_output="Fully styled HTML newsletter ready for email delivery.",
    agent=writer,
)

# 4. EJECUCIÓN DEL PIPELINE
crew = Crew(
    agents=[scout, writer],
    tasks=[scout_task, write_task],
    process=Process.sequential,
    verbose=True,
)

result = crew.kickoff()
newsletter_body_html = str(result).replace("```html", "").replace("```", "").strip()

# 5. PLANTILLA EDITORIAL CON TU PALETA DE COLOR
today_str = datetime.now().strftime("%A, %B %d, %Y")

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

# 6. ENVÍO A GMAIL
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
