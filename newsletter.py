import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime
from crewai import Agent, Crew, LLM, Process, Task
from crewai_tools import SerperDevTool

# 1. MODELO Y HERRAMIENTAS
gemini_llm = LLM(
    model="gemini/gemini-3.8-flash",
    api_key=os.environ.get("GEMINI_API_KEY")
)

search_tool = SerperDevTool()

# 2. AGENTES CON INSTRUCCIONES DE IMAGEN Y FORMATO EDITORIAL
researcher = Agent(
    role="Culture & AI Scout",
    goal="Find 3 compelling stories on AI shifts, cultural friction, and human psychology with direct links and image URLs.",
    backstory=(
        "You are an Intellectual Scout and Daily Trends Curator. "
        "You scout human-interest AI developments and cultural friction pieces from top publications "
        "(Substack, The Atlantic, NYT, Wired). For each story, you collect the core idea, a memorable quote, "
        "the verified original URL, and an image URL from the article/source when available."
    ),
    tools=[search_tool],
    llm=gemini_llm,
    verbose=True,
)

writer = Agent(
    role="Essayist & Editorial Designer",
    goal="Craft an elegant, engaging morning dispatch formatted in clean HTML ready for an email client.",
    backstory=(
        "You write in the sharp, observant style of The New York Times culture desk and high-end Substacks. "
        "Your voice is witty and conversational. You craft clear titles, brief subtitles, and format "
        "each story with an introductory image, body paragraphs, highlighted takeaways, and clean link buttons."
    ),
    llm=gemini_llm,
    verbose=True,
)

# 3. TAREAS
research_task = Task(
    description=(
        "Scout today's web for 3 compelling stories: 2 around AI's impact on work, thinking, and society, "
        "and 1 about modern culture, psychology, or everyday habits. "
        "For each story, output:\n"
        "- Title & Provocative Angle\n"
        "- Image URL (from the article or a high-quality relevant source)\n"
        "- The 'So What?': 2-3 sentences on the core paradox\n"
        "- Standout Quote\n"
        "- Source URL"
    ),
    expected_output="Structured research brief with verified links, quotes, and image URLs.",
    agent=researcher,
)

write_task = Task(
    description=(
        "Write today's edition of 'Your Morning AI & Culture Dispatch'.\n"
        "Generate raw, clean HTML content (DO NOT wrap in markdown code blocks like ```html).\n"
        "Format requirements:\n"
        "- Elegant header with issue date and subtitle.\n"
        "- The Morning Kick: A 2-3 sentence observational hook on today's cultural current.\n"
        "- 3 Stories: Each story MUST have:\n"
        "  1. A category pill (e.g., 'AI & Society', 'Mind & Craft', 'Culture').\n"
        "  2. A sharp Title (H2) and a brief 1-sentence Subtitle.\n"
        "  3. If an image URL is provided in the research, an <img> tag with full width and rounded corners.\n"
        "  4. 2 engaging narrative paragraphs breaking down the tension or paradox.\n"
        "  5. A prominent Takeaway box.\n"
        "  6. A clean, styled 'Read on [Outlet] →' button/link.\n"
        "- Parting Thought at the bottom."
    ),
    expected_output="Clean HTML body ready for email embedding.",
    agent=writer,
)

# 4. EJECUCIÓN
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.sequential,
    verbose=True,
)

result = crew.kickoff()
newsletter_body_html = str(result).replace("```html", "").replace("```", "").strip()

# 5. PLANTILLA DE EMAIL CON TU PALETA DE COLORES
today_str = datetime.now().strftime("%A, %B %d, %Y")

full_email_html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    body {{
      margin: 0;
      padding: 0;
      background-color: #FEFFE3;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
      color: #255A4B;
      line-height: 1.65;
    }}
    .container {{
      max-width: 620px;
      margin: 30px auto;
      background: #FFFFFF;
      border-radius: 18px;
      overflow: hidden;
      box-shadow: 0 4px 20px rgba(37, 90, 75, 0.06);
      border: 1px solid #EAEFD2;
    }}
    .header {{
      background: #255A4B;
      padding: 40px 30px;
      text-align: center;
      color: #FEFFE3;
    }}
    .header-tag {{
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
    }}
    .header h1 {{
      margin: 0 0 8px 0;
      font-size: 26px;
      font-weight: 800;
      letter-spacing: -0.5px;
      color: #FEFFE3;
    }}
    .header p {{
      margin: 0;
      font-size: 14px;
      color: #A6FFE8;
    }}
    .content {{
      padding: 35px 30px;
    }}
    .intro {{
      font-size: 16px;
      line-height: 1.7;
      color: #255A4B;
      padding-bottom: 25px;
      border-bottom: 2px solid #FEFFE3;
      margin-bottom: 30px;
    }}
    .pill {{
      display: inline-block;
      background-color: #A6FFE8;
      color: #255A4B;
      font-size: 1
