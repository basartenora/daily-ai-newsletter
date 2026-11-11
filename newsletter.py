import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from crewai import Agent, Crew, LLM, Process, Task
from crewai_tools import SerperDevTool
import markdown

# 1. CONFIGURACIÓN DEL MODELO GEMINI
gemini_llm = LLM(
    model="gemini/gemini-3.8-flash", api_key=os.environ.get("GEMINI_API_KEY")
)

search_tool = SerperDevTool()

# 2. AGENTES (Se les asigna el modelo Gemini explícitamente)
researcher = Agent(
    role="Culture & AI Scout",
    goal="Find 3-4 compelling stories on AI shifts, cultural friction, and human psychology with direct links.",
    backstory=(
        "You are an Intellectual Scout and Daily Trends Curator. "
        "You identify human-interest AI developments and cultural shifts rather than dry PR releases. "
        "You always verify links and extract direct quotes and core takeaways."
    ),
    tools=[search_tool],
    llm=gemini_llm,
    verbose=True,
)

writer = Agent(
    role="Essayist & Newsletter Stylist",
    goal="Transform curated research into a punchy, 4-minute morning dispatch with embedded links.",
    backstory=(
        "You write in the style of The New York Times culture desk and top Substack essayists. "
        "Your tone is witty, observant, and engaging. You ban AI buzzwords like 'delve' or 'game-changer', "
        "and format every item with bold hooks, bulleted takeaways, and links to original sources."
    ),
    llm=gemini_llm,
    verbose=True,
)

# 3. TAREAS
research_task = Task(
    description=(
        "Scout today's web for 3-4 stories: 2-3 around AI's impact on work, writing, and society, "
        "and 1 cultural curiosity or psychology piece. Output the core angle, key quote, and source URL for each."
    ),
    expected_output="Structured research brief with verified links and quotes.",
    agent=researcher,
)

write_task = Task(
    description=(
        "Transform the research brief into a 400-700 word morning dispatch formatted in clean Markdown, "
        "starting with an observational hook, followed by 3-4 numbered bites with takeaways and markdown links, "
        "ending with a parting thought."
    ),
    expected_output="Polished Markdown newsletter ready for publication.",
    agent=writer,
)

# 4. CREW
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.sequential,
    verbose=True,
)

result = crew.kickoff()
content_str = str(result)

# 5. ENVÍO POR GMAIL
gmail_user = os.environ.get("GMAIL_ADDRESS")
gmail_password = os.environ.get("GMAIL_APP_PASSWORD")

html_body = f"""
<html>
  <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #1a1a1a; max-width: 650px; margin: 0 auto; padding: 20px;">
    {markdown.markdown(content_str)}
  </body>
</html>
"""

msg = MIMEMultipart("alternative")
msg["Subject"] = "☕ Your Morning AI & Culture Dispatch"
msg["From"] = gmail_user
msg["To"] = gmail_user
msg.attach(MIMEText(content_str, "plain"))
msg.attach(MIMEText(html_body, "html"))

with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
  server.login(gmail_user, gmail_password)
  server.send_message(msg)

print("Dispatch sent successfully!")
