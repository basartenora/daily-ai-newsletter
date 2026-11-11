import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from crewai import Agent, Crew, LLM, Process, Task
from crewai_tools import SerperDevTool

# 1. CONFIGURACIÓN DEL MODELO Y HERRAMIENTA DE BÚSQUEDA
gemini_llm = LLM(
    model="gemini/gemini-3.8-flash",
    api_key=os.environ.get("GEMINI_API_KEY")
)

search_tool = SerperDevTool()

# 2. AGENTES OPTIMIZADOS PARA VELOCIDAD Y ENFOQUE
researcher = Agent(
    role="Culture & AI Scout",
    goal="Find 3 compelling stories on AI shifts, cultural friction, and human psychology with direct links and image URLs.",
    backstory=(
        "You are an Intellectual Scout and Daily Trends Curator. "
        "You scout human-interest AI developments and cultural friction pieces from top publications "
        "(Substack, The Atlantic, NYT, Wired). For each story, you extract the core paradox, a memorable quote, "
        "the verified original URL, and an image URL when available."
    ),
    tools=[search_tool],
    llm=gemini_llm,
    max_iter=2,             # Evita bucles infinitos de búsqueda
    max_execution_time=90,  # Límite máximo de 90 segundos
    verbose=True,
)

writer = Agent(
    role="Essayist & Editorial Designer",
    goal="Craft an elegant morning dispatch formatted in clean HTML ready for an email client.",
    backstory=(
        "You write in the sharp, observant style of The New York Times culture desk and high-end Substacks. "
        "Your voice is witty and conversational. You craft clear titles, brief subtitles, and format "
        "each story with an image (if available), engaging paragraphs, highlighted takeaways, and clean link buttons."
    ),
    llm=gemini_llm,
    max_iter=2,
    max_execution_time=90,
    verbose=True,
)

# 3. TAREAS PRECISAS (UNA SOLA BÚSQUEDA WEB)
research_task = Task(
    description=(
        "Run ONE search query using the tool: 'site:theatlantic.com OR site:wired.com OR site:nytimes.com artificial intelligence culture essay'. "
        "From the first results, choose the 3 most provocative stories. "
        "For each story, output:\n"
        "- Title & Provocative Angle\n"
        "- Image URL (from the article or a reliable web image URL)\n"
        "- The 'So What?': 2 sentences on the human tension or paradox\n"
        "- Standout Quote\n"
        "- Verified Source URL\n"
        "Do not run more than 1 search."
    ),
    expected_output="Structured research brief with verified links, quotes, and image URLs.",
    agent=researcher,
)

write_task = Task(
    description=(
        "Write today's edition of 'Your Morning AI & Culture Dispatch'.\n"
        "Generate raw, clean HTML content (DO NOT wrap in markdown code blocks like ```html).\n"
        "Format requirements:\n"
        "- The Morning Kick: A 2-3 sentence personal or cultural observation introducing today's theme.\n"
        "- 3 Stories: Each story MUST have:\n"
        "  1. A category pill: <span class='pill'>AI &amp; Society</span> (or 'Mind &amp; Craft', 'Culture').\n"
        "  2. A sharp Title: <h2>Title</h2>\n"
        "  3. A 1-sentence Subtitle: <p class='subtitle'>One line explaining the core conflict</p>\n"
        "  4. If an image URL exists: <img src='URL' class='story-img' alt='Story visual' />\n"
        "  5. 2 narrative paragraphs breaking down the tension or paradox.\n"
        "  6. A Takeaway box: <div class='takeaway-box'><strong>The takeaway:</strong> Core insight.</div>\n"
        "  7. A read button: <a href='URL' class='read-btn'>Read full piece &rarr;</a>\n"
        "- Parting Thought at the end."
    ),
    expected_output="Clean HTML body ready for email embedding.",
    agent=writer,
)

# 4. EJECUCIÓN DEL PIPELINE MULTIAGENTE
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.sequential,
    verbose=True,
)

result = crew.kickoff()
newsletter_
