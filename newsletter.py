import os
import smtplib
import time
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from google import genai

# 1. INICIALIZACIÓN DEL CLIENTE Y CONFIGURACIÓN DE RESILIENCIA
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

MODELS = ["gemini-3.8-flash", "gemini-2.5-pro"]

def generate_quick(prompt: str) -> str:
    """Intenta el modelo principal con espera y recurre a Pro si Flash está saturado."""
    for model_name in MODELS:
        for attempt in range(1, 3):
            try:
                print(f"-> Calling {model_name} (attempt {attempt}/2)...")
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
                if response and response.text:
                    return response.text
            except Exception as e:
                err_msg = str(e)
                print(f"Notice on {model_name}: {err_msg}")

                # Aborta de inmediato si la cuota diaria se agotó (429)
                if "429" in err_msg or "resource_exhausted" in err_msg.lower():
                    raise e

                # Si hay saturación (503) o desconexión, espera 6 segundos antes de reintentar
                if "503" in err_msg or "high demand" in err_msg.lower() or "disconnected" in err_msg.lower():
                    time.sleep(6)
                else:
                    break

    raise RuntimeError("Todos los modelos estuvieron temporalmente no disponibles.")

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
scout_output = generate_quick(scout_prompt)

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
writer_output = generate_quick(writer
