
"""Ticket QA Agent package."""

import os
from pathlib import Path
from dotenv import load_dotenv

try:
    import vertexai
except ImportError:
    vertexai = None

PROJECT_ID=os.getenv("PROJECT_ID") 
LOCATION=os.getenv("LOCATION")

for candidate in [Path(".env"), Path("..") / ".env"]:
    if candidate.exists():
        load_dotenv(candidate, override=False)
        break

if vertexai is not None and PROJECT_ID and LOCATION:
    try:
        vertexai.init(project=PROJECT_ID, location=LOCATION)
    except Exception as e:
        print(f"[ticket_qa_agent] Vertex AI init failed: {e}")
