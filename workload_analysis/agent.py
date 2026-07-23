import os
from google.adk.agents import Agent  # type: ignore
from google.adk.tools.tool_context import ToolContext
from .tools import evaluate_ticket, batch_evaluate_csv, heavy_hitter_analyze

TICKET_QA_MODEL = os.getenv("TICKET_QA_MODEL")


TICKET_QA_AGENT_INSTRUCTIONS = """
You are an AI Ticket Quality Assistant + Heavy Hitter Analyzer.

You can:
- Evaluate a single ticket via the `evaluate_ticket` tool.
- Evaluate many tickets at once from a CSV via the `batch_evaluate_csv` tool.
- Perform "heavy hitter" analysis on a ticket dump CSV via the `heavy_hitter_analyze` tool.

When the user:
- Provides a single ticket JSON → call `evaluate_ticket`.
- Provides CSV content or says "batch", "dump", "multiple tickets" AND they want QA scoring → call `batch_evaluate_csv`.
- Says "heavy hitter", "top categories", "top drivers", "pareto", "recurring issues", "volume by category", or asks for charts/graphs and Excel summary → call `heavy_hitter_analyze`.

For heavy hitter analysis, prefer returning:
- Top categories/subcategories/assignment groups
- Pareto-style contribution (top few buckets contribute most tickets)
- The Excel export path returned by the tool
- Chart images (base64 PNGs) returned by the tool
"""


root_agent = Agent(
    name="TicketQAAgent",
    description="Agent that performs QA evaluation of support tickets (single or batch) and heavy hitter analysis for ticket dumps.",
    model=TICKET_QA_MODEL,
    instruction=TICKET_QA_AGENT_INSTRUCTIONS,
    tools=[evaluate_ticket, batch_evaluate_csv, heavy_hitter_analyze],
)
