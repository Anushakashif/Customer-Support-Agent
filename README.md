# Multi-Agent Customer Support System

> Autonomous AI support pipeline that triages, answers, and escalates customer messages —
> built with LangGraph, FastAPI, and Twilio.

---

## What This Does

Most support systems make customers wait for a human. This system deploys three specialized AI agents that work in sequence to handle support requests autonomously — classifying the issue, searching a knowledge base, generating a response, and escalating to a human only when genuinely needed.

**Result:** ~80% of tickets resolved without human intervention. Average response time under 10 seconds.

---

## How It Works

```
Customer Message (WhatsApp / SMS / API)
          │
          ▼
  ┌───────────────┐
  │ Triage Agent  │  →  Classifies category, assigns priority, extracts entities
  └───────┬───────┘
          │
          ▼
  ┌───────────────┐
  │ Answer Agent  │  →  Searches knowledge base (RAG), generates response + confidence score
  └───────┬───────┘
          │
     confident?
    ┌─────┴─────┐
   YES          NO
    │            │
    ▼            ▼
 Customer   ┌──────────────────┐
 receives   │ Escalation Agent │  →  Slack alert + handoff summary for human agent
  answer    └──────────────────┘
          │
          ▼
   Ticket logged to database
```

The confidence threshold (default 0.65) is configurable — raise it for stricter automation, lower it for more autonomous handling.

---

## Stack

| Layer | Technology |
|---|---|
| Agent Orchestration | LangGraph |
| LLM | GPT-4o-mini |
| API Server | FastAPI |
| Messaging | Twilio (WhatsApp + SMS) |
| Vector Search | FAISS + OpenAI Embeddings |
| Human Alerts | Slack Block Kit |
| Database | SQLite / PostgreSQL |
| Config | Pydantic Settings |
| Tests | Pytest — 19 tests |

---

## Project Structure

```
multi-agent-support/
│
├── agents/
│   ├── triage_agent.py        # Classifies + prioritizes incoming messages
│   ├── answer_agent.py        # RAG-powered response with confidence scoring
│   └── escalation_agent.py    # Handoff summary + Slack notification
│
├── graph/
│   ├── state.py               # Shared TypedDict passed between all agents
│   └── workflow.py            # LangGraph state machine + routing logic
│
├── api/
│   ├── main.py                # FastAPI app, startup, middleware
│   └── routes.py              # REST endpoints
│
├── integrations/
│   ├── twilio_webhook.py      # WhatsApp + SMS webhook handlers
│   └── slack_notifier.py      # Escalation alerts via Slack Block Kit
│
├── rag/
│   ├── embedder.py            # Document chunking + FAISS index builder
│   └── retriever.py           # Semantic similarity search
│
├── db/
│   ├── database.py            # SQLAlchemy engine + session management
│   └── models.py              # Ticket + ConversationTurn models
│
├── config/
│   └── settings.py            # Pydantic-validated config from .env
│
├── tests/
│   └── test_agents.py         # 19 tests across agents + full pipeline
│
├── chat.html                  # Standalone chat UI (no build step needed)
├── docker-compose.yml
└── Dockerfile
```

---

## Setup

### Prerequisites

- Python 3.11+
- OpenAI API key
- (Optional) Twilio account for WhatsApp/SMS
- (Optional) Slack workspace for escalation alerts

### 1. Clone

```bash
git clone https://github.com/yourusername/multi-agent-support.git
cd multi-agent-support
```

### 2. Virtual environment

```bash
# Windows
python -m venv venv
.\venv\Scripts\Activate.ps1

# Mac / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

```bash
cp .env.example .env
```

```env
# Required
OPENAI_API_KEY=sk-your-key-here

# Messaging (optional)
TWILIO_ACCOUNT_SID=your-sid
TWILIO_AUTH_TOKEN=your-token
TWILIO_WHATSAPP_NUMBER=+1234567890

# Alerts (optional)
SLACK_BOT_TOKEN=xoxb-your-token
SLACK_ESCALATION_CHANNEL=#support-escalations

# Agent behaviour
CONFIDENCE_THRESHOLD=0.65
MAX_CONVERSATION_TURNS=10
```

### 5. Load knowledge base

```bash
python -c "from rag.embedder import load_knowledge_base; load_knowledge_base()"
```

### 6. Run

```bash
uvicorn api.main:app --reload --port 8000
```

Server starts at `http://localhost:8000`
Interactive API docs at `http://localhost:8000/docs`

---

## API

### Process a message

```http
POST /api/v1/message
Content-Type: application/json

{
  "customer_id": "customer-001",
  "message": "How do I reset my password?"
}
```

**Response**

```json
{
  "customer_id": "customer-001",
  "response": "To reset your password, go to Settings > Security > Reset Password. You will receive an email within 2 minutes.",
  "ticket_status": "resolved",
  "category": "technical",
  "priority": "low",
  "confidence_score": 0.95,
  "should_escalate": false,
  "escalation_reason": null
}
```

### All endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | App info |
| GET | `/api/v1/health` | Health check |
| POST | `/api/v1/message` | Process customer message |
| POST | `/webhook/whatsapp` | Twilio WhatsApp webhook |
| POST | `/webhook/sms` | Twilio SMS webhook |

---

## Chat UI

Open `chat.html` directly in a browser while the server is running.
No build step, no npm, no framework — just open and use.

Features: suggestion chips, confidence score pills, escalation status, typing indicator.

---

## Escalation Flow

When confidence falls below threshold:

1. Escalation Agent generates a structured handoff package
2. Slack alert fires with — customer message, category, priority, why AI failed, what was tried, suggested next action
3. Customer receives a polite handoff message instantly
4. Ticket logged as `escalated` in the database with full context

Human agent opens Slack, reads the summary, and resolves the ticket with full context already there.

---

## Tests

```bash
pytest tests/test_agents.py -v
```

```
TestTriageAgent::test_triage_classifies_billing          PASSED
TestTriageAgent::test_triage_assigns_urgent_priority     PASSED
TestTriageAgent::test_triage_never_returns_none_category PASSED
TestAnswerAgent::test_answer_returns_confidence_score    PASSED
TestAnswerAgent::test_answer_high_confidence_not_escalated PASSED
TestEscalationAgent::test_escalation_sets_status         PASSED
TestFullPipeline::test_pipeline_resolves_known_question  PASSED
TestFullPipeline::test_pipeline_always_returns_response  PASSED
...

19 passed in 73.11s
```

---

## Connecting Twilio WhatsApp

1. Sign up at [twilio.com](https://twilio.com) and enable the WhatsApp sandbox
2. Add credentials to `.env`
3. Expose local server with ngrok:

```bash
ngrok http 8000
```

4. Set webhook URL in Twilio console:

```
https://your-ngrok-url.ngrok.io/webhook/whatsapp
```

Messages sent to your Twilio number now flow through the full pipeline automatically.

---

## Extending the Knowledge Base

Replace or extend `SAMPLE_DOCS` in `rag/embedder.py` with your actual support documentation, then rebuild the index:

```bash
python -c "from rag.embedder import load_knowledge_base; load_knowledge_base()"
```

The system immediately searches the updated knowledge base on the next request.

---

## Author

**Anusha Kashif**
CS Student · Karachi, Pakistan
[LinkedIn](https://linkedin.com/in/yourprofile) · [GitHub](https://github.com/yourusername)
