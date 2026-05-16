import json
import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from graph.state import SupportState, Category, Priority
from config.settings import settings

logger = logging.getLogger(__name__)

# ── LLM setup ────────────────────────────────────────────────────────────────
llm = ChatOpenAI(
    model="gpt-4o-mini",           # fast and cheap for classification tasks
    temperature=0,                 # 0 = deterministic, we want consistent labels
    api_key=settings.OPENAI_API_KEY
)

# ── Prompt ───────────────────────────────────────────────────────────────────
triage_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a customer support triage agent. 
        Analyze the customer message and return ONLY a JSON object with no extra text.

        Categories: billing, technical, complaint, general, unknown
        Priorities:
          - urgent  → service down, data loss, security issue
          - high    → broken feature, payment failed
          - medium  → general question with some frustration
          - low     → general question, feedback

        Return this exact JSON structure:
        {{
            "category": "<category>",
            "priority": "<priority>",
            "extracted_entities": {{
                "order_id": "<if mentioned, else null>",
                "account_id": "<if mentioned, else null>",
                "product": "<if mentioned, else null>",
                "error_code": "<if mentioned, else null>"
            }},
            "reasoning": "<one sentence on why you chose this category and priority>"
        }}"""
    ),
    (
        "human",
        "Customer message: {customer_message}"
    )
])

# ── Triage chain ─────────────────────────────────────────────────────────────
triage_chain = triage_prompt | llm

# ── Main node function (called by LangGraph) ─────────────────────────────────
def triage_node(state: SupportState) -> SupportState:
    """
    Receives the state, classifies the message,
    returns updated state with category, priority, entities.
    """
    logger.info(f"Triage Agent started for customer: {state['customer_id']}")

    try:
        # Call the LLM
        response = triage_chain.invoke({
            "customer_message": state["customer_message"]
        })

        # Parse the JSON response
        raw = response.content.strip()
        result = json.loads(raw)

        # Map strings to our Enum types
        category = Category(result.get("category", "unknown"))
        priority = Priority(result.get("priority", "medium"))
        entities = result.get("extracted_entities", {})

        logger.info(f"Triage result → Category: {category}, Priority: {priority}")

        # Return updated state
        return {
            **state,
            "category":           category,
            "priority":           priority,
            "extracted_entities": entities,
        }

    except json.JSONDecodeError as e:
        # If LLM returns malformed JSON, default to safe values
        logger.error(f"Triage Agent JSON parse error: {e}")
        return {
            **state,
            "category":           Category.UNKNOWN,
            "priority":           Priority.MEDIUM,
            "extracted_entities": {},
        }

    except Exception as e:
        logger.error(f"Triage Agent unexpected error: {e}")
        raise