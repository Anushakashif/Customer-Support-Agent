import json
import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from graph.state import SupportState, Category, TicketStatus
from rag.retriever import retrieve_relevant_docs
from config.settings import settings

logger = logging.getLogger(__name__)

# ── LLM setup ────────────────────────────────────────────────────────────────
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,        # slight creativity for natural responses
    api_key=settings.OPENAI_API_KEY
)

# ── Prompt ───────────────────────────────────────────────────────────────────
answer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful customer support agent.
        Use the provided knowledge base excerpts to answer the customer's question.
        Be concise, friendly, and accurate.

        If the knowledge base does not contain enough information to answer confidently,
        still provide your best answer but reflect that in a lower confidence score.

        Return ONLY a JSON object with no extra text:
        {{
            "answer": "<your response to the customer>",
            "confidence": <float between 0.0 and 1.0>,
            "reasoning": "<one sentence explaining your confidence level>"
        }}

        Confidence guide:
        - 0.9+ : Answer is directly and completely covered by the knowledge base
        - 0.7  : Answer is mostly covered but has some gaps
        - 0.5  : Answer is partially inferred, not directly in knowledge base
        - 0.3  : Answer is a best guess, knowledge base has little relevant info
        """
    ),
    (
        "human",
        """Customer message: {customer_message}

        Category: {category}
        Priority: {priority}

        Relevant knowledge base excerpts:
        {retrieved_docs}

        Provide your answer and confidence score."""
    )
])

# ── Answer chain ──────────────────────────────────────────────────────────────
answer_chain = answer_prompt | llm

# ── Main node function ────────────────────────────────────────────────────────
def answer_node(state: SupportState) -> SupportState:
    """
    Retrieves relevant docs, generates an answer,
    assigns a confidence score, flags for escalation if needed.
    """
    logger.info(f"Answer Agent started for customer: {state['customer_id']}")

    try:
        # Step 1: Retrieve relevant docs from knowledge base
        retrieved_docs = retrieve_relevant_docs(
            query=state["customer_message"],
            k=3
        )

        # Format docs as numbered list for the prompt
        docs_text = "\n".join([
            f"{i+1}. {doc}"
            for i, doc in enumerate(retrieved_docs)
        ]) if retrieved_docs else "No relevant documents found."

        # Step 2: Call the LLM
        response = answer_chain.invoke({
            "customer_message": state["customer_message"],
            "category":         str(state["category"].value),
            "priority":         str(state["priority"].value),
            "retrieved_docs":   docs_text
        })

        # Step 3: Parse the response
        raw = response.content.strip()
        result = json.loads(raw)

        answer     = result.get("answer", "")
        confidence = float(result.get("confidence", 0.5))

        logger.info(f"Answer Agent confidence: {confidence}")

        # Step 4: Decide if escalation is needed
        should_escalate = confidence < settings.CONFIDENCE_THRESHOLD

        if should_escalate:
            logger.info(f"Confidence {confidence} below threshold {settings.CONFIDENCE_THRESHOLD} → escalating")
        else:
            logger.info(f"Confidence {confidence} sufficient → responding directly")

        return {
            **state,
            "generated_answer":  answer,
            "confidence_score":  confidence,
            "retrieved_docs":    retrieved_docs,
            "should_escalate":   should_escalate,
            "final_response":    answer if not should_escalate else None,
            "ticket_status":     TicketStatus.RESOLVED if not should_escalate else TicketStatus.OPEN,
        }

    except json.JSONDecodeError as e:
        logger.error(f"Answer Agent JSON parse error: {e}")
        return {
            **state,
            "generated_answer": "",
            "confidence_score": 0.0,
            "retrieved_docs":   [],
            "should_escalate":  True,    # escalate if something went wrong
            "final_response":   None,
        }

    except Exception as e:
        logger.error(f"Answer Agent unexpected error: {e}")
        raise