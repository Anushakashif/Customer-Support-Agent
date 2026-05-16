import logging
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from graph.state import SupportState
from config.settings import settings

logger = logging.getLogger(__name__)

# ── Slack client ──────────────────────────────────────────────────────────────
def get_slack_client() -> WebClient:
    return WebClient(token=settings.SLACK_BOT_TOKEN)


# ── Format escalation message as Slack blocks ─────────────────────────────────
def build_escalation_blocks(state: SupportState) -> list:
    """
    Builds a rich Slack message using Block Kit.
    Blocks render as a structured card in Slack.
    """
    priority_emoji = {
        "urgent": "🔴",
        "high":   "🟠",
        "medium": "🟡",
        "low":    "🟢"
    }

    priority    = state["priority"].value if state["priority"] else "unknown"
    category    = state["category"].value if state["category"] else "unknown"
    emoji       = priority_emoji.get(priority, "⚪")
    customer_id = state["customer_id"]
    confidence  = state.get("confidence_score", 0.0)

    return [
        # ── Header ────────────────────────────────────────────────────────────
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{emoji} Escalated Support Ticket"
            }
        },
        # ── Customer info ─────────────────────────────────────────────────────
        {
            "type": "section",
            "fields": [
                {
                    "type": "mrkdwn",
                    "text": f"*Customer ID:*\n{customer_id}"
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Priority:*\n{emoji} {priority.upper()}"
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Category:*\n{category.capitalize()}"
                },
                {
                    "type": "mrkdwn",
                    "text": f"*AI Confidence:*\n{confidence:.0%}"
                }
            ]
        },
        {"type": "divider"},
        # ── Customer message ──────────────────────────────────────────────────
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Customer Message:*\n>{state['customer_message']}"
            }
        },
        # ── Escalation reason ─────────────────────────────────────────────────
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Why Escalated:*\n{state.get('escalation_reason', 'Low confidence score')}"
            }
        },
        # ── Summary ───────────────────────────────────────────────────────────
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Summary for Human Agent:*\n{state.get('escalation_summary', 'No summary available')}"
            }
        },
        {"type": "divider"},
        # ── AI attempted answer ───────────────────────────────────────────────
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*AI Attempted Answer:*\n_{state.get('generated_answer', 'No answer attempted')}_"
            }
        },
        # ── Footer ────────────────────────────────────────────────────────────
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": "🤖 Multi-Agent Support System | Action required by human agent"
                }
            ]
        }
    ]


# ── Send escalation notification ──────────────────────────────────────────────
def send_escalation_notification(state: SupportState) -> bool:
    """
    Sends the escalation alert to the configured Slack channel.
    Returns True if successful, False otherwise.
    """
    # Skip if Slack is not configured
    if not settings.SLACK_BOT_TOKEN:
        logger.warning("Slack token not configured — skipping notification")
        return False

    try:
        client = get_slack_client()
        blocks = build_escalation_blocks(state)

        client.chat_postMessage(
            channel= settings.SLACK_ESCALATION_CHANNEL,
            text=    f"Escalated ticket from customer {state['customer_id']}",  # fallback text
            blocks=  blocks
        )

        logger.info(f"Escalation notification sent to {settings.SLACK_ESCALATION_CHANNEL}")
        return True

    except SlackApiError as e:
        logger.error(f"Slack API error: {e.response['error']}")
        return False

    except Exception as e:
        logger.error(f"Slack notifier unexpected error: {e}")
        return False