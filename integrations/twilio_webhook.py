import logging
from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import PlainTextResponse
from twilio.rest import Client
from twilio.request_validator import RequestValidator
from graph.workflow import run_support_pipeline
from config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Twilio client ─────────────────────────────────────────────────────────────
def get_twilio_client() -> Client:
    return Client(
        settings.TWILIO_ACCOUNT_SID,
        settings.TWILIO_AUTH_TOKEN
    )

# ── Send reply back to customer ───────────────────────────────────────────────
def send_whatsapp_reply(to: str, message: str) -> None:
    """
    Sends a WhatsApp message back to the customer via Twilio.
    """
    try:
        client = get_twilio_client()
        client.messages.create(
            from_=f"whatsapp:{settings.TWILIO_WHATSAPP_NUMBER}",
            to=f"whatsapp:{to}",
            body=message
        )
        logger.info(f"Reply sent to {to}")
    except Exception as e:
        logger.error(f"Failed to send WhatsApp reply: {e}")
        raise


# ── Validate Twilio signature ─────────────────────────────────────────────────
def validate_twilio_request(request: Request, form_data: dict) -> bool:
    """
    Verifies the request actually came from Twilio and not
    a random bad actor hitting your webhook URL.
    """
    validator  = RequestValidator(settings.TWILIO_AUTH_TOKEN)
    signature  = request.headers.get("X-Twilio-Signature", "")
    url        = str(request.url)
    return validator.validate(url, form_data, signature)


# ── WhatsApp webhook endpoint ─────────────────────────────────────────────────
@router.post("/whatsapp")
async def whatsapp_webhook(
    request:     Request,
    From:        str = Form(...),    # customer's WhatsApp number
    Body:        str = Form(...),    # customer's message text
    ProfileName: str = Form(""),     # customer's WhatsApp display name
):
    """
    Twilio calls this endpoint when a customer sends a WhatsApp message.
    Runs the message through the pipeline and replies automatically.
    """
    logger.info(f"WhatsApp message from {From}: {Body[:50]}")

    # Extract customer phone number as ID
    customer_id = From.replace("whatsapp:", "").replace("+", "")

    try:
        # Run through the full agent pipeline
        result = run_support_pipeline(
            customer_id=     customer_id,
            customer_message=Body,
        )

        final_response = result["final_response"]
        ticket_status  = result["ticket_status"].value

        logger.info(f"Pipeline result for {customer_id}: {ticket_status}")

        # Send reply back to customer on WhatsApp
        send_whatsapp_reply(
            to=      From.replace("whatsapp:", ""),
            message= final_response
        )

        # Return 200 to Twilio so it knows we handled it
        return PlainTextResponse(content="OK", status_code=200)

    except Exception as e:
        logger.error(f"Webhook error for {From}: {e}")
        # Still return 200 to Twilio — otherwise it retries endlessly
        return PlainTextResponse(content="OK", status_code=200)


# ── SMS webhook endpoint ──────────────────────────────────────────────────────
@router.post("/sms")
async def sms_webhook(
    request: Request,
    From:    str = Form(...),
    Body:    str = Form(...),
):
    """
    Same as WhatsApp but for regular SMS messages.
    """
    logger.info(f"SMS message from {From}: {Body[:50]}")

    customer_id = From.replace("+", "")

    try:
        result = run_support_pipeline(
            customer_id=     customer_id,
            customer_message=Body,
        )

        client = get_twilio_client()
        client.messages.create(
            from_=settings.TWILIO_WHATSAPP_NUMBER,
            to=   From,
            body= result["final_response"]
        )

        return PlainTextResponse(content="OK", status_code=200)

    except Exception as e:
        logger.error(f"SMS webhook error for {From}: {e}")
        return PlainTextResponse(content="OK", status_code=200)