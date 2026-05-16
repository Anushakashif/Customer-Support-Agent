from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # OpenAI
    OPENAI_API_KEY: str

    # App
    APP_NAME: str = "Multi-Agent Support System"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"

    # Redis
    REDIS_URL: str = "redis://localhost:6379"

    # Twilio
    TWILIO_ACCOUNT_SID: str = ""
    TWILIO_AUTH_TOKEN: str = ""
    TWILIO_WHATSAPP_NUMBER: str = ""

    # Slack
    SLACK_BOT_TOKEN: str = ""
    SLACK_ESCALATION_CHANNEL: str = "#support-escalations"

    # Agent Config
    CONFIDENCE_THRESHOLD: float = 0.65
    MAX_CONVERSATION_TURNS: int = 10

    class Config:
        env_file = ".env"
        extra = "allow"

settings = Settings()