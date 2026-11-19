import dataclasses
import os


@dataclasses.dataclass
class Settings:
    upstream_base_url: str
    upstream_api_key: str
    log_level: str


def get_settings() -> Settings:
    return Settings(
        upstream_base_url=os.getenv("UPSTREAM_BASE_URL", "https://api.openai.com/v1"),
        upstream_api_key=os.getenv("UPSTREAM_API_KEY", ""),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )
