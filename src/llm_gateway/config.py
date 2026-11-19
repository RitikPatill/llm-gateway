import dataclasses
import os


@dataclasses.dataclass
class Settings:
    upstream_base_url: str
    upstream_api_key: str
    log_level: str
    similarity_threshold: float
    cache_ttl_seconds: int
    cache_db_path: str


def get_settings() -> Settings:
    return Settings(
        upstream_base_url=os.getenv("UPSTREAM_BASE_URL", "https://api.openai.com/v1"),
        upstream_api_key=os.getenv("UPSTREAM_API_KEY", ""),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.92")),
        cache_ttl_seconds=int(os.getenv("CACHE_TTL_SECONDS", "86400")),
        cache_db_path=os.getenv("CACHE_DB_PATH", "cache.db"),
    )
