import dataclasses
import logging
import os

import yaml

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Settings:
    upstream_base_url: str
    upstream_api_key: str
    log_level: str
    similarity_threshold: float
    cache_ttl_seconds: int
    cache_db_path: str
    routes_config_path: str = "config/routes.yaml"
    complexity_threshold: int = 50  # informational; thresholds live in routes.yaml


def get_settings() -> Settings:
    return Settings(
        upstream_base_url=os.getenv("UPSTREAM_BASE_URL", "https://api.openai.com/v1"),
        upstream_api_key=os.getenv("UPSTREAM_API_KEY", ""),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.92")),
        cache_ttl_seconds=int(os.getenv("CACHE_TTL_SECONDS", "86400")),
        cache_db_path=os.getenv("CACHE_DB_PATH", "cache.db"),
        routes_config_path=os.getenv("ROUTES_CONFIG_PATH", "config/routes.yaml"),
        complexity_threshold=int(os.getenv("COMPLEXITY_THRESHOLD", "50")),
    )


def load_routes_config(path: str) -> dict:
    """Load and return the parsed routes YAML config.

    Returns {} on missing file or parse error so the proxy degrades gracefully.
    """
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.warning("routes config not found at %s — routing disabled", path)
        return {}
    except Exception as exc:
        logger.warning("failed to load routes config from %s: %s — routing disabled", path, exc)
        return {}
