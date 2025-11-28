import os
from typing import Optional

_PROVIDER_KEY_ENV_VARS = {
    "openai": ["OPENAI_API_KEY"],
    "openrouter": ["OPENROUTER_API_KEY", "OPENAI_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"],
    "google": ["GOOGLE_API_KEY"],
    # Ollama and other local runtimes typically do not require a key
}


def get_provider_api_key(provider: str) -> Optional[str]:
    """Return the configured API key for the given provider.

    Providers can optionally define multiple environment variables. The first
    non-empty variable encountered is returned so callers can keep provider
    tokens separate (e.g., OPENROUTER_API_KEY vs OPENAI_API_KEY).
    """

    env_var_candidates = _PROVIDER_KEY_ENV_VARS.get(provider.lower(), [])
    for env_var in env_var_candidates:
        api_key = os.getenv(env_var)
        if api_key:
            return api_key
    return None
