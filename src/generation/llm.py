import os
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from config.settings import (
    GEMINI_MODEL,
    GROQ_MODEL,
    LLM_MAX_RETRIES,
    LLM_PROVIDERS,
    LLM_TEMPERATURE,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL,
)

# Attempt to load from streamlit.secrets if running on Streamlit Cloud
try:
    import streamlit as st
    if hasattr(st, "secrets"):
        for k, v in st.secrets.items():
            if isinstance(v, str) and k not in os.environ:
                os.environ[k] = v
except Exception:
    pass

KEY_NAMES = {
    "groq": "GROQ_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


def get_secret(key: str) -> str:
    """Check os.environ first, then fallback to st.secrets."""
    val = os.getenv(key, "")
    if val:
        return val.strip()
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key]).strip()
    except Exception:
        pass
    return ""


def build_single_llm(provider: str) -> BaseChatModel:
    key_name = KEY_NAMES.get(provider)
    api_key = get_secret(key_name) if key_name else ""
    if not api_key:
        raise ValueError(f"{key_name} is missing. Add it to your .env or Streamlit Secrets.")

    if provider == "groq":
        return ChatGroq(
            model=GROQ_MODEL,
            api_key=api_key,
            temperature=LLM_TEMPERATURE,
            max_retries=LLM_MAX_RETRIES,
        )

    if provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=api_key,
            temperature=LLM_TEMPERATURE,
            max_retries=LLM_MAX_RETRIES,
        )

    return ChatOpenAI(
        model=OPENROUTER_MODEL,
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        temperature=LLM_TEMPERATURE,
        max_retries=LLM_MAX_RETRIES,
    )


def build_llm():
    """Builds LLM with available providers in LLM_PROVIDERS that have valid API keys set."""
    # Ensure environment has providers list
    providers = LLM_PROVIDERS
    env_providers = get_secret("LLM_PROVIDERS")
    if env_providers:
        providers = [p.strip() for p in env_providers.split(",") if p.strip()]

    # If still empty or default, test standard list
    if not providers:
        providers = ["groq", "gemini", "openrouter"]

    available_models = []
    for provider in providers:
        key_name = KEY_NAMES.get(provider)
        if key_name and get_secret(key_name):
            try:
                available_models.append(build_single_llm(provider))
            except Exception:
                pass

    if len(available_models) == 0:
        raise ValueError("No valid LLM providers found. Please check your .env or Streamlit Secrets.")

    primary = available_models[0]
    fallbacks = available_models[1:]

    if len(fallbacks) > 0:
        return primary.with_fallbacks(fallbacks)
    return primary
