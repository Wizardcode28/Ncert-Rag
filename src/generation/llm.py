import os
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from config.settings import (
    GEMINI_MODEL,
    GROQ_MODEL,
    LLM_MAX_RETRIES,
    LLM_TEMPERATURE,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL,
)

KEY_NAMES = {
    "groq": "GROQ_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


def get_secret(key_name: str) -> str:
    """Robust lookup across os.environ, st.secrets (direct, case-insensitive, nested)."""
    # 1. Environment variable
    val = os.getenv(key_name, "")
    if val and val.strip():
        return val.strip()

    # 2. Streamlit secrets
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            # Direct key
            if key_name in st.secrets:
                return str(st.secrets[key_name]).strip()
            
            # Case-insensitive or nested lookup
            for sec_key in st.secrets:
                if sec_key.lower() == key_name.lower():
                    return str(st.secrets[sec_key]).strip()
                try:
                    sec_obj = st.secrets[sec_key]
                    if hasattr(sec_obj, "items"):
                        for sub_k, sub_v in sec_obj.items():
                            if sub_k.lower() == key_name.lower():
                                return str(sub_v).strip()
                except Exception:
                    pass
    except Exception:
        pass

    return ""


def build_single_llm(provider: str) -> BaseChatModel:
    key_name = KEY_NAMES.get(provider)
    api_key = get_secret(key_name) if key_name else ""
    if not api_key:
        raise ValueError(f"{key_name} is missing.")

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
    """Builds LLM with available providers that have valid API keys set."""
    # Check providers in priority order
    provider_candidates = ["groq", "gemini", "openrouter"]
    env_providers = get_secret("LLM_PROVIDERS")
    if env_providers:
        provider_candidates = [p.strip() for p in env_providers.split(",") if p.strip()]

    available_models = []
    for provider in provider_candidates:
        key_name = KEY_NAMES.get(provider)
        if key_name and get_secret(key_name):
            try:
                available_models.append(build_single_llm(provider))
            except Exception:
                pass

    if len(available_models) == 0:
        raise ValueError("No valid LLM providers found.")

    primary = available_models[0]
    fallbacks = available_models[1:]

    if len(fallbacks) > 0:
        return primary.with_fallbacks(fallbacks)
    return primary
