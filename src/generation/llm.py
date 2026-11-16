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

KEY_NAMES = {
    "groq": "GROQ_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


def build_single_llm(provider: str) -> BaseChatModel:
    key_name = KEY_NAMES.get(provider)
    if not key_name or not os.getenv(key_name):
        raise ValueError(f"{key_name} is missing. Add it to your .env file.")

    if provider == "groq":
        return ChatGroq(
            model=GROQ_MODEL,
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=LLM_TEMPERATURE,
            max_retries=LLM_MAX_RETRIES,
        )

    if provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=LLM_TEMPERATURE,
            max_retries=LLM_MAX_RETRIES,
        )

    return ChatOpenAI(
        model=OPENROUTER_MODEL,
        base_url=OPENROUTER_BASE_URL,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=LLM_TEMPERATURE,
        max_retries=LLM_MAX_RETRIES,
    )


def build_llm():
    """Builds LLM with available providers in LLM_PROVIDERS that have valid API keys set."""
    available_models = []
    for provider in LLM_PROVIDERS:
        key_name = KEY_NAMES.get(provider)
        if key_name and os.getenv(key_name):
            try:
                available_models.append(build_single_llm(provider))
            except Exception:
                pass

    if len(available_models) == 0:
        raise ValueError("No valid LLM providers found. Please check your .env API keys.")

    primary = available_models[0]
    fallbacks = available_models[1:]

    if len(fallbacks) > 0:
        return primary.with_fallbacks(fallbacks)
    return primary
