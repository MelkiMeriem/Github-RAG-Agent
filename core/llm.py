from groq import Groq
from core.config import GROQ_API_KEY, GROQ_MODEL

_client = None

def get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=GROQ_API_KEY)
    return _client

def call_llm(prompt: str, max_tokens: int = 1500, temperature: float = 0.1) -> str:
    """Wrapper unifié pour tous les appels LLM du projet."""
    client = get_client()
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()
