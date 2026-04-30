from typing import List, Dict


SYSTEM_TEMPLATE = """\
You are an expert data analyst assistant. A dataset summary is provided below.

Output requirements and command protocol:
- By default, answer concisely in plain English.
- When the user specifically asks you to perform an action (for example: "Export the top 10 rows as CSV", "Show only numeric columns", "Provide a SQL-style filter and return a downloadable CSV"), respond with a JSON object only (no surrounding prose) that follows this schema:

    {
        "type": "answer" | "command",
        "content": "<human-readable text summary or instructions>",
        "command": {
            "name": "<command_name>",
            "args": { /* optional command arguments */ }
        } | null
    }

    - `type` must be "answer" for normal textual replies and "command" when you are instructing the system to perform a task.
    - `content` is a short human-readable explanation the UI should show.
    - `command` is present only when `type` == "command". `name` is the action to run (examples below), `args` contains structured parameters.

Example commands (implementations may vary):
- `show_columns` → args: { }
- `export_sample_csv` → args: { "rows": 10 }
- `filter_rows` → args: { "filter": "Country == 'US' and Sales > 1000" }

When the user does NOT request an action, produce a plain-text answer (no JSON).

Rules:
- Use the pre-computed statistics EXACTLY — they are calculated from the full dataset by pandas (not estimates).
- When asked for sums, averages, min/max — quote the exact values from the summary.
- For filtering/row-level questions, reason from the sample rows and distributions.
- Be clear when estimating vs using exact values.
- Format numbers with commas. Keep answers clear and concise.
- If a question cannot be answered from the available summary, say so explicitly.

{summary}
"""


class _OpenAICompatClient:
    """Works for both OpenAI and Groq (Groq is OpenAI-compatible)."""

    def __init__(self, api_key: str, model: str, base_url: str = None):
        import openai
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = openai.OpenAI(**kwargs)
        self.model = model

    def chat(self, summary: str, history: List[Dict], question: str) -> str:
        messages = [{"role": "system", "content": SYSTEM_TEMPLATE.format(summary=summary)}]
        messages.extend(history[-20:])
        messages.append({"role": "user", "content": question})
        resp = self._client.chat.completions.create(
            model=self.model, messages=messages, temperature=0.2, max_tokens=2048
        )
        return resp.choices[0].message.content


class _GeminiClient:
    def __init__(self, api_key: str, model: str):
        from google import genai
        from google.genai import types
        self._client = genai.Client(api_key=api_key)
        self._types = types
        self.model = model

    def chat(self, summary: str, history: List[Dict], question: str) -> str:
        system = SYSTEM_TEMPLATE.format(summary=summary)
        contents = []
        for m in history[-20:]:
            role = "user" if m["role"] == "user" else "model"
            contents.append(self._types.Content(
                role=role, parts=[self._types.Part(text=m["content"])]
            ))
        contents.append(self._types.Content(
            role="user", parts=[self._types.Part(text=question)]
        ))
        resp = self._client.models.generate_content(
            model=self.model,
            contents=contents,
            config=self._types.GenerateContentConfig(
                system_instruction=system,
                temperature=0.2,
                max_output_tokens=2048,
            ),
        )
        return resp.text


PROVIDER_DEFAULTS = {
    "groq": {
        "model": "llama-3.3-70b-versatile",
        "base_url": "https://api.groq.com/openai/v1",
    },
    "openai": {
        "model": "gpt-4o-mini",
        "base_url": None,
    },
    "gemini": {
        "model": "gemini-1.5-flash",
    },
}

PROVIDER_MODELS = {
    "groq": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "llama-3.1-70b-versatile", "gemma2-9b-it", "mixtral-8x7b-32768"],
    "openai": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
    "gemini": ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"],
}


def create_client(provider: str, api_key: str, model: str):
    if not api_key:
        raise ValueError(
            f"No API key set for provider '{provider}'. "
            f"Please add it to backend/.env and restart."
        )
    provider = provider.lower()
    defaults = PROVIDER_DEFAULTS.get(provider)
    if not defaults:
        raise ValueError(f"Unknown provider '{provider}'. Choose: groq, openai, gemini")

    resolved_model = model or defaults["model"]

    if provider == "gemini":
        return _GeminiClient(api_key, resolved_model)
    else:
        return _OpenAICompatClient(api_key, resolved_model, defaults.get("base_url"))
