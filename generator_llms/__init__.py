"""Generator LLM backends.

Cloud backends (AWS Bedrock Claude, Azure OpenAI) require extra SDKs and API
key files that are only needed when those backends are actually used. Import
them lazily so that local-only workflows (e.g. the vLLM generator used during
training) can import this package without cloud credentials installed.
"""

__all__ = [
    "get_claude_response",
    "gpt_chat_35_msg",
    "gpt_chat_4omini",
    "gpt_chat_4o",
]


def __getattr__(name):
    if name == "get_claude_response":
        from .claude_api import get_claude_response

        return get_claude_response
    if name in ("gpt_chat_35_msg", "gpt_chat_4omini", "gpt_chat_4o"):
        from . import gpt_azure

        return getattr(gpt_azure, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
