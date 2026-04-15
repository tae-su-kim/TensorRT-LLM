import os

if os.getenv("TRT_LLM_MINIMAL_IMPORT", "0") == "1":
    __all__ = []
else:
    from .llm import LLM

    __all__ = ["LLM"]
