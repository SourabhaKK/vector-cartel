"""
Standalone launcher for the SecureOps Assistant Gradio demo.

Loads GEMINI_API_KEY / HF_API_KEY from .env (gitignored, never committed)
instead of having them passed as literals on the command line.
"""

from dotenv import load_dotenv

load_dotenv()

import gradio as gr
from src.gradio_demo import setup, chat_fn

llm_router, retrieval_fn = setup()

gr.ChatInterface(
    fn=lambda m, h: chat_fn(m, h, llm_router, retrieval_fn),
    title="SecureOps Assistant",
).launch()
