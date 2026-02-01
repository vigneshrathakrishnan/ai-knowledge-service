# utils/prompts.py

SUMMARY_PROMPT = """
You are an AI assistant. Summarize the following content into a concise, clear summary:
{text}
"""

OVERVIEW_PROMPT = """
You are an AI assistant. Provide a high-level overview of this document in 3-4 sentences:
{text}
"""

FALLBACK_PROMPT = """
The user asked: {question}
You have the following retrieved context:
{context}

If the answer is present in the context, provide it. If not, politely say that the document does not provide this information.
"""
