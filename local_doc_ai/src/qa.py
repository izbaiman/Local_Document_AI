"""
qa.py  (OPTIONAL BONUS)
=======================
Local question-answering over retrieved document context using
an open-source HuggingFace model.

Workflow:
    1. The retrieval engine finds the top-k most relevant chunks.
    2. Those chunks are concatenated into a context string.
    3. A local generative/seq2seq model produces an answer.

Recommended models (all run on CPU, though GPU is faster):
    • google/flan-t5-base        ~250 MB  — fast, decent quality
    • google/flan-t5-large       ~770 MB  — better quality
    • mistralai/Mistral-7B-v0.1  ~14 GB   — highest quality, requires GPU/RAM
    • tiiuae/falcon-7b-instruct  ~14 GB   — alternative instruction model

Default: google/flan-t5-base  (balanced CPU-friendly choice)

No internet or API keys are required; the model is downloaded once by
HuggingFace's caching mechanism and then runs offline.
"""

from __future__ import annotations

import logging
import textwrap
from typing import List, Optional

logger = logging.getLogger(__name__)


class LocalQAEngine:
    """
    Runs local QA using a HuggingFace text-generation or seq2seq model.

    Args:
        llm_model:   HuggingFace model name or local path.
        max_context: Maximum number of context characters fed to the model.
        max_new_tokens: Token budget for the generated answer.
    """

    def __init__(
        self,
        llm_model: str = "google/flan-t5-base",
        max_context: int = 2000,
        max_new_tokens: int = 256,
    ):
        self.llm_model = llm_model
        self.max_context = max_context
        self.max_new_tokens = max_new_tokens
        self._pipeline = None

    def _get_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline

        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
        except ImportError:
            raise ImportError(
                "transformers is required for QA.\n"
                "Install with: pip install transformers torch"
            )

        logger.info("Loading QA model: %s (this may take a while first time) …", self.llm_model)

        # Detect task type from model name
        if any(x in self.llm_model.lower() for x in ["t5", "bart", "pegasus"]):
            task = "text2text-generation"
        else:
            task = "text-generation"

        try:
            self._pipeline = pipeline(
                task,
                model=self.llm_model,
                device=-1,          # CPU; set to 0 for CUDA GPU
                max_new_tokens=self.max_new_tokens,
            )
        except Exception as exc:
            logger.error("Failed to load QA model: %s", exc)
            raise

        logger.info("QA model loaded (%s).", task)
        return self._pipeline

    def _build_prompt(self, question: str, context_docs: List[str]) -> str:
        """
        Build an instruction-style prompt combining the question and context.
        Works well with Flan-T5 and similar instruction-tuned models.
        """
        # Truncate/concatenate context
        combined = "\n\n---\n\n".join(context_docs)
        if len(combined) > self.max_context:
            combined = combined[: self.max_context] + " [truncated]"

        prompt = textwrap.dedent(f"""
            Answer the following question based only on the documents provided.
            If the answer is not in the documents, say "I don't know."

            Documents:
            {combined}

            Question: {question}

            Answer:
        """).strip()

        return prompt

    def answer(self, question: str, context_docs: List[str]) -> str:
        """
        Generate an answer to the question given context documents.

        Args:
            question:     The user's question.
            context_docs: List of document text chunks from the retrieval engine.

        Returns:
            Generated answer string.
        """
        if not context_docs:
            return "No relevant documents found to answer this question."

        pipe = self._get_pipeline()
        prompt = self._build_prompt(question, context_docs)

        try:
            raw = pipe(prompt)
            if isinstance(raw, list) and raw:
                output = raw[0]
                # text2text-generation returns {"generated_text": "..."}
                if isinstance(output, dict):
                    return output.get("generated_text", str(output)).strip()
                return str(output).strip()
            return str(raw).strip()
        except Exception as exc:
            logger.error("QA generation failed: %s", exc)
            return f"[ERROR] Could not generate answer: {exc}"
