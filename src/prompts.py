"""Frozen prompt variants (written before any test-set result was inspected).

Three families of five prompts each, all asking for the same research conclusion
(yes / no / maybe) about the same abstract:

  para : P0 + four semantic paraphrases of the fully specified instruction (same layout).
         Mechanism in the review: instruction-paraphrase sensitivity (Mizrahi et al. 2024;
         Errica et al. 2025).
  fmt  : P0 + four surface-format variants with P0's wording (separators, casing,
         field names, field order). Mechanism: formatting sensitivity (Sclar et al. 2024).
  min  : five underspecified prompts that omit label definitions and output format.
         Mechanism: prompt underspecification (Pecher et al. 2026).

P0 is the canonical prompt used for the single-prompt baseline (B1). It belongs to both
the para and fmt families.
"""
from __future__ import annotations

P0 = (
    "You are assisting with a systematic review. Read the abstract excerpt below and answer "
    "the research question using only the evidence it reports.\n"
    'Answer "yes" if the evidence supports a positive answer, "no" if it supports a negative '
    'answer, and "maybe" if the evidence is inconclusive or mixed.\n'
    "Respond with exactly one word: yes, no, or maybe.\n\n"
    "Abstract:\n{context}\n\nQuestion: {question}"
)

PROMPTS: dict[str, dict] = {
    "P0": {"families": ["para", "fmt"], "text": P0},
    # ---- semantic paraphrases (same layout, same specification) -------------------
    "P1": {"families": ["para"], "text": (
        "As part of an evidence synthesis, examine the following study abstract and decide what "
        "it concludes about the research question, relying solely on the reported findings.\n"
        'Reply "yes" when the findings support an affirmative answer, "no" when they support a '
        'negative answer, and "maybe" when they are inconclusive or mixed.\n'
        "Your reply must be a single word: yes, no, or maybe.\n\n"
        "Abstract:\n{context}\n\nQuestion: {question}")},
    "P2": {"families": ["para"], "text": (
        "Based only on the evidence in the abstract below, what is the answer to the research "
        "question?\n"
        'Say "yes" if the reported evidence points to yes, "no" if it points to no, and "maybe" '
        "if the evidence does not allow a clear answer.\n"
        "Give one word only: yes, no, or maybe.\n\n"
        "Abstract:\n{context}\n\nQuestion: {question}")},
    "P3": {"families": ["para"], "text": (
        "Act as a careful reviewer appraising a biomedical study. Using nothing but the results "
        "described in this abstract, judge the answer to the question.\n"
        'Use "yes" for evidence in favour, "no" for evidence against, and "maybe" for evidence '
        "that is mixed or insufficient.\n"
        "Output exactly one of these words: yes, no, maybe.\n\n"
        "Abstract:\n{context}\n\nQuestion: {question}")},
    "P4": {"families": ["para"], "text": (
        "Your task is to infer the study's conclusion from its abstract. Consider only what the "
        "abstract reports.\n"
        'The answer is "yes" if the evidence supports the question, "no" if the evidence '
        'contradicts it, and "maybe" if the evidence is uncertain.\n'
        "Answer with a single word (yes, no, or maybe).\n\n"
        "Abstract:\n{context}\n\nQuestion: {question}")},
    # ---- surface-format variants of P0 (identical wording) -----------------------
    "F1": {"families": ["fmt"], "text": P0.replace("Abstract:\n", "### ABSTRACT\n")
                                          .replace("Question: ", "### QUESTION\n")},
    "F2": {"families": ["fmt"], "text": P0.upper().replace("{CONTEXT}", "{context}")
                                          .replace("{QUESTION}", "{question}")},
    "F3": {"families": ["fmt"], "text": (
        P0.split("Abstract:")[0] + "Question: {question}\n\nAbstract:\n{context}")},
    "F4": {"families": ["fmt"], "text": P0.replace("\n\nAbstract:\n", "\n\nPASSAGE ::: ")
                                          .replace("\n\nQuestion: ", " ||| QUERY ::: ")},
    # ---- underspecified prompts (no label definitions, no output format) ---------
    "M1": {"families": ["min"], "text": "{context}\n\n{question}"},
    "M2": {"families": ["min"], "text": "{question}\n\n{context}"},
    "M3": {"families": ["min"], "text": "Context: {context}\nQuestion: {question}"},
    "M4": {"families": ["min"], "text": "Read the abstract and answer the question.\n\n{context}\n\n{question}"},
    "M5": {"families": ["min"], "text": "Abstract:\n{context}\n\nWhat does this study conclude? {question}"},
}

FAMILIES = {f: [k for k, v in PROMPTS.items() if f in v["families"]] for f in ("para", "fmt", "min")}
CANONICAL = "P0"
ALL_IDS = list(PROMPTS)


def render(prompt_id: str, context: str, question: str) -> str:
    return PROMPTS[prompt_id]["text"].format(context=context, question=question)
