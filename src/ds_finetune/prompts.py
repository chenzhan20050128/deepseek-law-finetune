from __future__ import annotations

PROMPT_STYLE = (
    "下面是一个法律咨询问题，请提供一个回复来解决咨询问题，不需要提供思考过程。\n"
    "### 指令:\n"
    "你是一个法律咨询专家，请回答以下问题，不需要提供思考过程。\n\n"
    "### 问题:\n"
    "{}\n\n"
    "### 回复:\n"
    "{}"
)

TRAIN_PROMPT_STYLE = PROMPT_STYLE


def format_example(question: str, answer: str, eos_token: str) -> str:
    return TRAIN_PROMPT_STYLE.format(question, answer) + eos_token


def format_inference_prompt(question: str) -> str:
    return PROMPT_STYLE.format(question, "")
