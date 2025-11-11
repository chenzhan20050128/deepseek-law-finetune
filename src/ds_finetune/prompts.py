# 从 __future__ 导入 annotations，允许在类型提示中使用前向引用。
from __future__ import annotations

# 定义了模型交互的核心提示模板。
# 这个模板结构化了输入，指导模型扮演一个法律咨询专家的角色，并以特定的格式回应。
# 模板包含了三个部分：
# 1. 引导语： "下面是一个法律咨询问题，请提供一个回复来解决咨询问题，不需要提供思考过程。"
#    这部分为模型设定了场景和任务目标。
# 2. 指令 (Instruction): "你是一个法律咨询专家，请回答以下问题，不需要提供思考过程。"
#    这部分明确了模型的角色和行为准则（例如，直接回答，不展示思考步骤）。
# 3. 问题 (Question): "{}"
#    这是一个占位符，用于在运行时插入具体的法律问题。
# 4. 回复 (Response): "{}"
#    这同样是一个占位符。在训练时，这里会填入标准答案；在推理时，这里为空，等待模型生成内容。
PROMPT_STYLE = (
    "下面是一个法律咨询问题，请提供一个回复来解决咨询问题，不需要提供思考过程。\n"
    "### 指令:\n"
    "你是一个法律咨询专家，请回答以下问题，不需要提供思考过程。\n\n"
    "### 问题:\n"
    "{}\n\n"
    "### 回复:\n"
    "{}"
)

# 训练时使用的提示模板。
# 在这个项目中，训练和推理使用了相同的基本模板结构，
# 因此直接将 `PROMPT_STYLE` 赋值给 `TRAIN_PROMPT_STYLE`。
# 这样做的好处是，如果未来需要为训练设计不同的提示格式，只需修改这里即可，而不会影响推理。
TRAIN_PROMPT_STYLE = PROMPT_STYLE


def format_example(question: str, answer: str, eos_token: str) -> str:
    """
    将一个问答对（question-answer pair）格式化为模型训练所需的单个字符串。

    Args:
        question (str): 原始的法律问题文本。
        answer (str): 对应的标准答案文本。
        eos_token (str): 句末结束符（End-Of-Sentence token），例如 "</s>"。
                         这个 token 告诉模型一个完整的生成序列已经结束。

    Returns:
        str: 格式化后的完整训练样本字符串。
             例如："指令...问题...回复...</s>"
    """
    # 使用训练提示模板，将问题和答案填充到占位符中，
    # 并在末尾追加 eos_token，构成一个完整的训练样本。
    return TRAIN_PROMPT_STYLE.format(question, answer) + eos_token


def format_inference_prompt(question: str) -> str:
    """
    将一个问题格式化为模型推理（或称为“生成”）时所需的输入字符串。

    Args:
        question (str): 用户提出的法律问题。

    Returns:
        str: 格式化后的、准备好输入给模型进行推理的提示字符串。
             回复部分为空，等待模型生成。
    """
    # 使用通用的提示模板，只填充问题部分，
    # 将答案部分留空。模型将从 "### 回复:\n" 之后开始生成文本。
    return PROMPT_STYLE.format(question, "")
