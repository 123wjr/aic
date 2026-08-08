"""外部提示词 provider 示例。"""

from grounding.types import PromptRequest


def build_prompt(request: PromptRequest) -> str:
    """可在此调用外部 LLM；不要读取历史预测。"""

    return request.record.query.strip().rstrip(" .")
