"""外部全量提示词 provider 示例。"""

from grounding.types import PromptRequest


def build_prompts(requests: tuple[PromptRequest, ...]) -> dict[str, str]:
    """一次接收本次运行的完整 query 集合，再返回 query_id -> prompt。"""

    # 在这里替换成任意组合、检索或外部 LLM 流程；不要读取历史预测。
    return {
        request.record.query_id: (
            "Locate the target described by this query and return one bbox.\n"
            f"Query: {request.record.query.strip()}"
        )
        for request in requests
    }


build_prompts.__prompt_batch__ = True
