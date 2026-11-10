"""大模型（OpenAI 兼容协议）的公共工具：请求地址拼接与请求头构造。

与业务无关的纯函数，模型管理、会话等模块都复用它，避免各写一套地址拼接逻辑。
"""


def chat_completions_url(base_url: str) -> str:
    """拼接 ``/chat/completions`` 地址；base_url 没带 ``/v1`` 时自动补上。"""
    base = base_url.rstrip("/")
    if not base.endswith("/v1") and "/v1/" not in base:
        base = f"{base}/v1"
    return f"{base}/chat/completions"


def auth_headers(api_key: str | None) -> dict[str, str]:
    """构造请求头：有 Key 时带 Bearer 认证。"""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers
