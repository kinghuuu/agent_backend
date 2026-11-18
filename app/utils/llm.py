"""大模型（OpenAI 兼容协议）的公共工具：请求地址拼接与请求头构造。

与业务无关的纯函数，模型管理、会话等模块都复用它，避免各写一套地址拼接逻辑。
"""


def _v1_base(base_url: str) -> str:
    """把服务地址规整到以 ``/v1`` 结尾（已带则原样保留）。"""
    base = base_url.rstrip("/")
    if not base.endswith("/v1") and "/v1/" not in base:
        base = f"{base}/v1"
    return base


def chat_completions_url(base_url: str) -> str:
    """拼接 ``/chat/completions`` 地址；base_url 没带 ``/v1`` 时自动补上。"""
    return f"{_v1_base(base_url)}/chat/completions"


def embeddings_url(base_url: str) -> str:
    """拼接 ``/embeddings`` 地址（Embedding 模型不能走 chat 端点）。"""
    return f"{_v1_base(base_url)}/embeddings"


def rerank_urls(base_url: str) -> list[str]:
    """返回 rerank 端点的候选地址列表，按优先级排列（探测时依次尝试）。

    Rerank 没有官方统一协议，但主流服务（Cohere / Jina / 硅基流动 / 本地
    Xinference）都是 ``POST {base}/v1/rerank``，而本地 HuggingFace TEI 只暴露
    根路径 ``POST {base}/rerank``，所以两个都给出，由调用方按 404 回退。

    参数:
        base_url: 模型配置里的服务地址，可带可不带 ``/v1``。

    返回:
        去重后的候选地址；``base_url`` 已带 ``/v1`` 时只返回一个。
    """
    v1_path = f"{_v1_base(base_url)}/rerank"
    root_path = f"{base_url.rstrip('/')}/rerank"
    return [v1_path] if v1_path == root_path else [v1_path, root_path]


def auth_headers(api_key: str | None) -> dict[str, str]:
    """构造请求头：有 Key 时带 Bearer 认证。"""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers
