"""日志配置：统一格式、输出到 stdout，便于容器采集。

应用内获取 logger 一律用 ``get_logger(__name__)``，不要直接用 ``print``。
"""

import logging
import sys

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s:%(lineno)d - %(message)s"


def setup_logging(level: int = logging.INFO) -> None:
    """统一日志格式，方便学习阶段观察请求链路。"""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt="%H:%M:%S"))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # 第三方库降噪
    logging.getLogger("uvicorn.access").handlers = [handler]


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
