# ============================
# 基础镜像
# ============================
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 国内网络建议用清华源；若通畅可删掉 -i 参数
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV PIP_INDEX_URL=${PIP_INDEX_URL}

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --timeout 120 --retries 5 -r requirements.txt

COPY . .

EXPOSE 8000

# 开发模式：--reload，代码改动自动重启
# 入口在根目录：main.py -> app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
