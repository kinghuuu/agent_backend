"""全局配置。

优先级：环境变量 > .env 文件 > 类默认值（pydantic-settings 的行为）。
全应用只通过 ``settings`` 单例读取配置，禁止在业务代码里硬编码地址与密钥。
"""

from urllib.parse import quote_plus

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """全局配置：优先读取环境变量 / .env 文件。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ===== 应用 =====
    PROJECT_NAME: str = "Agent Backend"
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 15401
    CORS_ORIGINS: str = "http://localhost:15173,http://127.0.0.1:15173"

    # ===== PostgreSQL =====
    DB_HOST: str = "127.0.0.1"
    DB_PORT: int = 15400
    DB_USER: str = "root"
    DB_PASSWORD: str = "root"
    DB_NAME: str = "agent_backend"

    # ===== Redis =====
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str = ""

    @property
    def DATABASE_URL(self) -> str:
        """异步驱动 asyncpg 的连接串。"""
        pwd = quote_plus(self.DB_PASSWORD)
        return (
            f"postgresql+asyncpg://{self.DB_USER}:{pwd}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    @property
    def REDIS_URL(self) -> str:
        auth = f":{quote_plus(self.REDIS_PASSWORD)}@" if self.REDIS_PASSWORD else ""
        return f"redis://{auth}{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    @property
    def cors_origins(self) -> list[str]:
        return [i.strip() for i in self.CORS_ORIGINS.split(",") if i.strip()]


settings = Settings()
