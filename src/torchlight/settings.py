from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="TORCHLIGHT_", extra="ignore")

    data_dir: Path = Path("data")
    hf_token: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("HF_TOKEN", "HF_API")
    )
