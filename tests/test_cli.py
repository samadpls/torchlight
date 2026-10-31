import pytest
from typer.testing import CliRunner

from torchlight.cli import app
from torchlight.settings import Settings


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for name in ("HF_TOKEN", "HF_API"):
        monkeypatch.delenv(name, raising=False)


def test_settings_reports_secrets_without_printing_them(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_secret_value")

    result = CliRunner().invoke(app, ["settings"])

    assert result.exit_code == 0
    assert "HF token: set" in result.output
    assert "hf_secret_value" not in result.output


def test_hf_api_is_accepted_as_hf_token(monkeypatch):
    monkeypatch.setenv("HF_API", "from_hf_api")

    assert Settings().hf_token.get_secret_value() == "from_hf_api"


def test_env_file_is_read(tmp_path):
    (tmp_path / ".env").write_text("HF_TOKEN=from_file\n")

    assert Settings().hf_token.get_secret_value() == "from_file"
