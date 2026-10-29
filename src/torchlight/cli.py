import typer
from pydantic import SecretStr

from torchlight.settings import Settings

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.callback()
def main() -> None:
    """Build the torchlight dataset, train the model and evaluate it."""


def _is_set(secret: SecretStr | None) -> bool:
    return secret is not None and bool(secret.get_secret_value())


@app.command()
def settings() -> None:
    """Show resolved settings. Secrets are reported only as set or missing."""
    s = Settings()
    typer.echo(f"data_dir: {s.data_dir.resolve()}")
    typer.echo(f"HF token: {'set' if _is_set(s.hf_token) else 'missing'}")
    typer.echo(f"Anthropic API key: {'set' if _is_set(s.anthropic_api_key) else 'missing'}")
