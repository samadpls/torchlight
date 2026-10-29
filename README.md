# torchlight

torchlight is a small PyTorch helper model trained on resolved threads from the [PyTorch forums](https://discuss.pytorch.org). This repo also holds the pipeline that builds its dataset, trains the model and evaluates it against a search-tool baseline.

This is a work in progress.

## Development

```
uv sync --all-extras
uv run torchlight --help
uv run pytest
```

Training uses [MLX](https://github.com/ml-explore/mlx) and needs an Apple Silicon Mac.

## License

The code is licensed under Apache-2.0. The dataset will be published under CC BY 4.0, following the PyTorch forums' terms.
