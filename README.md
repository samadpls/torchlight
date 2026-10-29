<p align="center">
  <img alt="torchlight" src="assets/hero.png" width="100%">
</p>

<p align="center">
  <a href="https://huggingface.co/datasets/samadpls/torchlight-threads"><img alt="Dataset on Hugging Face" src="https://img.shields.io/badge/Hugging%20Face-torchlight--threads-FFD21E?logo=huggingface&logoColor=000"></a>
  <a href="https://samadpls.com"><img alt="samadpls.com" src="https://img.shields.io/badge/by-samadpls.com-EE4C2C"></a>
  <a href="LICENSE"><img alt="Code license: Apache-2.0" src="https://img.shields.io/badge/code-Apache--2.0-EE4C2C"></a>
  <a href="https://creativecommons.org/licenses/by/4.0/"><img alt="Data license: CC BY 4.0" src="https://img.shields.io/badge/data-CC%20BY%204.0-EE4C2C"></a>
  <img alt="Python 3.12" src="https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=fff">
</p>

<p align="center">
  <b>A humble PyTorch helper, trained on resolved forum threads.</b><br>
  It gives a specific fix when the question has enough detail, and asks for a minimal repro when it doesn't.
</p>

---

torchlight is a small PyTorch helper model that learns from [PyTorch forum](https://discuss.pytorch.org) threads. This repo covers the whole project, in three parts:

| | Part | What it is | Status |
|---|---|---|---|
| 1 | **Dataset** | Forum questions paired with the answer that resolved them | Released (v0.1.0) |
| 2 | **Model** | LoRA fine-tune of a small LLM on Apple Silicon with MLX | Planned |
| 3 | **Comparison** | The fine-tuned model against a general LLM that answers by calling a forum-search tool | Next: evaluation harness and baselines |

It's built in public, one piece at a time. ⭐ Star the repo to follow along.

## Dataset

[`samadpls/torchlight-threads`](https://huggingface.co/datasets/samadpls/torchlight-threads) holds 6,925 training chats and 260 evaluation questions, built from 22,484 forum threads. For each thread it keeps the post the forum marked as the solution, or the one the asker said fixed their problem, whoever wrote it.

How it's crawled, reviewed and built: [`src/torchlight/scrape/`](src/torchlight/scrape/).

## 🚀 Reproduce it

```bash
uv sync --all-extras
uv run torchlight crawl      # about 2 hours at 3 requests/second
uv run torchlight build      # builds the dataset from the crawl cache
```

Commands for training and evaluation will be added as those parts land.

<details>
<summary><b>Repo layout</b></summary>

```
src/torchlight/
├── scrape/      forum client, response cache, endpoints, crawl
├── build/       parsing, cleaning, answer selection, review, examples
├── publish/     dataset card
└── cli.py
prompts/         instructions for the review agents
assets/          logo and banner
tests/
```
</details>

## Credits

Built by **Abdul Samad** · [samadpls.com](https://samadpls.com) · [GitHub @samadpls](https://github.com/samadpls) · abdulsamadsid1@gmail.com

The answers in the dataset were written by the PyTorch forum community, above all [ptrblck](https://discuss.pytorch.org/u/ptrblck). Every example links back to its source thread.

## License

The code is licensed under [Apache-2.0](LICENSE). The dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), following the PyTorch forums' terms. The logo is inspired by PyTorch's flame but is not the PyTorch logo; PyTorch is a trademark of the Linux Foundation.
