"""The comparison: local arms answered with mlx-lm, arm C and judge run as agents.

- `generate`: greedy batched answers, with Phi-4-mini's native search_forum tool calls
- `apis`: whether every torch name in an answer's code exists in the installed PyTorch
- `batches`: batch files for arm C and the judge, and parsers for their results
- `report`: the results table
"""
