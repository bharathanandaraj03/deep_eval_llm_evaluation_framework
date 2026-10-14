# DeepEval LLM Evaluation Framework

Regression tests for LLM agents, built on [DeepEval](https://github.com/confident-ai/deepeval). Each test runs the agent on a golden input and scores the output with LLM-as-judge and deterministic metrics. CI fails when any score is below its threshold.

## How it works

1. `datasets/goldens.json` holds the golden cases: input, expected output, reference context and expected tools.
2. `tests/test_agent.py` runs `app.agent.run_enterprise_agent` once per golden.
3. The agent's output, retrieved context and tool calls become an `LLMTestCase`.
4. `assert_test` scores the case against every metric and fails the test if any metric is below its threshold.
5. GitHub Actions runs the suite on every push and pull request to `main`/`master`.

## Project layout

| Path | Purpose |
|---|---|
| `app/agent.py` | Agent under test. Picks a tool, builds a response, and returns `input`, `output`, `retrieval_context` and `tools_called`. |
| `app/tools.py` | Agent tools. `@observe(type="tool")` records each call in DeepEval's trace. |
| `datasets/goldens.json` | Golden test cases. |
| `tests/test_agent.py` | Parametrized DeepEval test with the quality gates. |
| `.github/workflows/deepevaltests.yaml` | CI pipeline. |
| `requirements.txt` | Pinned dependencies. |

## Quality gates

All scores range from 0 to 1, and higher is better. A metric passes when its score is at or above the threshold. The LLM judge is pinned to `gpt-4.1` (`JUDGE_MODEL` in the test) so that scores don't drift when DeepEval changes its default judge.

| Metric | Checks | Uses | Threshold |
|---|---|---|---|
| Faithfulness | Output makes no claims that contradict the retrieved context | `actual_output`, `retrieval_context` | 0.7 |
| Answer Relevancy | Output answers the question asked | `input`, `actual_output` | 0.8 |
| Hallucination | Output doesn't contradict the golden reference context | `actual_output`, `context` | 0.8 |
| Tool Correctness | Agent called every expected tool | `tools_called`, `expected_tools` | 1.0 |

In DeepEval 4.x, `HallucinationMetric` scores like the other metrics: 1.0 means no hallucination, and `threshold` is the minimum passing score. Older versions treated `threshold` as a maximum.

## Golden format

```json
{
  "input": "What are the rules regarding remote work?",
  "expected_output": "Employees can work remotely up to 2 days a week with manager approval.",
  "context": ["Company Policy: Employees are allowed up to 2 days of remote work per week with manager approval."],
  "expected_tools": ["lookup_internal_docs"]
}
```

To add a test case, append an object in this format to `datasets/goldens.json`. You don't need to change any code.

## Run locally

```powershell
pip install -r requirements.txt
$env:OPENAI_API_KEY = "<your-key>"     # or add it to .env (gitignored)
$env:PYTHONUTF8 = 1                    # Windows only: avoids UnicodeEncodeError in DeepEval's console output
deepeval test run tests/test_agent.py
deepeval test run tests/test_agent.py -n 4   # run in parallel with pytest-xdist
```

Results are written to `.deepeval/` (gitignored). Each golden costs about $0.01 in judge calls.

## CI

`.github/workflows/deepevaltests.yaml` installs Python 3.10, installs the dependencies and runs the suite. Before the first run, add `OPENAI_API_KEY` under **Settings → Secrets and variables → Actions**.

## Roadmap

| Area | Planned work |
|---|---|
| Agent | Replace the stubbed response with a real LLM call behind a provider adapter |
| Config | Move thresholds and the judge model into `eval_config.yaml` |
| Coverage | Add Contextual Precision/Recall, `GEval` policy rubrics, `TaskCompletionMetric` and multi-turn `ConversationalTestCase` |
| Safety | Add Bias, Toxicity and PII metrics, plus `deepteam` red teaming |
| Ops | Generate synthetic goldens with `Synthesizer`, track results over time on Confident AI, and run a quick suite on each PR with the full suite nightly |
