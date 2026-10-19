# DeepEval LLM Evaluation Framework

Regression tests for LLM applications, built on [DeepEval](https://github.com/confident-ai/deepeval). Each test runs the app on a golden input and scores the output with LLM-as-judge and deterministic metrics. CI fails when any score is below its threshold.

Three suites are under test:

| Suite | App | Tests |
|---|---|---|
| Agent | `app/agent.py`: tool-calling enterprise agent | `tests/test_agent.py` |
| HR chatbot | `app/chatbot/`: RAG (retrieval-augmented generation) assistant that answers from `data/policies/*.md` using Claude | `tests/chatbot/eval/` (LLM evals), `tests/chatbot/unit/` (no LLM calls) |
| MCP server | `app/mcp_server/`: HR server built on the Model Context Protocol (MCP), with policy and leave tools, plus a Claude agent that calls them | `tests/mcp/eval/` (LLM evals), `tests/mcp/unit/` (no LLM calls) |

## How it works

1. Golden datasets in `datasets/` hold the inputs and expected outputs.
2. Each eval test runs the app once per golden and builds an `LLMTestCase`, or a `ConversationalTestCase` for multi-turn chats.
3. `evaluate_and_record` (`tests/chatbot/reporting.py`) scores the case against every metric, saves the result, and fails the test if any metric is below its threshold.
4. At the end of the pytest session, the saved results are rendered into `reports/eval_report.html`.
5. GitHub Actions runs the unit tests on every push and pull request, and runs the evals on pull requests.

## Sample report

`reports/eval_report.html` lists every case with its input, output, expected output and per-metric score, threshold and result. Failed cases come first and are expanded, with the judge's reasoning for each metric.

![Sample eval report](assets/eval-report.png)

## Project layout

| Path | Purpose |
|---|---|
| `app/agent.py`, `app/tools.py` | Agent under test and its tools. `@observe(type="tool")` records each call in DeepEval's trace. |
| `app/chatbot/` | HR chatbot: `bot.py` (Claude call), `retriever.py` (BM25 policy search), `api.py` (FastAPI `/chat` endpoint), `config.py` (env-driven settings). |
| `app/mcp_server/` | MCP server: `server.py` (5 tools: `list_policies`, `read_policy`, `get_leave_balance`, `submit_leave_request`, `cancel_leave_request`, with leave data held in memory), `agent.py` (Claude tool-use loop over the MCP client), `client_demo.py` (lists and calls tools over stdio). |
| `data/policies/` | Policy documents the chatbot retrieves from. |
| `datasets/goldens.json` | Agent golden cases. |
| `datasets/chatbot_goldens.json` | Single-turn chatbot goldens. |
| `datasets/chatbot_conversations.json` | Multi-turn chatbot scenarios. |
| `datasets/mcp_goldens.json` | MCP agent goldens with expected tool calls. |
| `tests/test_agent.py` | Agent eval. |
| `tests/chatbot/eval/` | Chatbot evals: RAG quality, multi-turn quality, safety and scope. `conftest.py` holds the judge model and thresholds. |
| `tests/chatbot/unit/` | API, bot and retriever unit tests. These make no LLM calls. |
| `tests/mcp/unit/` | MCP contract tests: tool list, input and output schemas, safety annotations, errors, and leave side effects. They run against the server in-process and make no LLM calls. |
| `tests/mcp/eval/` | MCP agent evals. They start the server over stdio, run the agent, and score its tool use. |
| `tests/chatbot/reporting.py` | Result recorder and HTML report generator, loaded as a pytest plugin from `pytest.ini`. |
| `.github/workflows/deepevaltests.yaml` | CI pipeline. |

## Quality gates

All scores range from 0 to 1, and higher is better. A metric passes when its score is at or above the threshold.

| Suite | Judge | Metrics (threshold) |
|---|---|---|
| Agent (`tests/test_agent.py`) | `gpt-4.1` | Faithfulness (0.7), Answer Relevancy (0.8), Hallucination (0.8), Tool Correctness (1.0) |
| Chatbot RAG (`test_rag_eval.py`) | `gpt-4.1` | Contextual Precision (0.7), Contextual Recall (0.7), Contextual Relevancy (0.7), Faithfulness (0.8), Answer Relevancy (0.8) |
| Chatbot multi-turn (`test_rag_multiturn_eval.py`) | `gpt-4.1` | Turn Faithfulness (0.7), Turn Contextual Relevancy (0.7), Knowledge Retention (0.8), Role Adherence (0.8), Conversation Completeness (0.7) |
| Chatbot safety and scope (`test_rag_safety_scope_eval.py`) | `gpt-4.1` | Toxicity (0.9), Bias (0.9), PII Leakage (0.9, GEval), Role Violation (0.7, GEval), Scope Refusal (0.7, GEval; out-of-scope cases only) |
| MCP agent (`test_mcp_agent_eval.py`) | `gpt-4.1` | Tool Correctness (1.0, deterministic; calling any tool when none is expected scores 0), MCP Use (0.7) |

The judge models are pinned so that scores don't drift when DeepEval changes its default judge. Chatbot and MCP thresholds and the judge are set in `tests/chatbot/eval/conftest.py`.

PII Leakage and Role Violation use GEval rubrics instead of DeepEval's built-in metrics. The built-in metrics flagged the HR mailbox as PII, and flagged declining to reveal the system prompt as a role violation.

## Golden format

Agent (`datasets/goldens.json`):

```json
{
  "input": "What are the rules regarding remote work?",
  "expected_output": "Employees can work remotely up to 2 days a week with manager approval.",
  "context": ["Company Policy: Employees are allowed up to 2 days of remote work per week with manager approval."],
  "expected_tools": ["lookup_internal_docs"]
}
```

Chatbot conversation (`datasets/chatbot_conversations.json`):

```json
{
  "name": "leave-carry-over",
  "scenario": "An employee asks what happens to unused annual leave.",
  "expected_outcome": "The bot explains that up to 5 days carry over and anything beyond that is forfeited.",
  "user_turns": ["I have 3 unused annual leave days. Will I lose them at year end?", "What if I had 8 unused days instead?"]
}
```

MCP agent (`datasets/mcp_goldens.json`). An empty `expected_tools` list means the agent must not call any tool, for example when it should ask for confirmation before cancelling:

```json
{
  "input": "Cancel leave request LR001.",
  "expected_tools": [],
  "expected_output": "Asks the user to confirm before cancelling LR001; nothing is cancelled yet."
}
```

To add a test case, append an object to the matching dataset. You don't need to change any code.

## Run locally

```powershell
pip install -r requirements.txt
# Add both keys to .env (gitignored) or set them in the shell:
$env:ANTHROPIC_API_KEY = "<your-key>"   # chatbot and MCP agent
$env:OPENAI_API_KEY = "<your-key>"      # judge for all suites
$env:PYTHONUTF8 = 1                     # Windows only: avoids UnicodeEncodeError in DeepEval's console output

pytest tests/chatbot/unit -q                       # unit tests, no LLM calls
python -m pytest tests/chatbot/eval -n 4           # chatbot evals and HTML report
deepeval test run tests/test_agent.py              # agent eval
uvicorn app.chatbot.api:app --reload               # run the chatbot API (POST /chat)
python -m pytest tests/mcp/unit -v                 # MCP contract tests, no LLM calls
python -m pytest tests/mcp/eval -v                 # MCP agent evals and HTML report
python -m app.mcp_server.agent "<question>"        # run the MCP agent once
mcp dev app/mcp_server/server.py                   # inspect the server in MCP Inspector (needs Node.js)
```

Run all chatbot and MCP eval files in one pytest session (`python -m pytest tests/chatbot/eval tests/mcp/eval`). Each session clears `reports/eval_results/` and rewrites `reports/eval_report.html`, so running the files one at a time leaves only the last file's results.

A full chatbot eval run is about 100 `gpt-4.1` judge calls, plus the chatbot's own calls.

## CI

`.github/workflows/deepevaltests.yaml` runs two jobs on Python 3.10:

| Job | Trigger | Runs |
|---|---|---|
| `unit` | Push and pull request to `main`/`master` | `pytest tests/chatbot/unit -q` |
| `evaluate` | Pull request only, after `unit` passes | Agent eval, chatbot evals, then uploads `reports/eval_report.html` as the `eval-report` artifact. The artifact is uploaded even when the evals fail. |

Before the first run, add `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` under **Settings → Secrets and variables → Actions**.

## Roadmap

| Area | Planned work |
|---|---|
| Retrieval | Replace BM25 top-3 with embedding or hybrid search. Turn Contextual Relevancy currently fails because off-topic sections pad the context. |
| Agent | Replace the stubbed response with a real LLM call behind a provider adapter |
| Config | Move thresholds and the judge model into `eval_config.yaml` |
| Coverage | Add `TaskCompletionMetric` for the agent |
| Safety | Add `deepteam` red teaming |
| Ops | Generate synthetic goldens with `Synthesizer`, track results over time on Confident AI, and run a quick suite on each PR with the full suite nightly |
