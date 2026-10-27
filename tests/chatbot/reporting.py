import json, os, shutil, uuid
from datetime import datetime
from pathlib import Path
from jinja2 import Template
from deepeval.test_case import ConversationalTestCase
from deepeval.test_case.llm_test_case import _MLLM_IMAGE_REGISTRY
import re

RESULTS_DIR = Path(os.getenv("EVAL_RESULTS_DIR", "reports/eval_results"))
REPORT_PATH = Path(os.getenv("EVAL_REPORT_PATH", "reports/eval_report.html"))


def _io(tc):
    if isinstance(tc, ConversationalTestCase):
        transcript = "\n\n".join(f"{t.role.upper()}: {t.content}" for t in tc.turns)
        return tc.scenario or "", transcript, tc.expected_outcome
    return tc.input, tc.actual_output, tc.expected_output


def evaluate_and_record(test_case, metrics):
    for m in metrics:
        try:
            m.measure(test_case)
        except Exception as e:  # judge/API failure is recorded, not hidden
            m.error, m.success = str(e), False
    inp, out, expected = _io(test_case)
    rows = [{"name": m.__name__, "score": m.score, "threshold": m.threshold,
             "success": bool(m.success), "reason": m.reason, "error": m.error} for m in metrics]
    record = {"test_id": os.environ.get("PYTEST_CURRENT_TEST", "").rsplit(" ", 1)[0],
              "input": inp, "output": out, "expected": expected,
              "metrics": rows, "passed": all(r["success"] for r in rows)}
    ids = re.findall(r"\[DEEPEVAL:IMAGE:(.*?)\]", inp or "")
    record["images"] = [f"data:{_MLLM_IMAGE_REGISTRY[i].mimeType};base64,{_MLLM_IMAGE_REGISTRY[i].dataBase64}" for i in ids]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{uuid.uuid4().hex}.json").write_text(json.dumps(record, default=str))
    failed = [r for r in rows if not r["success"]]
    assert not failed, " | ".join(
        f'{r["name"]} {r["score"]} < {r["threshold"]}: {r["error"] or r["reason"]}' for r in failed)


def _is_controller(config):
    return not hasattr(config, "workerinput")  # skip xdist workers


def pytest_sessionstart(session):
    if _is_controller(session.config):
        shutil.rmtree(RESULTS_DIR, ignore_errors=True)


def pytest_sessionfinish(session, exitstatus):
    if not _is_controller(session.config) or not RESULTS_DIR.exists():
        return
    records = [json.loads(p.read_text()) for p in RESULTS_DIR.glob("*.json")]
    records.sort(key=lambda r: (r["passed"], r["test_id"]))  # failures first
    REPORT_PATH.write_text(_TEMPLATE.render(
        records=records, total=len(records), passed=sum(r["passed"] for r in records),
        generated=datetime.now().strftime("%Y-%m-%d %H:%M")), encoding="utf-8")


_TEMPLATE = Template("""<!doctype html><html><head><meta charset="utf-8"><title>Eval Report</title>
<style>
body{font-family:system-ui,sans-serif;margin:24px;background:#fafafa;color:#222}
.case{background:#fff;border:1px solid #ddd;border-left:6px solid #2e7d32;border-radius:6px;margin:12px 0;padding:8px 16px}
.case.fail{border-left-color:#c62828}
summary{cursor:pointer;font-weight:600}
pre{white-space:pre-wrap;background:#f4f4f4;padding:8px;border-radius:4px}
table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:6px;text-align:left;vertical-align:top}
.ok{color:#2e7d32}.bad{color:#c62828;font-weight:600}
</style></head><body>
<h1>Eval Report</h1><p>{{ passed }} / {{ total }} passed · {{ generated }}</p>
{% for r in records %}
<details class="case {{ '' if r.passed else 'fail' }}" {{ '' if r.passed else 'open' }}>
<summary>{{ 'PASS' if r.passed else 'FAIL' }} · {{ r.test_id }}</summary>
<h4>Input</h4><pre>{{ r.input }}</pre>
{% for src in r.images or [] %}<img src="{{ src }}" style="max-width:240px">{% endfor %}
<h4>Output</h4><pre>{{ r.output }}</pre>
{% if r.expected %}<h4>Expected</h4><pre>{{ r.expected }}</pre>{% endif %}
<table><tr><th>Metric</th><th>Score</th><th>Threshold</th><th>Result</th><th>Reason</th></tr>
{% for m in r.metrics %}<tr><td>{{ m.name }}</td>
<td>{{ "%.2f"|format(m.score) if m.score is not none else "-" }}</td><td>{{ m.threshold }}</td>
<td class="{{ 'ok' if m.success else 'bad' }}">{{ 'PASS' if m.success else 'FAIL' }}</td>
<td>{{ m.error or m.reason or '' }}</td></tr>{% endfor %}
</table></details>
{% endfor %}</body></html>""", autoescape=True)
