# ModelRouter

## Evaluation runner

From `backend`, configure `GEMINI_API_KEY` and run:

```powershell
python -m app.evaluation.runner --output evaluation-report.json
```

The runner performs 12 calls in a fixed order: each of the three standard
queries against `fast`, `general`, and `reasoning`, followed by the same three
queries through routing. The JSON report includes each response, selected
model, retries, fallback state, latency, token usage, estimated cost, and
aggregate routing-selection metrics.
