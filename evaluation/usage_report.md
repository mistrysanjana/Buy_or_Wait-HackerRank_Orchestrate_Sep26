# Token Usage and Cost Report

## HackerRank Orchestrate — Buy or Wait?

This report corresponds to the final full-dataset run that produced the submitted `output.csv`.

### Final run

| Metric | Value |

| Requests processed | 250 |
| Output rows produced | 250 |
| Validation result | PASSED |
| Model providers used | None |
| Models used | None |
| Model/API calls | 0 |
| Input tokens | 0 |
| Output tokens | 0 |
| Total tokens | 0 |
| Average input tokens/request | 0 |
| Average output tokens/request | 0 |
| Average total tokens/request | 0 |
| Estimated total model cost | $0.00 |
| Estimated model cost/request | $0.00 |
| Wall-clock runtime | 25.11 seconds |

### Implementation

The final solution uses a deterministic Python implementation and does not call an external LLM, hosted AI model, banking API, live market-data service, or live exchange-rate service. Financial-state reconstruction, 90-day forecasting, payment-plan selection, spending-change evaluation, image-linked amount resolution, and output validation are performed locally from the participant-provided dataset.

Because there were no model/API calls in the final run, there are no provider-specific token counts or model charges to report. Input and output token counts are therefore zero, and the estimated model/API cost is $0.00.

### Dataset run details

- Dataset requests: `dataset/requests.csv`
- Requests evaluated: 250
- Predictions generated: 250
- Image-derived event amounts resolved: 16
- Output file: repository-root `output.csv`
- Evaluation result: `VALIDATION: PASSED`
- Evaluation errors: 0

### Cost calculation

No external model was invoked, so:

```text
Total input tokens  = 0
Total output tokens = 0
Total tokens        = 0
Total model cost    = $0.00
Average tokens/request = 0
Average cost/request  = $0.00
```

No API keys, credentials, or sensitive configuration values are included in this report.
