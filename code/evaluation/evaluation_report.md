# Buy or Wait? Evaluation Report

- Repository root: `C:\Users\Aman kumar\OneDrive\Desktop\hackerrank-orchestrate-september26`
- Output file: `C:\Users\Aman kumar\OneDrive\Desktop\hackerrank-orchestrate-september26\output.csv`
- Requests: **250**
- Prediction rows: **250**
- Duplicate output rows: **0**
- Unknown request IDs: **0**
- Validation errors: **0**

## Result

**PASS**

The evaluator checks the participant-facing output contract and does not use organizer-only labels.

## Public Sample Comparison

- No sample request IDs overlap the evaluation output, so no label comparison was possible.

## Validation Errors

- None

## Checks Performed

- Exact required output columns and order
- One prediction per request_id
- Numeric amount_safe_to_pay bounds
- Allowed affordability statuses and payment methods
- affordable_now/full_payment consistency
- partial_payment two-payment structure and total
- installment schedule matching against supplied payment options
- spending-change syntax, flexibility, permissions, and minimum amounts
- chronological payment plans
