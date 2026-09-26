# HackerRank Orchestrate - Buy or Wait?
# code/evaluation/main.py
# Deterministic evaluator for the participant-facing dataset.
#
# Run from the repository root with:
#     python code/evaluation/main.py
#
# The evaluator validates output.csv against the public contract, checks every
# prediction against the request/profile/payment-option data, and compares any
# overlapping request IDs with sample_requests.csv. It never uses organizer-only
# files or hidden labels.

from __future__ import annotations

import csv
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# ----------------------------- Configuration -----------------------------

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "dataset"
OUTPUT = ROOT / "output.csv"
REPORT_DIR = ROOT / "evaluation"
REPORT = REPORT_DIR / "evaluation_report.md"

EXPECTED_COLUMNS = [
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
]

ALLOWED_STATUS = {
    "affordable_now",
    "affordable_with_plan",
    "affordable_later",
    "not_affordable",
}

ALLOWED_METHOD = {
    "full_payment",
    "partial_payment",
    "installments",
    "wait",
    "not_recommended",
}

FLEXIBLE_VALUES = {"flexible", "adjustable", "optional", "reducible", "stoppable"}
ACTIVE_EVENT_STATUSES = {"settled", "pending", "scheduled"}
EPS = 1e-6


# ------------------------------- CSV I/O ---------------------------------


def read_csv(path: Path) -> List[Dict[str, str]]:
    """Read a UTF-8 CSV and return rows as dictionaries."""
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_report(text: str) -> None:
    """Write the human-readable evaluation report."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text, encoding="utf-8", newline="\n")


def text(value: object) -> str:
    """Normalize a CSV value to a stripped string."""
    if value is None:
        return ""
    return str(value).strip()


def parse_money(value: object) -> Optional[float]:
    """Parse a numeric monetary field."""
    raw = text(value).replace(",", "")
    if not raw:
        return None
    try:
        number = float(raw)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def parse_bool(value: object) -> Optional[bool]:
    """Parse the common boolean spellings used by CSV datasets."""
    raw = text(value).lower()
    if raw in {"true", "1", "yes", "y", "t"}:
        return True
    if raw in {"false", "0", "no", "n", "f"}:
        return False
    return None


def parse_iso_date(value: object) -> Optional[date]:
    """Parse YYYY-MM-DD without accepting unrelated date formats."""
    raw = text(value)
    if not raw:
        return None
    try:
        return date.fromisoformat(raw)
    except ValueError:
        return None


def fmt_money(value: float) -> str:
    """Format money compactly for the report."""
    if abs(value - round(value)) < EPS:
        return f"{round(value):.0f}"
    return f"{value:.2f}".rstrip("0").rstrip(".")


# ---------------------------- Basic validators ---------------------------


def add_issue(issues: List[str], request_id: str, message: str) -> None:
    """Attach a request-specific validation issue."""
    issues.append(f"{request_id}: {message}")


def validate_schema(
    output_rows: List[Dict[str, str]],
    output_fieldnames: Sequence[str],
    requests: List[Dict[str, str]],
    issues: List[str],
) -> None:
    """Validate columns, row count, IDs, and duplicate IDs."""
    if list(output_fieldnames) != EXPECTED_COLUMNS:
        issues.append(
            "output.csv columns are incorrect. Expected exactly: "
            + ",".join(EXPECTED_COLUMNS)
        )

    request_ids = [text(row.get("request_id")) for row in requests]
    output_ids = [text(row.get("request_id")) for row in output_rows]
    expected_counter = Counter(request_ids)
    actual_counter = Counter(output_ids)

    if len(output_rows) != len(requests):
        issues.append(
            f"Row count mismatch: output has {len(output_rows)}, "
            f"requests.csv has {len(requests)}."
        )

    for request_id, count in expected_counter.items():
        if count != 1:
            issues.append(f"requests.csv contains duplicate request_id: {request_id}")
        if actual_counter.get(request_id, 0) != 1:
            issues.append(
                f"Expected exactly one output row for {request_id}; "
                f"found {actual_counter.get(request_id, 0)}."
            )

    for request_id, count in actual_counter.items():
        if request_id not in expected_counter:
            issues.append(f"output.csv contains unknown request_id: {request_id}")
        if count > 1:
            issues.append(f"output.csv contains duplicate request_id: {request_id}")


def validate_core_fields(
    row: Dict[str, str],
    request: Dict[str, str],
    issues: List[str],
) -> None:
    """Validate the core output contract for one request."""
    request_id = text(request.get("request_id"))
    requested = parse_money(request.get("requested_amount"))
    safe = parse_money(row.get("amount_safe_to_pay"))

    if safe is None:
        add_issue(issues, request_id, "amount_safe_to_pay is not numeric")
    elif requested is None:
        add_issue(issues, request_id, "requested_amount in requests.csv is not numeric")
    elif safe < -EPS or safe > requested + EPS:
        add_issue(
            issues,
            request_id,
            f"amount_safe_to_pay={safe} violates 0 <= safe <= requested={requested}",
        )

    status = text(row.get("affordability_status"))
    method = text(row.get("recommended_payment_method"))
    if status not in ALLOWED_STATUS:
        add_issue(issues, request_id, f"invalid affordability_status={status!r}")
    if method not in ALLOWED_METHOD:
        add_issue(issues, request_id, f"invalid recommended_payment_method={method!r}")

    request_date = parse_iso_date(request.get("request_date"))
    completion_date = parse_iso_date(request.get("desired_completion_date"))
    earliest_raw = text(row.get("earliest_date_for_full_payment"))
    earliest = parse_iso_date(earliest_raw) if earliest_raw else None

    if earliest_raw and earliest is None:
        add_issue(issues, request_id, "earliest_date_for_full_payment is not YYYY-MM-DD")
    if earliest and request_date and earliest < request_date:
        add_issue(issues, request_id, "earliest_date_for_full_payment is before request_date")
    if status == "affordable_now":
        if earliest != request_date:
            add_issue(issues, request_id, "affordable_now must have earliest date equal to request_date")
        if method != "full_payment":
            add_issue(issues, request_id, "affordable_now must use full_payment")
    if status == "affordable_with_plan" and method == "partial_payment":
        if not parse_bool(request.get("allows_partial_payment")):
            add_issue(issues, request_id, "partial_payment used when partial payment is not allowed")
        if safe is not None and requested is not None:
            if not (safe > EPS and safe < requested - EPS):
                add_issue(issues, request_id, "partial_payment requires 0 < safe < requested")
        if earliest is None:
            add_issue(issues, request_id, "partial_payment requires an earliest full-payment date")
        elif completion_date and earliest > completion_date:
            add_issue(issues, request_id, "partial_payment completes after desired_completion_date")

    if method == "not_recommended":
        if text(row.get("payment_plan")) != "none":
            add_issue(issues, request_id, "not_recommended should use payment_plan=none")

    if status == "affordable_later" and method != "wait":
        add_issue(issues, request_id, "affordable_later should use wait")
    if method == "wait" and status not in {"affordable_later", "affordable_with_plan"}:
        add_issue(issues, request_id, "wait is inconsistent with affordability_status")


# ---------------------------- Payment-plan parser ------------------------

PAYMENT_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}):(-?\d+(?:\.\d+)?)$")
STOP_RE = re.compile(r"^stop:([^:|]+)$")
REDUCE_RE = re.compile(r"^reduce_to:([^:|]+):(-?\d+(?:\.\d+)?)$")


def parse_payment_plan(value: str) -> Tuple[Optional[List[Tuple[date, float]]], Optional[str]]:
    """Parse chronological date:amount entries."""
    raw = text(value)
    if raw == "none":
        return [], None
    if not raw:
        return None, "payment_plan is empty; use 'none' when there is no plan"

    entries: List[Tuple[date, float]] = []
    for part in raw.split("|"):
        match = PAYMENT_RE.fullmatch(part.strip())
        if not match:
            return None, f"invalid payment_plan entry: {part!r}"
        parsed_date = parse_iso_date(match.group(1))
        amount = parse_money(match.group(2))
        if parsed_date is None or amount is None:
            return None, f"invalid payment_plan value: {part!r}"
        if amount < -EPS:
            return None, f"negative payment amount in plan: {part!r}"
        entries.append((parsed_date, amount))

    for previous, current in zip(entries, entries[1:]):
        if current[0] < previous[0]:
            return None, "payment_plan is not chronological"
    return entries, None


# ------------------------- Payment-option checks -------------------------


def build_options(rows: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    """Group payment options by request_id."""
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[text(row.get("request_id"))].append(row)
    return grouped


def option_plan(option: Dict[str, str]) -> Optional[List[Tuple[date, float]]]:
    """Expand one installment option into its exact scheduled payments."""
    amount = parse_money(option.get("payment_amount"))
    count_raw = text(option.get("number_of_payments"))
    first = parse_iso_date(option.get("first_payment_date"))
    frequency_raw = text(option.get("payment_frequency_days"))
    if amount is None or first is None:
        return None
    try:
        count = int(float(count_raw))
        frequency = int(float(frequency_raw)) if frequency_raw else 0
    except ValueError:
        return None
    if count <= 0:
        return None
    return [(first.fromordinal(first.toordinal() + i * frequency), amount) for i in range(count)]


def plans_equal(
    actual: List[Tuple[date, float]],
    expected: List[Tuple[date, float]],
) -> bool:
    """Compare two payment schedules with a monetary tolerance."""
    if len(actual) != len(expected):
        return False
    return all(
        left_date == right_date and abs(left_amount - right_amount) <= EPS
        for (left_date, left_amount), (right_date, right_amount) in zip(actual, expected)
    )


def validate_installment_plan(
    request_id: str,
    row: Dict[str, str],
    options: List[Dict[str, str]],
    issues: List[str],
) -> None:
    """Ensure an installment recommendation exactly matches a supplied option."""
    actual, error = parse_payment_plan(row.get("payment_plan", ""))
    if error:
        add_issue(issues, request_id, error)
        return
    assert actual is not None

    matches: List[str] = []
    for option in options:
        expected = option_plan(option)
        if expected is not None and plans_equal(actual, expected):
            matches.append(text(option.get("payment_option_id")))

    if not matches:
        add_issue(
            issues,
            request_id,
            "installment payment_plan does not exactly match any supplied payment option",
        )


def validate_partial_plan(
    request: Dict[str, str],
    row: Dict[str, str],
    issues: List[str],
) -> None:
    """Validate the exact two-payment partial-payment contract."""
    request_id = text(request.get("request_id"))
    actual, error = parse_payment_plan(row.get("payment_plan", ""))
    if error:
        add_issue(issues, request_id, error)
        return
    assert actual is not None
    if len(actual) != 2:
        add_issue(issues, request_id, "partial_payment must contain exactly two payments")
        return

    request_date = parse_iso_date(request.get("request_date"))
    requested = parse_money(request.get("requested_amount"))
    safe = parse_money(row.get("amount_safe_to_pay"))
    earliest = parse_iso_date(row.get("earliest_date_for_full_payment"))

    if request_date and actual[0][0] != request_date:
        add_issue(issues, request_id, "partial_payment first payment is not on request_date")
    if earliest and actual[1][0] != earliest:
        add_issue(issues, request_id, "partial_payment second payment is not earliest_date_for_full_payment")
    if safe is not None and abs(actual[0][1] - safe) > EPS:
        add_issue(issues, request_id, "partial_payment first amount differs from amount_safe_to_pay")
    if requested is not None and safe is not None:
        expected_remaining = requested - safe
        if abs(actual[1][1] - expected_remaining) > EPS:
            add_issue(issues, request_id, "partial_payment remaining amount is incorrect")
        if abs(actual[0][1] + actual[1][1] - requested) > EPS:
            add_issue(issues, request_id, "partial_payment amounts do not sum to requested_amount")


# ------------------------- Spending-change checks ------------------------


def build_events(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    """Index financial events by event_id."""
    return {text(row.get("event_id")): row for row in rows if text(row.get("event_id"))}


def validate_spending_changes(
    request: Dict[str, str],
    row: Dict[str, str],
    events: Dict[str, Dict[str, str]],
    profiles: Dict[str, Dict[str, str]],
    issues: List[str],
) -> None:
    """Validate spending-change syntax and user permissions."""
    request_id = text(request.get("request_id"))
    user_id = text(request.get("user_id"))
    profile = profiles.get(user_id, {})
    raw = text(row.get("spending_changes_needed"))
    if raw == "none":
        return
    if not raw:
        add_issue(issues, request_id, "spending_changes_needed is empty; use 'none'")
        return

    allowed_reduce = {
        item.strip().lower()
        for item in re.split(r"[|,]", text(profile.get("expense_categories_user_is_willing_to_reduce")))
        if item.strip()
    }
    allowed_stop = {
        item.strip().lower()
        for item in re.split(r"[|,]", text(profile.get("expense_categories_user_is_willing_to_stop")))
        if item.strip()
    }

    seen_events: set[str] = set()
    for action in raw.split("|"):
        stop_match = STOP_RE.fullmatch(action.strip())
        reduce_match = REDUCE_RE.fullmatch(action.strip())
        if stop_match:
            event_id = stop_match.group(1)
            if event_id in seen_events:
                add_issue(issues, request_id, f"event {event_id} is changed more than once")
            seen_events.add(event_id)
            event = events.get(event_id)
            if event is None:
                add_issue(issues, request_id, f"unknown spending-change event {event_id}")
                continue
            category = text(event.get("category")).lower()
            flexibility = text(event.get("flexibility")).lower()
            if flexibility not in FLEXIBLE_VALUES:
                add_issue(issues, request_id, f"stop targets non-flexible event {event_id}")
            if category not in allowed_stop:
                add_issue(issues, request_id, f"stop targets category not allowed to stop: {category}")
        elif reduce_match:
            event_id = reduce_match.group(1)
            new_amount = parse_money(reduce_match.group(2))
            if new_amount is None:
                add_issue(issues, request_id, f"invalid reduction amount for {event_id}")
                continue
            if event_id in seen_events:
                add_issue(issues, request_id, f"event {event_id} is changed more than once")
            seen_events.add(event_id)
            event = events.get(event_id)
            if event is None:
                add_issue(issues, request_id, f"unknown spending-change event {event_id}")
                continue
            category = text(event.get("category")).lower()
            flexibility = text(event.get("flexibility")).lower()
            original = parse_money(event.get("amount"))
            minimum = parse_money(event.get("minimum_allowed_amount"))
            if flexibility not in FLEXIBLE_VALUES:
                add_issue(issues, request_id, f"reduce_to targets non-flexible event {event_id}")
            if category not in allowed_reduce:
                add_issue(issues, request_id, f"reduce_to targets category not allowed to reduce: {category}")
            if original is not None and new_amount > original + EPS:
                add_issue(issues, request_id, f"reduce_to increases event {event_id}")
            if minimum is not None and new_amount < minimum - EPS:
                add_issue(issues, request_id, f"reduce_to goes below minimum_allowed_amount for {event_id}")
        else:
            add_issue(issues, request_id, f"invalid spending change syntax: {action!r}")


# -------------------------- Sample comparison ----------------------------


def numeric_close(left: str, right: str) -> bool:
    """Compare numeric strings with a small tolerance."""
    a = parse_money(left)
    b = parse_money(right)
    if a is None or b is None:
        return text(left) == text(right)
    return abs(a - b) <= EPS


def compare_with_samples(
    output_rows: List[Dict[str, str]],
    sample_rows: List[Dict[str, str]],
) -> Tuple[int, int, List[str]]:
    """Compare predictions for request IDs that overlap public solved samples."""
    output_by_id = {text(row.get("request_id")): row for row in output_rows}
    overlaps = 0
    exact_fields = 0
    total_fields = 0
    notes: List[str] = []

    for sample in sample_rows:
        request_id = text(sample.get("request_id"))
        predicted = output_by_id.get(request_id)
        if predicted is None:
            continue
        overlaps += 1
        row_matches = 0
        for field in EXPECTED_COLUMNS[1:]:
            total_fields += 1
            if field == "amount_safe_to_pay":
                same = numeric_close(predicted.get(field, ""), sample.get(field, ""))
            else:
                same = text(predicted.get(field)) == text(sample.get(field))
            if same:
                row_matches += 1
                exact_fields += 1
        notes.append(f"{request_id}: {row_matches}/{len(EXPECTED_COLUMNS) - 1} fields exact")

    return overlaps, exact_fields, notes


# ------------------------------ Reporting --------------------------------


def make_report(
    output_rows: List[Dict[str, str]],
    requests: List[Dict[str, str]],
    issues: List[str],
    sample_rows: List[Dict[str, str]],
) -> str:
    """Build the Markdown evaluation report."""
    request_ids = {text(row.get("request_id")) for row in requests}
    output_ids = [text(row.get("request_id")) for row in output_rows]
    duplicate_count = sum(count - 1 for count in Counter(output_ids).values() if count > 1)
    unknown_count = sum(1 for value in output_ids if value not in request_ids)

    overlaps, exact_fields, sample_notes = compare_with_samples(output_rows, sample_rows)
    sample_total = overlaps * (len(EXPECTED_COLUMNS) - 1)
    sample_accuracy = (100.0 * exact_fields / sample_total) if sample_total else None

    lines = [
        "# Buy or Wait? Evaluation Report",
        "",
        f"- Repository root: `{ROOT}`",
        f"- Output file: `{OUTPUT}`",
        f"- Requests: **{len(requests)}**",
        f"- Prediction rows: **{len(output_rows)}**",
        f"- Duplicate output rows: **{duplicate_count}**",
        f"- Unknown request IDs: **{unknown_count}**",
        f"- Validation errors: **{len(issues)}**",
        "",
        "## Result",
        "",
        ("**PASS**" if not issues else "**FAIL**"),
        "",
        "The evaluator checks the participant-facing output contract and does not use organizer-only labels.",
        "",
        "## Public Sample Comparison",
        "",
    ]

    if sample_total:
        lines.append(f"- Overlapping sample request IDs: **{overlaps}**")
        lines.append(f"- Exact output-field matches: **{exact_fields}/{sample_total} ({sample_accuracy:.2f}%)**")
        lines.extend(f"- {note}" for note in sample_notes)
    else:
        lines.append("- No sample request IDs overlap the evaluation output, so no label comparison was possible.")

    lines.extend(["", "## Validation Errors", ""])
    if issues:
        lines.extend(f"- {issue}" for issue in issues)
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## Checks Performed",
            "",
            "- Exact required output columns and order",
            "- One prediction per request_id",
            "- Numeric amount_safe_to_pay bounds",
            "- Allowed affordability statuses and payment methods",
            "- affordable_now/full_payment consistency",
            "- partial_payment two-payment structure and total",
            "- installment schedule matching against supplied payment options",
            "- spending-change syntax, flexibility, permissions, and minimum amounts",
            "- chronological payment plans",
            "",
        ]
    )
    return "\n".join(lines)


# ------------------------------- Main ------------------------------------


def main() -> int:
    """Run the evaluator and return a shell-friendly exit code."""
    try:
        requests = read_csv(DATASET / "requests.csv")
        profiles_rows = read_csv(DATASET / "financial_profiles.csv")
        events_rows = read_csv(DATASET / "financial_events.csv")
        options_rows = read_csv(DATASET / "request_payment_options.csv")
        sample_rows = read_csv(DATASET / "sample_requests.csv")

        if not OUTPUT.exists():
            raise FileNotFoundError(
                f"Missing prediction file: {OUTPUT}. Run `python code/main.py` first."
            )
        with OUTPUT.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            output_fieldnames = reader.fieldnames or []
            output_rows = list(reader)

        profiles = {text(row.get("user_id")): row for row in profiles_rows}
        events = build_events(events_rows)
        options = build_options(options_rows)

        issues: List[str] = []
        validate_schema(output_rows, output_fieldnames, requests, issues)

        request_by_id = {text(row.get("request_id")): row for row in requests}
        for output_row in output_rows:
            request_id = text(output_row.get("request_id"))
            request = request_by_id.get(request_id)
            if request is None:
                continue

            validate_core_fields(output_row, request, issues)
            validate_spending_changes(request, output_row, events, profiles, issues)

            method = text(output_row.get("recommended_payment_method"))
            if method == "installments":
                validate_installment_plan(
                    request_id,
                    output_row,
                    options.get(request_id, []),
                    issues,
                )
            elif method == "partial_payment":
                validate_partial_plan(request, output_row, issues)
            else:
                plan, error = parse_payment_plan(output_row.get("payment_plan", ""))
                if error:
                    add_issue(issues, request_id, error)
                elif method == "not_recommended" and plan:
                    add_issue(issues, request_id, "not_recommended should use payment_plan=none")

        report_text = make_report(output_rows, requests, issues, sample_rows)
        write_report(report_text)

        print("=" * 72)
        print("HackerRank Orchestrate - Buy or Wait? Evaluation")
        print("=" * 72)
        print(f"Repository : {ROOT}")
        print(f"Requests   : {len(requests)}")
        print(f"Predictions: {len(output_rows)}")
        print(f"Errors     : {len(issues)}")
        print(f"Report     : {REPORT}")
        print("-" * 72)

        if issues:
            for issue in issues[:100]:
                print(f"ERROR: {issue}")
            if len(issues) > 100:
                print(f"... {len(issues) - 100} additional errors written to the report.")
            print("\nVALIDATION: FAILED")
            return 1

        print("VALIDATION: PASSED")
        return 0

    except Exception as exc:
        print(f"FATAL ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
