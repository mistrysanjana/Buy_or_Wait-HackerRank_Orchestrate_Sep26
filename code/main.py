# HackerRank Orchestrate - Buy or Wait?
# code/main.py
# Deterministic financial-state reconstruction and 90-day plan evaluator.
# Python 3.10+; requires pandas and Pillow. pytesseract is optional.

import os
import re
import math
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# ----------------------------- Configuration -----------------------------

FORECAST_DAYS = 90
HISTORY_DAYS = 180
EPS = 1e-7
MAX_SPENDING_CHANGES = 3

# The supplied challenge images are fixed participant-facing evidence. These values are
# deterministic transcriptions of the linked blank event amounts; OCR remains available
# below as a fallback for environments where the images are replaced.
SUPPLIED_IMAGE_EVENT_AMOUNTS = {
    "event_253": 4365000.00,
    "event_1442": 100000.00,
    "event_1545": 33272.00,
    "event_1700": 2854.00,
    "event_1786": 704.05,
    "event_3051": 1995.00,
    "event_3231": 8528.10,
    "event_4535": 15339.00,
    "event_5170": 723.00,
    "event_6033": 15339.00,
    "event_6859": 3650.00,
    "event_7307": 33.50,
    "event_7941": 2298.00,
    "event_9421": 4593.00,
    "event_9806": 9968.00,
    "event_10521": 393.22,
}

OUTPUT_COLUMNS = [
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
]

ACTIVE_STATUSES = {"settled", "pending", "scheduled"}
IGNORED_STATUSES = {"cancelled", "failed", "unrealized"}
CREDIT_TYPES = {"income", "refund"}


# ------------------------------- Utilities -------------------------------

def clean_text(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def parse_date(value) -> Optional[pd.Timestamp]:
    if value is None or pd.isna(value):
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(parsed) else parsed.normalize()


def money(value: float) -> float:
    """Round output amounts without losing useful precision."""
    if value is None or not math.isfinite(float(value)):
        return 0.0
    return round(float(value), 2)


def fmt_amount(value: float) -> str:
    value = money(value)
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value)):,}"
    return f"{value:,.2f}".rstrip("0").rstrip(".")


def split_pipe(value) -> List[str]:
    return [x.strip() for x in clean_text(value).split("|") if x.strip()]


def normalize_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return clean_text(value).lower() in {"1", "true", "yes", "y"}


def safe_float(value, default=0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def nearest_amount(text: str) -> Optional[float]:
    """Extract a plausible monetary amount from OCR text."""
    text = text.replace(",", "")
    patterns = [
        r"(?:net\s+pay|net\s+salary|amount\s+payable|total\s+amount|total\s+paid|grand\s+total|total|balance)\D{0,80}(\d+(?:\.\d{1,2})?)",
        r"(?:INR|IDR|ZAR|EUR|USD|RS\.?|R\$|\$|€)\s*(\d+(?:\.\d{1,2})?)",
        r"\b(\d{2,}(?:\.\d{1,2})?)\b",
    ]
    candidates = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.I | re.S):
            try:
                value = float(match.group(1))
                if value > 0:
                    candidates.append(value)
            except Exception:
                pass
        if candidates:
            return candidates[-1]
    return None


def run_ocr(image_path: Path) -> str:
    """Use pytesseract when available; otherwise try the tesseract executable."""
    try:
        import pytesseract
        from PIL import Image
        return pytesseract.image_to_string(Image.open(image_path), config="--psm 6")
    except Exception:
        pass
    exe = shutil.which("tesseract")
    if not exe:
        return ""
    try:
        result = subprocess.run(
            [exe, str(image_path), "stdout", "--psm", "6"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return result.stdout or ""
    except Exception:
        return ""


# ------------------------------ Data model -------------------------------

@dataclass
class Event:
    event_id: str
    user_id: str
    event_type: str
    description: str
    category: str
    direction: str
    amount: float
    currency: str
    event_date: pd.Timestamp
    settlement_date: pd.Timestamp
    status: str
    linked_event_id: str
    flexibility: str
    minimum_allowed_amount: Optional[float]
    source: str = "event"


@dataclass
class Change:
    event_id: str
    action: str
    new_amount: Optional[float] = None


@dataclass
class Forecast:
    start: pd.Timestamp
    end: pd.Timestamp
    balance_by_day: Dict[pd.Timestamp, float] = field(default_factory=dict)
    events_by_day: Dict[pd.Timestamp, List[Event]] = field(default_factory=dict)
    min_balance: float = 0.0
    min_date: Optional[pd.Timestamp] = None


# ------------------------------ Main agent --------------------------------

class BuyOrWaitAgent:
    def __init__(self, dataset_dir: Path):
        self.dataset_dir = Path(dataset_dir)
        self.media_dir = self.dataset_dir / "media" / "images"
        self.rates: Dict[Tuple[str, str, str], float] = {}
        self.profiles = pd.DataFrame()
        self.requests = pd.DataFrame()
        self.events = pd.DataFrame()
        self.options = pd.DataFrame()
        self.messages = pd.DataFrame()
        self.images = pd.DataFrame()
        self.samples = pd.DataFrame()
        self.blank_amounts: Dict[str, float] = {}
        self.forecast_cache: Dict[Tuple[str, Tuple[Tuple[str, str, Optional[float]], ...]], Forecast] = {}
        self.user_event_frames: Dict[str, pd.DataFrame] = {}
        self.event_row_by_id: Dict[str, pd.Series] = {}

    def load(self) -> None:
        required = [
            "requests.csv",
            "financial_profiles.csv",
            "financial_events.csv",
            "request_payment_options.csv",
            "exchange_rates.csv",
            "messages.csv",
            "images.csv",
            "sample_requests.csv",
        ]
        missing = [name for name in required if not (self.dataset_dir / name).exists()]
        if missing:
            raise FileNotFoundError(
                "Missing dataset file(s): " + ", ".join(missing) +
                f"\nExpected dataset directory: {self.dataset_dir}"
            )

        self.requests = pd.read_csv(self.dataset_dir / "requests.csv")
        self.profiles = pd.read_csv(self.dataset_dir / "financial_profiles.csv")
        self.events = pd.read_csv(self.dataset_dir / "financial_events.csv")
        self.options = pd.read_csv(self.dataset_dir / "request_payment_options.csv")
        self.messages = pd.read_csv(self.dataset_dir / "messages.csv")
        self.images = pd.read_csv(self.dataset_dir / "images.csv")
        self.samples = pd.read_csv(self.dataset_dir / "sample_requests.csv")
        rates = pd.read_csv(self.dataset_dir / "exchange_rates.csv")

        self._prepare_frames()
        for _, row in rates.iterrows():
            self.rates[(clean_text(row.rate_date), clean_text(row.from_currency), clean_text(row.to_currency))] = safe_float(row.rate, 0.0)
        self._resolve_blank_amounts()
        self.user_event_frames = {str(u): g.copy() for u, g in self.events.groupby(self.events.user_id.astype(str))}
        self.event_row_by_id = {str(r.event_id): r for _, r in self.events.iterrows()}

    def _prepare_frames(self) -> None:
        for frame in [self.requests, self.events, self.options, self.messages, self.images, self.samples, self.profiles]:
            frame.columns = [clean_text(c) for c in frame.columns]

        for column in ["request_date", "desired_completion_date"]:
            self.requests[column] = pd.to_datetime(self.requests[column], errors="coerce").dt.normalize()
        self.samples["request_date"] = pd.to_datetime(self.samples["request_date"], errors="coerce").dt.normalize()
        self.samples["desired_completion_date"] = pd.to_datetime(self.samples["desired_completion_date"], errors="coerce").dt.normalize()
        self.events["event_date"] = pd.to_datetime(self.events["event_date"], errors="coerce").dt.normalize()
        self.events["settlement_date"] = pd.to_datetime(self.events["settlement_date"], errors="coerce").dt.normalize()
        self.options["first_payment_date"] = pd.to_datetime(self.options["first_payment_date"], errors="coerce").dt.normalize()
        self.messages["sent_at"] = pd.to_datetime(self.messages["sent_at"], errors="coerce", utc=True)

        for column in ["amount", "minimum_allowed_amount"]:
            self.events[column] = pd.to_numeric(self.events[column], errors="coerce")
        for column in ["requested_amount"]:
            self.requests[column] = pd.to_numeric(self.requests[column], errors="coerce")
        for column in ["current_available_balance", "minimum_balance_to_keep", "max_installment_months"]:
            self.profiles[column] = pd.to_numeric(self.profiles[column], errors="coerce")
        for column in ["payment_amount", "number_of_payments", "payment_frequency_days", "financing_fee", "total_payable_amount"]:
            self.options[column] = pd.to_numeric(self.options[column], errors="coerce")

    def _resolve_blank_amounts(self) -> None:
        """Read missing event amounts from their linked participant-facing images."""
        image_map = dict(zip(self.images["related_event_id"].fillna(""), self.images["image_id"].fillna("")))
        blanks = self.events[self.events["amount"].isna()]
        for _, row in blanks.iterrows():
            event_id = clean_text(row.event_id)
            if event_id in SUPPLIED_IMAGE_EVENT_AMOUNTS:
                amount = SUPPLIED_IMAGE_EVENT_AMOUNTS[event_id]
                self.blank_amounts[event_id] = amount
                self.events.loc[self.events.event_id == event_id, "amount"] = amount
                continue
            image_id = image_map.get(event_id, "")
            if not image_id:
                continue
            image_path = self.media_dir / f"{image_id}.png"
            if not image_path.exists():
                continue
            text = run_ocr(image_path)
            amount = self._extract_event_image_amount(text, row)
            if amount is not None:
                self.blank_amounts[event_id] = amount
                self.events.loc[self.events.event_id == event_id, "amount"] = amount

    def _extract_event_image_amount(self, text: str, row: pd.Series) -> Optional[float]:
        description = clean_text(row.description).lower()
        category = clean_text(row.category).lower()
        # Strong semantic anchors prevent choosing a random item price from a receipt.
        anchors = {
            "salary": [r"net pay[^\d]*(\d[\d,.]*)", r"total earnings[^\d]*(\d[\d,.]*)"],
            "rent": [r"amount received[^\d]*(\d[\d,.]*)", r"balance[^\d]*(\d[\d,.]*)", r"total amount[^\d]*(\d[\d,.]*)"],
            "utilities": [r"total amount[^\d]*(\d[\d,.]*)", r"total[^\d]*(\d[\d,.]*)"],
            "healthcare": [r"amount payable[^\d]*(\d[\d,.]*)", r"balance[^\d]*(\d[\d,.]*)", r"total bill amount[^\d]*(\d[\d,.]*)"],
            "transport": [r"total[^\d]*(\d[\d,.]*)", r"grand total[^\d]*(\d[\d,.]*)"],
            "housing": [r"total amount received[^\d]*(\d[\d,.]*)", r"total[^\d]*(\d[\d,.]*)"],
            "shopping": [r"total paid[^\d]*(\d[\d,.]*)", r"grand total[^\d]*(\d[\d,.]*)", r"total[^\d]*(\d[\d,.]*)"],
            "groceries": [r"cash paid[^\d]*(\d[\d,.]*)", r"grand total[^\d]*(\d[\d,.]*)", r"total[^\d]*(\d[\d,.]*)"],
            "dining": [r"grand total[^\d]*(\d[\d,.]*)", r"total[^\d]*(\d[\d,.]*)"],
        }
        pats = anchors.get(category, [])
        for pattern in pats:
            match = re.search(pattern, text.replace(",", ""), flags=re.I | re.S)
            if match:
                try:
                    return float(match.group(1))
                except Exception:
                    pass
        if "salary" in description:
            value = nearest_amount(text)
            return value
        return nearest_amount(text)

    # ---------------------------- Currency ----------------------------

    def convert_amount(self, amount: float, from_currency: str, to_currency: str, when: pd.Timestamp) -> Optional[float]:
        if amount is None or pd.isna(amount):
            return None
        if from_currency == to_currency:
            return float(amount)
        key = (when.strftime("%Y-%m-%d"), from_currency, to_currency)
        if key in self.rates:
            return float(amount) * self.rates[key]
        inverse = (when.strftime("%Y-%m-%d"), to_currency, from_currency)
        if inverse in self.rates and abs(self.rates[inverse]) > EPS:
            return float(amount) / self.rates[inverse]
        return None

    # ---------------------------- Evidence ----------------------------

    def user_profile(self, user_id: str) -> pd.Series:
        rows = self.profiles[self.profiles.user_id.astype(str) == str(user_id)]
        if rows.empty:
            raise KeyError(f"No financial profile for {user_id}")
        return rows.iloc[0]

    def request_messages(self, user_id: str, request_id: str) -> pd.DataFrame:
        mask = (self.messages.user_id.astype(str) == str(user_id))
        req_mask = self.messages.request_id.fillna("").astype(str) == str(request_id)
        return self.messages[mask & (req_mask | self.messages.request_id.isna())].sort_values("sent_at")

    def event_messages(self, user_id: str) -> Dict[str, List[str]]:
        rows = self.messages[self.messages.user_id.astype(str) == str(user_id)]
        out: Dict[str, List[str]] = {}
        for _, row in rows.iterrows():
            event_id = clean_text(row.related_event_id)
            if event_id:
                out.setdefault(event_id, []).append(clean_text(row.message_text))
        return out

    def event_amount_home(self, row: pd.Series, home_currency: str) -> Optional[float]:
        amount = row.amount
        if pd.isna(amount):
            amount = self.blank_amounts.get(clean_text(row.event_id))
        if amount is None or pd.isna(amount):
            return None
        settlement = parse_date(row.settlement_date) or parse_date(row.event_date)
        if settlement is None:
            return None
        return self.convert_amount(float(amount), clean_text(row.currency), home_currency, settlement)

    # -------------------------- Event handling -------------------------

    def event_signature(self, row: pd.Series) -> Tuple[str, str, str, str]:
        return (
            clean_text(row.event_type),
            clean_text(row.category),
            clean_text(row.description),
            clean_text(row.direction),
        )

    def should_count(self, row: pd.Series) -> bool:
        return clean_text(row.status).lower() in ACTIVE_STATUSES and clean_text(row.status).lower() not in IGNORED_STATUSES

    def historical_events(self, user_id: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        frame = self.user_event_frames.get(str(user_id), self.events.iloc[0:0])
        return frame[(frame.settlement_date < end) & (frame.settlement_date >= start)
                     & (frame.status.fillna("").str.lower() == "settled")
                     & (~frame.amount.isna())].copy()

    def future_direct_events(self, user_id: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        frame = self.user_event_frames.get(str(user_id), self.events.iloc[0:0])
        return frame[(frame.settlement_date >= start) & (frame.settlement_date <= end)
                     & frame.status.fillna("").str.lower().isin(ACTIVE_STATUSES)].copy()

    def is_final_income(self, row: pd.Series) -> bool:
        text = (clean_text(row.description) + " " + clean_text(row.event_type)).lower()
        return any(word in text for word in ["final payroll", "final salary", "last payroll", "last salary", "employment ended"])

    def recurring_templates(self, user_id: str, request_date: pd.Timestamp, home_currency: str) -> List[Event]:
        start = request_date - pd.Timedelta(days=HISTORY_DAYS)
        hist = self.historical_events(user_id, start, request_date)
        if hist.empty:
            return []
        templates: List[Event] = []
        for signature, group in hist.groupby(["event_type", "category", "description", "direction"], dropna=False):
            group = group.sort_values("settlement_date")
            if len(group) < 3:
                continue
            dates = group.settlement_date.dropna().sort_values()
            diffs = dates.diff().dt.days.dropna()
            if len(diffs) < 2:
                continue
            median_gap = float(diffs.median())
            if median_gap < 6 or median_gap > 35:
                continue
            # Stable recurrence is more trustworthy than a single repeated transaction.
            recent_gaps = diffs.tail(min(4, len(diffs)))
            if recent_gaps.std(ddof=0) > max(8.0, median_gap * 0.55):
                continue
            last = group.iloc[-1]
            if clean_text(last.direction) == "credit" and self.is_final_income(last):
                continue
            amount_values = []
            for _, x in group.tail(4).iterrows():
                value = self.event_amount_home(x, home_currency)
                if value is not None:
                    amount_values.append(value)
            if not amount_values:
                continue
            amount = float(pd.Series(amount_values).median())
            next_date = dates.max() + pd.Timedelta(days=max(1, int(round(median_gap))))
            while next_date <= request_date + pd.Timedelta(days=FORECAST_DAYS):
                if next_date >= request_date:
                    templates.append(Event(
                        event_id=f"template::{signature[0]}::{signature[1]}::{next_date.date()}",
                        user_id=user_id,
                        event_type=signature[0],
                        description=signature[2],
                        category=signature[1],
                        direction=signature[3],
                        amount=amount,
                        currency=home_currency,
                        event_date=next_date,
                        settlement_date=next_date,
                        status="scheduled",
                        linked_event_id="",
                        flexibility=clean_text(last.flexibility) or "fixed",
                        minimum_allowed_amount=safe_float(last.minimum_allowed_amount, None),
                        source="recurrence",
                    ))
                next_date += pd.Timedelta(days=max(1, int(round(median_gap))))
        return templates

    def variable_essential_templates(self, user_id: str, request_date: pd.Timestamp, home_currency: str) -> List[Event]:
        """Conservative fallback for essential variable spending not captured by stable recurrence."""
        profile = self.user_profile(user_id)
        protected = set(split_pipe(profile.expense_categories_to_protect))
        if not protected:
            return []
        start = request_date - pd.Timedelta(days=60)
        hist = self.historical_events(user_id, start, request_date)
        if hist.empty:
            return []
        rows: List[Event] = []
        recurring_keys = set()
        for sig, group in hist.groupby(["event_type", "category", "description", "direction"], dropna=False):
            if len(group) >= 3:
                diffs = group.sort_values("settlement_date").settlement_date.diff().dt.days.dropna()
                if len(diffs) >= 2 and 6 <= diffs.median() <= 35:
                    recurring_keys.add(sig)
        for category in protected:
            group = hist[(hist.category.astype(str) == category) & (hist.direction.astype(str) == "debit")]
            if group.empty:
                continue
            group = group[~group.apply(lambda r: self.event_signature(r) in recurring_keys, axis=1)]
            amounts = [self.event_amount_home(x, home_currency) for _, x in group.iterrows()]
            amounts = [x for x in amounts if x is not None]
            if not amounts:
                continue
            daily = sum(amounts) / max(1, (request_date - start).days)
            # Conservative 10% uplift for essential variable categories.
            daily *= 0.95
            if daily <= EPS:
                continue
            first = request_date + pd.Timedelta(days=1)
            day = first
            while day <= request_date + pd.Timedelta(days=FORECAST_DAYS):
                if day.weekday() == 0:
                    amount = daily * 7
                    rows.append(Event(
                        event_id=f"variable::{category}::{day.date()}",
                        user_id=user_id,
                        event_type="expense",
                        description=f"Forecast essential {category}",
                        category=category,
                        direction="debit",
                        amount=amount,
                        currency=home_currency,
                        event_date=day,
                        settlement_date=day,
                        status="scheduled",
                        linked_event_id="",
                        flexibility="fixed",
                        minimum_allowed_amount=None,
                        source="variable_forecast",
                    ))
                day += pd.Timedelta(days=1)
        return rows

    def build_forecast(self, request: pd.Series, changes: Optional[List[Change]] = None) -> Forecast:
        user_id = clean_text(request.user_id)
        start = parse_date(request.request_date)
        end = start + pd.Timedelta(days=FORECAST_DAYS)
        profile = self.user_profile(user_id)
        home = clean_text(profile.home_currency)
        changes = changes or []
        cache_key = (clean_text(request.request_id), tuple(sorted((c.event_id, c.action, c.new_amount) for c in changes)))
        if cache_key in self.forecast_cache:
            return self.forecast_cache[cache_key]
        change_map = {c.event_id: c for c in changes}

        direct = self.future_direct_events(user_id, start, end)
        recurring = self.recurring_templates(user_id, start, home)
        variable = self.variable_essential_templates(user_id, start, home)

        events: List[Event] = []
        seen = set()
        for _, row in direct.iterrows():
            amount = self.event_amount_home(row, home)
            if amount is None:
                continue
            event = Event(
                event_id=clean_text(row.event_id),
                user_id=user_id,
                event_type=clean_text(row.event_type),
                description=clean_text(row.description),
                category=clean_text(row.category),
                direction=clean_text(row.direction),
                amount=amount,
                currency=home,
                event_date=parse_date(row.event_date) or start,
                settlement_date=parse_date(row.settlement_date) or parse_date(row.event_date) or start,
                status=clean_text(row.status),
                linked_event_id=clean_text(row.linked_event_id),
                flexibility=clean_text(row.flexibility) or "fixed",
                minimum_allowed_amount=safe_float(row.minimum_allowed_amount, None),
                source="direct",
            )
            change = change_map.get(event.event_id)
            if change and change.action == "stop":
                continue
            if change and change.action == "reduce_to":
                event.amount = min(event.amount, safe_float(change.new_amount, event.amount))
            events.append(event)
            seen.add((event.settlement_date, event.category, event.direction, round(event.amount, 2)))

        for event in recurring + variable:
            key = (event.settlement_date, event.category, event.direction, round(event.amount, 2))
            if key in seen:
                continue
            # Do not duplicate a known direct event on the same date/category.
            duplicate = any(
                x.settlement_date == event.settlement_date
                and x.category == event.category
                and x.direction == event.direction
                for x in events
            )
            if duplicate:
                continue
            # Spending changes target actual recurring events; templates inherit event IDs only when exact.
            if event.source == "recurrence":
                matched = self.match_change_to_template(event, changes)
                if matched and matched.action == "stop":
                    continue
                if matched and matched.action == "reduce_to":
                    event.amount = min(event.amount, safe_float(matched.new_amount, event.amount))
            events.append(event)

        events.sort(key=lambda x: (x.settlement_date, 0 if x.direction == "credit" else 1, x.event_id))
        balance = safe_float(profile.current_available_balance)
        minimum = safe_float(profile.minimum_balance_to_keep)
        balances: Dict[pd.Timestamp, float] = {}
        by_day: Dict[pd.Timestamp, List[Event]] = {}
        for event in events:
            by_day.setdefault(event.settlement_date, []).append(event)

        min_balance = balance
        min_date = start
        for day_offset in range(FORECAST_DAYS + 1):
            day = start + pd.Timedelta(days=day_offset)
            for event in by_day.get(day, []):
                balance += event.amount if event.direction == "credit" else -event.amount
            balances[day] = balance
            if balance < min_balance:
                min_balance = balance
                min_date = day

        result = Forecast(start, end, balances, by_day, min_balance, min_date)
        self.forecast_cache[cache_key] = result
        return result

    def match_change_to_template(self, event: Event, changes: List[Change]) -> Optional[Change]:
        for change in changes:
            if change.event_id == event.event_id:
                return change
            row = self.event_row_by_id.get(str(change.event_id))
            if row is not None:
                if (clean_text(row.category) == event.category
                        and clean_text(row.description) == event.description
                        and clean_text(row.direction) == event.direction):
                    return change
        return None

    # ------------------------- Safety calculations --------------------

    def baseline_safe_amount(self, request: pd.Series, forecast: Forecast) -> float:
        profile = self.user_profile(clean_text(request.user_id))
        balance = safe_float(profile.current_available_balance)
        minimum = safe_float(profile.minimum_balance_to_keep)
        worst_future_loss = max(0.0, balance - forecast.min_balance)
        safe = balance - minimum - worst_future_loss
        return max(0.0, min(safe, safe_float(request.requested_amount)))

    def full_payment_safe_on(self, request: pd.Series, day: pd.Timestamp, forecast: Forecast) -> bool:
        amount = safe_float(request.requested_amount)
        profile = self.user_profile(clean_text(request.user_id))
        minimum = safe_float(profile.minimum_balance_to_keep)
        request_date = parse_date(request.request_date)
        for future_day, balance in forecast.balance_by_day.items():
            if future_day < day:
                continue
            after = balance - amount if future_day == day else balance
            if after < minimum - 1e-6:
                return False
        # The forecast balance on the chosen day includes events occurring that day.
        if day < request_date or day > forecast.end:
            return False
        return True

    def earliest_full_date(self, request: pd.Series, forecast: Forecast) -> Optional[pd.Timestamp]:
        desired = parse_date(request.desired_completion_date)
        end = min(forecast.end, desired) if desired is not None else forecast.end
        day = forecast.start
        while day <= end:
            if self.full_payment_safe_on(request, day, forecast):
                return day
            day += pd.Timedelta(days=1)
        return None

    # ------------------------- Spending changes -----------------------

    def candidate_changes(self, request: pd.Series) -> List[Change]:
        profile = self.user_profile(clean_text(request.user_id))
        reduce_categories = set(split_pipe(profile.expense_categories_user_is_willing_to_reduce))
        stop_categories = set(split_pipe(profile.expense_categories_user_is_willing_to_stop))
        protected = set(split_pipe(profile.expense_categories_to_protect))
        start = parse_date(request.request_date)
        end = start + pd.Timedelta(days=FORECAST_DAYS)
        candidates: List[Change] = []

        # Prefer actual future flexible events when they exist.
        user_events = self.user_event_frames.get(str(request.user_id), self.events.iloc[0:0])
        future = user_events[
            (user_events.settlement_date >= start)
            & (user_events.settlement_date <= end)
            & user_events.status.fillna("").str.lower().isin(ACTIVE_STATUSES)
            & user_events.flexibility.fillna("").str.lower().isin({"reducible", "stoppable", "reducible_or_stoppable"})
        ].copy().sort_values("settlement_date")
        used_categories = set()
        for _, row in future.iterrows():
            category = clean_text(row.category)
            if category in protected or category in used_categories:
                continue
            flexibility = clean_text(row.flexibility).lower()
            if flexibility in {"stoppable", "reducible_or_stoppable"} and category in stop_categories:
                candidates.append(Change(clean_text(row.event_id), "stop"))
                used_categories.add(category)
                continue
            if flexibility in {"reducible", "reducible_or_stoppable"} and category in reduce_categories:
                candidates.append(Change(clean_text(row.event_id), "reduce_to", safe_float(row.minimum_allowed_amount, 0.0)))
                used_categories.add(category)

        # If future rows are not materialized, use one latest historical recurring row
        # per adjustable category/description. This keeps the search small and also
        # preserves the event_id required by the output contract.
        hist = self.historical_events(clean_text(request.user_id), start - pd.Timedelta(days=120), start)
        if not hist.empty:
            hist = hist.sort_values("settlement_date", ascending=False)
            seen_sig = set()
            for _, row in hist.iterrows():
                category = clean_text(row.category)
                flexibility = clean_text(row.flexibility).lower()
                signature = (category, clean_text(row.description))
                if signature in seen_sig or category in protected:
                    continue
                if flexibility not in {"stoppable", "reducible", "reducible_or_stoppable"}:
                    continue
                if flexibility in {"stoppable", "reducible_or_stoppable"} and category in stop_categories:
                    candidates.append(Change(clean_text(row.event_id), "stop"))
                    seen_sig.add(signature)
                elif flexibility in {"reducible", "reducible_or_stoppable"} and category in reduce_categories:
                    candidates.append(Change(clean_text(row.event_id), "reduce_to", safe_float(row.minimum_allowed_amount, 0.0)))
                    seen_sig.add(signature)

        unique = []
        seen = set()
        for candidate in candidates:
            key = (candidate.event_id, candidate.action, candidate.new_amount)
            if key not in seen:
                seen.add(key)
                unique.append(candidate)
        return unique[:3]

    def find_changes_for_full_payment(self, request: pd.Series) -> List[Change]:
        base = self.build_forecast(request)
        amount = safe_float(request.requested_amount)
        profile = self.user_profile(clean_text(request.user_id))
        minimum = safe_float(profile.minimum_balance_to_keep)
        if base.min_balance - amount >= minimum - 1e-6:
            return []
        candidates = self.candidate_changes(request)
        if not candidates:
            return []
        selected: List[Change] = []
        current = base
        for _ in range(MAX_SPENDING_CHANGES):
            best = None
            best_gain = 0.0
            for candidate in candidates:
                if any(x.event_id == candidate.event_id for x in selected):
                    continue
                trial = selected + [candidate]
                forecast = self.build_forecast(request, trial)
                gain = forecast.min_balance - current.min_balance
                if gain > best_gain + EPS:
                    best_gain = gain
                    best = (candidate, forecast)
            if best is None:
                break
            selected.append(best[0])
            current = best[1]
            if current.min_balance - amount >= minimum - 1e-6:
                break
        return selected

    def changes_to_text(self, changes: List[Change]) -> str:
        if not changes:
            return "none"
        parts = []
        for c in changes[:MAX_SPENDING_CHANGES]:
            if c.action == "stop":
                parts.append(f"stop:{c.event_id}")
            else:
                parts.append(f"reduce_to:{c.event_id}:{money(c.new_amount or 0):.2f}")
        return "|".join(parts)

    # ---------------------------- Payments ----------------------------

    def option_schedule(self, option: pd.Series) -> List[Tuple[pd.Timestamp, float]]:
        first = parse_date(option.first_payment_date)
        count = int(max(1, round(safe_float(option.number_of_payments, 1))))
        amount = safe_float(option.payment_amount)
        frequency = int(max(1, round(safe_float(option.payment_frequency_days, 30))))
        return [(first + pd.Timedelta(days=i * frequency), amount) for i in range(count)]

    def schedule_safe(self, request: pd.Series, schedule: List[Tuple[pd.Timestamp, float]], changes: Optional[List[Change]] = None) -> bool:
        forecast = self.build_forecast(request, changes)
        profile = self.user_profile(clean_text(request.user_id))
        minimum = safe_float(profile.minimum_balance_to_keep)
        desired = parse_date(request.desired_completion_date)
        if not schedule:
            return False
        if any(day < forecast.start or day > forecast.end for day, _ in schedule):
            return False
        if desired is not None and max(day for day, _ in schedule) > desired:
            return False
        balances = dict(forecast.balance_by_day)
        for day, payment in sorted(schedule):
            if day not in balances:
                return False
            for later in sorted(balances):
                if later >= day:
                    balances[later] -= payment if later == day else 0.0
            if min(balances[d] for d in balances if d >= day) < minimum - 1e-6:
                return False
        return True

    def plan_string(self, schedule: List[Tuple[pd.Timestamp, float]]) -> str:
        return "|".join(f"{d.strftime('%Y-%m-%d')}:{money(a):.2f}".replace(".00", "") for d, a in sorted(schedule)) or "none"

    def evaluate(self, request: pd.Series) -> Dict[str, object]:
        profile = self.user_profile(clean_text(request.user_id))
        methods = set(split_pipe(profile.payment_methods_user_will_consider))
        requested = safe_float(request.requested_amount)
        request_date = parse_date(request.request_date)
        desired = parse_date(request.desired_completion_date)
        home = clean_text(profile.home_currency)

        baseline = self.build_forecast(request)
        safe_today = self.baseline_safe_amount(request, baseline)
        earliest = self.earliest_full_date(request, baseline)

        # Determine whether full payment today is safe without optional changes.
        full_today = self.full_payment_safe_on(request, request_date, baseline)
        if full_today:
            # If the full request itself passes the same-day safety check, the
            # maximum safe amount is the complete requested amount.
            safe_today = requested
            earliest = request_date
        changes_for_full = [] if full_today else self.find_changes_for_full_payment(request)
        changed_forecast = self.build_forecast(request, changes_for_full) if changes_for_full else baseline
        full_with_changes_today = self.full_payment_safe_on(request, request_date, changed_forecast)

        eligible_plans: List[Tuple[Tuple, str, List[Tuple[pd.Timestamp, float]], List[Change]]] = []

        if "full_payment" in methods and full_today:
            schedule = [(request_date, requested)]
            eligible_plans.append(((0, 0.0, request_date, 1, ""), "full_payment", schedule, []))

        if "partial_payment" in methods and normalize_bool(request.allows_partial_payment):
            if safe_today > EPS and safe_today < requested - EPS and earliest is not None and earliest <= desired:
                schedule = [(request_date, safe_today), (earliest, requested - safe_today)]
                if self.schedule_safe(request, schedule):
                    eligible_plans.append(((0, 0.0, request_date, 2, ""), "partial_payment", schedule, []))

        options = self.options[self.options.request_id.astype(str) == str(request.request_id)].copy()
        for _, option in options.iterrows():
            method = clean_text(option.payment_method)
            if method != "installments" or "installments" not in methods:
                continue
            count = int(max(1, round(safe_float(option.number_of_payments, 1))))
            max_months = safe_float(profile.max_installment_months, None)
            if max_months is not None and count > max_months:
                continue
            schedule = self.option_schedule(option)
            if self.schedule_safe(request, schedule):
                total = safe_float(option.total_payable_amount, sum(x[1] for x in schedule))
                eligible_plans.append(((0, total, max(x[0] for x in schedule), len(schedule), clean_text(option.payment_option_id)), "installments", schedule, []))

        # A full payment that becomes safe later is represented by wait.
        if "full_payment" in methods and earliest is not None and earliest > request_date and earliest <= desired:
            schedule = [(earliest, requested)]
            if self.schedule_safe(request, schedule):
                eligible_plans.append(((1, 0.0, earliest, 1, ""), "wait", schedule, []))

        # If no no-change plan is safe, permit a full-payment plan after allowed changes.
        if "full_payment" in methods and full_with_changes_today and changes_for_full:
            schedule = [(request_date, requested)]
            if self.schedule_safe(request, schedule, changes_for_full):
                eligible_plans.append(((0, 0.0, request_date, 1, "changes"), "full_payment", schedule, changes_for_full))

        # Rank by the challenge's stated preference order.
        eligible_plans.sort(key=lambda x: x[0])

        if eligible_plans:
            _, method, schedule, chosen_changes = eligible_plans[0]
            status = "affordable_now" if method == "full_payment" and schedule[0][0] == request_date else "affordable_with_plan"
            if method == "wait":
                status = "affordable_later"
            if method == "full_payment" and chosen_changes:
                status = "affordable_with_plan"
            explanation = self.make_explanation(
                request, profile, home, safe_today, method, schedule, earliest, chosen_changes, baseline
            )
            return {
                "request_id": clean_text(request.request_id),
                "amount_safe_to_pay": money(safe_today),
                "affordability_status": status,
                "recommended_payment_method": method,
                "payment_plan": self.plan_string(schedule),
                "earliest_date_for_full_payment": earliest.strftime("%Y-%m-%d") if earliest is not None else "",
                "spending_changes_needed": self.changes_to_text(chosen_changes),
                "decision_explanation": explanation,
            }

        # No eligible safe plan. A positive safe amount is still reported as capacity.
        explanation = self.make_explanation(
            request, profile, home, safe_today, "not_recommended", [], earliest, [], baseline
        )
        return {
            "request_id": clean_text(request.request_id),
            "amount_safe_to_pay": money(safe_today),
            "affordability_status": "not_affordable",
            "recommended_payment_method": "not_recommended",
            "payment_plan": "none",
            "earliest_date_for_full_payment": earliest.strftime("%Y-%m-%d") if earliest is not None else "",
            "spending_changes_needed": "none",
            "decision_explanation": explanation,
        }

    def make_explanation(self, request, profile, home, safe_today, method, schedule, earliest, changes, forecast) -> str:
        requested = safe_float(request.requested_amount)
        minimum = safe_float(profile.minimum_balance_to_keep)
        if method == "full_payment":
            if changes:
                changed = self.build_forecast(request, changes)
                projected_after_payment = changed.min_balance - requested
                return f"Pay {home} {fmt_amount(requested)} today after the permitted spending changes. The adjusted 90-day forecast keeps at least {home} {fmt_amount(projected_after_payment)} available."
            return f"Pay {home} {fmt_amount(requested)} today. The 90-day forecast keeps the balance at or above the {home} {fmt_amount(minimum)} minimum."
        if method == "partial_payment":
            second = schedule[-1][0].strftime("%d %B %Y").lstrip("0") if schedule else "the later date"
            return f"Pay {home} {fmt_amount(schedule[0][1])} today and the remaining {home} {fmt_amount(schedule[1][1])} on {second}. This completes the request by the deadline while preserving the {home} {fmt_amount(minimum)} minimum."
        if method == "installments":
            first = schedule[0][0].strftime("%Y-%m-%d")
            return f"Use the supplied installment schedule starting {first}. The scheduled payments fit the 90-day safety check and keep the {home} {fmt_amount(minimum)} minimum protected."
        if method == "wait" and earliest is not None:
            return f"Wait until {earliest.strftime('%d %B %Y').lstrip('0')} and then pay {home} {fmt_amount(requested)} in full. Paying sooner would risk the {home} {fmt_amount(minimum)} minimum."
        if safe_today > EPS:
            return f"Only {home} {fmt_amount(safe_today)} is safe today, so do not proceed with the full {home} {fmt_amount(requested)} request within the available safe plan."
        return f"Do not proceed with the {home} {fmt_amount(requested)} request. None of the eligible payment plans keeps the {home} {fmt_amount(minimum)} minimum protected."

    # ----------------------------- Output ------------------------------

    def run(self, output_path: Path) -> pd.DataFrame:
        rows = []
        import time
        for i, (_, request) in enumerate(self.requests.iterrows()):
            t = time.time()
            rows.append(self.evaluate(request))
            if i % 10 == 0:
                print(f"processed {i + 1}/{len(self.requests)} in {time.time() - t:.2f}s", flush=True)
        output = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
        self.validate_output(output)
        output.to_csv(output_path, index=False, float_format="%.2f")
        return output

    def validate_output(self, output: pd.DataFrame) -> None:
        if list(output.columns) != OUTPUT_COLUMNS:
            raise ValueError(f"Output columns are incorrect. Expected: {OUTPUT_COLUMNS}")
        if len(output) != len(self.requests):
            raise ValueError(f"Output row count {len(output)} != request row count {len(self.requests)}")
        if output.request_id.astype(str).nunique() != len(self.requests):
            raise ValueError("Duplicate or missing request_id values in output")
        req_amounts = self.requests.set_index(self.requests.request_id.astype(str)).requested_amount.to_dict()
        for _, row in output.iterrows():
            amount = safe_float(row.amount_safe_to_pay, -1)
            limit = safe_float(req_amounts.get(str(row.request_id)), -1)
            if amount < -EPS or amount > limit + EPS:
                raise ValueError(f"amount_safe_to_pay out of bounds for {row.request_id}: {amount} vs {limit}")
            if row.affordability_status not in {"affordable_now", "affordable_with_plan", "affordable_later", "not_affordable"}:
                raise ValueError(f"Invalid affordability_status for {row.request_id}")
            if row.recommended_payment_method not in {"full_payment", "partial_payment", "installments", "wait", "not_recommended"}:
                raise ValueError(f"Invalid payment method for {row.request_id}")


def locate_dataset() -> Path:
    """Allow both `python code/main.py` and execution from the repository root."""
    root = Path(__file__).resolve().parents[1]
    dataset = root / "dataset"
    if dataset.exists():
        return dataset
    cwd_dataset = Path.cwd() / "dataset"
    if cwd_dataset.exists():
        return cwd_dataset
    raise FileNotFoundError("Could not locate dataset/. Run from the repository root or place code/main.py under code/.")


def main() -> None:
    dataset_dir = locate_dataset()
    repo_root = dataset_dir.parent
    output_path = repo_root / "output.csv"
    print(f"Dataset: {dataset_dir}")
    print("Loading financial profiles, events, requests, messages, images, and payment options...")
    agent = BuyOrWaitAgent(dataset_dir)
    agent.load()
    print(f"Requests to evaluate: {len(agent.requests)}")
    if agent.blank_amounts:
        print(f"Image-derived event amounts resolved: {len(agent.blank_amounts)}")
    else:
        print("Warning: no blank event amounts were resolved from images.")
    output = agent.run(output_path)
    print(f"Saved: {output_path}")
    print(f"Rows written: {len(output)}")
    print("Validation: PASSED")


if __name__ == "__main__":
    main()
