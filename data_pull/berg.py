"""
Tiny, well‑organized Bloomberg API helper.

Covers: bdp, bds, bdh (historical), intraday bars/ticks, security search/field info,
        and lightweight real‑time subscriptions. Minimal, dependency‑light.

Usage examples are at the bottom under `if __name__ == "__main__":`.

Requires: blpapi, pandas
"""
from __future__ import annotations

import datetime as dt
import threading
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union, Callable

import blpapi  # type: ignore
import pandas as pd

# ────────────────────────────────────────────────────────────────────────────────
# Session management
# ────────────────────────────────────────────────────────────────────────────────
_REFDATA_SVC = "//blp/refdata"
_MKTDATA_SVC = "//blp/mktdata"
_INSTR_SVC   = "//blp/instruments"
_APIFLDS_SVC = "//blp/apiflds"

@dataclass
class BbgConfig:
    host: str = "localhost"
    port: int = 8194
    timeout_ms: int = 40000


class Bbg:
    """Singleton‑ish Bloomberg session wrapper with small, focused helpers."""

    def __init__(self, cfg: BbgConfig | None = None):
        self.cfg = cfg or BbgConfig()
        self._session: Optional[blpapi.Session] = None
        self._ensure_session()

    # — internals —
    def _ensure_session(self) -> blpapi.Session:
        if self._session:
            return self._session
        opts = blpapi.SessionOptions()
        opts.setServerHost(self.cfg.host)
        opts.setServerPort(self.cfg.port)
        sess = blpapi.Session(opts)
        if not sess.start():
            raise RuntimeError("Cannot start Bloomberg session (is Terminal open?)")
        self._session = sess
        return sess

    @property
    def session(self) -> blpapi.Session:
        return self._ensure_session()

    def _open_service(self, svcn: str) -> blpapi.Service:
        # Always try to open synchronously; if it fails, raise with a useful message.
        if not self.session.openService(svcn):
            raise RuntimeError(
                f"Cannot open service {svcn}. "
                "Ensure Bloomberg Terminal is running (Desktop API) or point to your B-PIPE host."
            )
        svc = self.session.getService(svcn)
        # Sanity: try a no-op call that would throw if handle is invalid.
        try:
            _ = svc.name()  # access something on the service to ensure it's valid
        except Exception as e:
            raise RuntimeError(f"Service handle invalid for {svcn}: {e!r}")
        return svc

    def _create_request(self, service: str, req_name: str) -> blpapi.Request:
        svc = self._open_service(service)
        # Try exact name first
        try:
            return svc.createRequest(req_name)
        except blpapi.exception.NotFoundException:
            # Fallback: scan available operations and try case variants
            ops = [svc.getOperation(i).name() for i in range(svc.numOperations())]
            # Common alias for instruments search:
            aliases = [req_name,
                       req_name[0].lower() + req_name[1:]]  # e.g., InstrumentListRequest -> instrumentListRequest
            for alt in aliases:
                if alt in ops:
                    return svc.createRequest(alt)
            raise RuntimeError(f"Operation '{req_name}' not available on {service}. Available: {ops}")

    # — generic request/response helper —
    def _send_request(
            self,
            service: str,
            req_name: str,
            build: Callable[[blpapi.Request], None]
    ) -> List[blpapi.Message]:
        svc = self._open_service(service)
        try:
            req = svc.createRequest(req_name)
        except blpapi.exception.NotFoundException:
            ops = [svc.getOperation(i).name() for i in range(svc.numOperations())]
            alt_name = req_name[0].lower() + req_name[1:]
            if alt_name in ops:
                req = svc.createRequest(alt_name)
            else:
                raise RuntimeError(
                    f"Operation '{req_name}' not available on {service}. "
                    f"Available operations: {ops}"
                )
        build(req)
        cid = blpapi.CorrelationId()
        self.session.sendRequest(req, correlationId=cid)
        msgs: List[blpapi.Message] = []
        start = dt.datetime.now()
        while True:
            ev = self.session.nextEvent(self.cfg.timeout_ms)
            et = ev.eventType()
            for msg in ev:
                cids = msg.correlationIds()
                if not cids:
                    continue
                if cids[0] == cid:
                    msgs.append(msg)
            if et == blpapi.Event.RESPONSE:  # final response
                break
            if et == blpapi.Event.PARTIAL_RESPONSE:
                continue
            elapsed_ms = (dt.datetime.now() - start).total_seconds() * 1000
            if elapsed_ms > self.cfg.timeout_ms * 3:
                raise TimeoutError(f"Bloomberg request timed out after {elapsed_ms / 1000:.1f}s")
        return msgs

    # ────────────────────────────────────────────────────────────────────────
    # Reference (BDP) and Bulk (BDS)
    # ────────────────────────────────────────────────────────────────────────
    def bdp(self, securities: Sequence[str], fields: Sequence[str], *, overrides: Dict[str, Any] | None = None) -> pd.DataFrame:
        """Point‑in‑time reference values.
        Returns DataFrame indexed by security, with columns=fields.
        """
        securities = list(securities)
        fields = list(fields)
        msgs = self._send_request(
            _REFDATA_SVC,
            "ReferenceDataRequest",
            lambda r: _build_ref_or_bulk(r, securities, fields, overrides)
        )
        return _parse_bdp_messages(msgs, fields)

    def bds(self, securities: Sequence[str], field: str, *, overrides: Dict[str, Any] | None = None) -> Dict[str, pd.DataFrame]:
        """Bulk field values (e.g., DES_BULK, IDX_MEMBERS). Returns dict per security -> DataFrame."""
        msgs = self._send_request(
            _REFDATA_SVC,
            "ReferenceDataRequest",
            lambda r: _build_ref_or_bulk(r, list(securities), [field], overrides)
        )
        return _parse_bds_messages(msgs, field)

    # ────────────────────────────────────────────────────────────────────────
    # Historical (BDH)
    # ────────────────────────────────────────────────────────────────────────
    def bdh(
        self,
        securities: Sequence[str],
        fields: Sequence[str],
        start: Union[str, dt.date],
        end: Union[str, dt.date],
        *,
        periodicity: str = "DAILY",
        adjustments: Optional[Dict[str, bool]] = None,
        overrides: Dict[str, Any] | None = None,
    ) -> Dict[str, pd.DataFrame]:
        """Historical time series. Returns dict: security -> DataFrame(date index, fields)."""
        start_s = _as_yyyymmdd(start)
        end_s = _as_yyyymmdd(end)

        def build(r: blpapi.Request) -> None:

            _append_many(r.getElement("securities"), securities)
            _append_many(r.getElement("fields"), fields)
            r.set("startDate", start_s)
            r.set("endDate", end_s)
            r.set("periodicitySelection", periodicity)
            if adjustments:
                for k, v in adjustments.items():
                    r.set(k, bool(v))
            if overrides:
                _apply_overrides(r, overrides)
        msgs = self._send_request(_REFDATA_SVC, "HistoricalDataRequest", build)
        return _parse_bdh_messages(msgs)

    # ────────────────────────────────────────────────────────────────────────
    # Intraday (bars & ticks)
    # ────────────────────────────────────────────────────────────────────────
    def intraday_bars(
            self,
            security: str,
            start: dt.datetime,
            end: dt.datetime,
            *,
            event_type: str = "TRADE",
            interval_minutes: int = 5,
            gap_fill: bool = False,
            adjustment_normal: bool = False,
            adjustment_abnormal: bool = False,
            adjustment_split: bool = False,
            adjustment_followDPDF: bool = False,
    ) -> pd.DataFrame:
        def build(r: blpapi.Request) -> None:
            r.set("security", security)
            r.set("eventType", event_type)
            r.set("interval", int(interval_minutes))  # ← was barInterval
            r.set("startDateTime", _to_bbg_dt(start))
            r.set("endDateTime", _to_bbg_dt(end))
            r.set("gapFillInitialBar", bool(gap_fill))
            r.set("adjustmentNormal", adjustment_normal)
            r.set("adjustmentAbnormal", adjustment_abnormal)
            r.set("adjustmentSplit", adjustment_split)
            r.set("adjustmentFollowDPDF", adjustment_followDPDF)

        msgs = self._send_request(_REFDATA_SVC, "IntradayBarRequest", build)
        return _parse_intraday_bars(msgs)

    def intraday_ticks(
        self,
        security: str,
        start: dt.datetime,
        end: dt.datetime,
        *,
        event_types: Sequence[str] = ("TRADE",),
        include_condition_codes: bool = False,
    ) -> pd.DataFrame:
        """Intraday tick data as a DataFrame (TIMESTAMP, TYPE, VALUE, SIZE, CC?)."""
        def build(r: blpapi.Request) -> None:
            r.set("security", security)
            et = r.getElement("eventTypes")
            for t in event_types: et.appendValue(t)
            r.set("startDateTime", _to_bbg_dt(start))
            r.set("endDateTime", _to_bbg_dt(end))
            r.set("includeConditionCodes", include_condition_codes)
        msgs = self._send_request(_REFDATA_SVC, "IntradayTickRequest", build)
        return _parse_intraday_ticks(msgs)

    # ────────────────────────────────────────────────────────────────────────
    # Discovery: security search & field info
    # ────────────────────────────────────────────────────────────────────────
    def security_search(self, query: str, *, max_results: int = 50) -> pd.DataFrame:
        def build(r: blpapi.Request) -> None:
            r.set("query", query)
            r.set("maxResults", int(max_results))
        msgs = self._send_request(_INSTR_SVC, "InstrumentListRequest", build)  # CHANGED service
        return _parse_security_search(msgs)

    def field_info(self, fields: Sequence[str]) -> pd.DataFrame:
        def build(r: blpapi.Request) -> None:
            _append_many(r.getElement("id"), list(fields))
            r.set("returnFieldDocumentation", True)  # ← ask for mnemonic/description/datatype/etc
        msgs = self._send_request(_APIFLDS_SVC, "FieldInfoRequest", build)
        return _parse_field_info(msgs)

    # ────────────────────────────────────────────────────────────────────────
    # Live subscriptions (very small helper)
    # ────────────────────────────────────────────────────────────────────────
    def subscribe(
        self,
        securities: Sequence[str],
        fields: Sequence[str],
        *,
        on_data: Callable[[str, Dict[str, Any], dt.datetime], None],
        on_status: Optional[Callable[[str, str], None]] = None,
    ) -> "LiveFeed":
        self._open_service(_MKTDATA_SVC)
        return LiveFeed(self.session, list(securities), list(fields), on_data, on_status)


# ────────────────────────────────────────────────────────────────────────────────
# Builders & parsers
# ────────────────────────────────────────────────────────────────────────────────
def _as_date_obj(x: Any) -> dt.date:
    # normalize blpapi.Datetime | datetime | date -> date
    if isinstance(x, dt.datetime):
        return x.date()
    if isinstance(x, dt.date):
        return x
    # fallback: try attributes (blpapi.Datetime-like)
    try:
        return dt.date(x.year(), x.month(), x.day())
    except Exception:
        return pd.to_datetime(x).date()


def _append_many(el, values):
    for v in values:
        el.appendValue(v)


def _build_ref_or_bulk(r: blpapi.Request, securities: List[str], fields: List[str], overrides: Dict[str, Any] | None) -> None:
    _append_many(r.getElement("securities"), list(securities))
    _append_many(r.getElement("fields"), list(fields))
    if overrides:
        _apply_overrides(r, overrides)


def _apply_overrides(r: blpapi.Request, overrides: Dict[str, Any]) -> None:
    o = r.getElement("overrides")
    for k, v in overrides.items():
        ov = o.appendElement()
        ov.setElement("fieldId", k)
        ov.setElement("value", v)


def _parse_errors(security_data_el: blpapi.Element) -> Optional[str]:
    if security_data_el.hasElement("securityError"):
        return str(security_data_el.getElement("securityError").getElement("message").getValue())
    return None


def _parse_bdp_messages(msgs: List[blpapi.Message], fields: List[str]) -> pd.DataFrame:
    rows = []
    for msg in msgs:
        data = msg.getElement("securityData")
        for i in range(data.numValues()):
            s = data.getValueAsElement(i)
            sec = s.getElementAsString("security")
            if err := _parse_errors(s):
                rows.append({"security": sec, "error": err})
                continue
            fvals = s.getElement("fieldData")
            row = {"security": sec}
            for f in fields:
                if fvals.hasElement(f):
                    row[f] = fvals.getElement(f).getValue()
            rows.append(row)
    df = pd.DataFrame(rows).set_index("security")
    if "error" in df.columns:
        # keep error column at the end
        cols = [c for c in df.columns if c != "error"] + ["error"]
        df = df[cols]
    return df


def _parse_bds_messages(msgs: List[blpapi.Message], field: str) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for msg in msgs:
        data = msg.getElement("securityData")
        for i in range(data.numValues()):
            s = data.getValueAsElement(i)
            sec = s.getElementAsString("security")
            if err := _parse_errors(s):
                out[sec] = pd.DataFrame({"error": [err]})
                continue
            fdata = s.getElement("fieldData")
            if fdata.hasElement(field):
                bulk = fdata.getElement(field)
                rows: List[Dict[str, Any]] = []
                for j in range(bulk.numValues()):
                    el = bulk.getValueAsElement(j)
                    row: Dict[str, Any] = {}
                    for k in range(el.numElements()):
                        sub = el.getElement(k)
                        row[sub.name()] = sub.getValue()
                    rows.append(row)
                out[sec] = pd.DataFrame(rows)
            else:
                out[sec] = pd.DataFrame()
    return out


def _parse_bdh_messages(msgs: List[blpapi.Message]) -> Dict[str, pd.DataFrame]:
    """
    Parse Bloomberg HistoricalDataResponse into:
        dict[security] -> DataFrame(index=date, columns=fields)
    """
    out: Dict[str, pd.DataFrame] = {}
    for msg in msgs:
        sec_data = msg.getElement("securityData")
        sec = sec_data.getElementAsString("security")
        # Security-level error
        if err := _parse_errors(sec_data):
            out[sec] = pd.DataFrame({"error": [err]})
            continue
        fds = sec_data.getElement("fieldData")
        rows: List[Dict[str, Any]] = []
        for i in range(fds.numValues()):
            el = fds.getValueAsElement(i)
            row: Dict[str, Any] = {}
            date_val: Optional[dt.date] = None
            for j in range(el.numElements()):
                sub = el.getElement(j)
                name = str(sub.name())  # force to plain string
                val = sub.getValue()
                if name.lower() == "date":
                    # Normalize date-ish things (blpapi.Datetime / datetime / date / string)
                    date_val = _as_date_obj(val)
                else:
                    row[name] = val
            # If we somehow didn't get a date for this row, skip it
            if date_val is None:
                continue
            row["date"] = date_val
            rows.append(row)
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.set_index("date").sort_index()
        out[sec] = df
    return out


def _parse_intraday_bars(msgs: List[blpapi.Message]) -> pd.DataFrame:
    rows: list[dict] = []

    for msg in msgs:
        if not msg.hasElement("barData"):
            continue
        bars = msg.getElement("barData").getElement("barTickData")

        for i in range(bars.numValues()):
            el = bars.getValueAsElement(i)
            bdt = el.getElementAsDatetime("time")

            # Convert Bloomberg datetime -> Python datetime, preserving time
            if isinstance(bdt, dt.datetime):
                # Already a real datetime with time; keep as-is (don’t strip tz here)
                t = bdt
            elif isinstance(bdt, dt.date):
                # Truly date-only (rare for intraday); no time info to preserve
                t = dt.datetime(bdt.year, bdt.month, bdt.day)
            else:
                # Assume blpapi.Datetime
                has_time = bdt.hasParts(blpapi.DatetimeParts.TIME)
                has_ms   = bdt.hasParts(blpapi.DatetimeParts.MILLISECONDS)

                hh = bdt.hour if has_time else 0
                mm = bdt.minute if has_time else 0
                ss = bdt.second if has_time else 0
                us = (bdt.milliSecond if has_ms else 0) * 1000

                t = dt.datetime(bdt.year, bdt.month, bdt.day, hh, mm, ss, us)

            rows.append({
                "time": t,
                "open": float(el.getElementAsFloat("open")),
                "high": float(el.getElementAsFloat("high")),
                "low":  float(el.getElementAsFloat("low")),
                "close": float(el.getElementAsFloat("close")),
                "volume": int(el.getElementAsInteger("volume")),
                "numEvents": int(el.getElementAsInteger("numEvents")),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Ensure proper dtype and ordering; do NOT normalize/floor the datetime
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time").sort_index()



def _parse_intraday_ticks(msgs: List[blpapi.Message]) -> pd.DataFrame:
    for msg in msgs:
        ticks = msg.getElement("tickData").getElement("tickData")
        rows = []
        for i in range(ticks.numValues()):
            el = ticks.getValueAsElement(i)
            t = el.getElementAsDatetime("time")
            if isinstance(t, dt.date):
                t = dt.datetime(t.year, t.month, t.day)
            else:
                t = t.replace(tzinfo=None)
            row = {
                "time": t,
                "type": el.getElementAsString("type"),
                "value": el.getElement("value").getValue(),
                "size": el.getElementAsInteger("size"),
            }
            if el.hasElement("conditionCodes"):
                row["conditionCodes"] = el.getElementAsString("conditionCodes")
            rows.append(row)
        df = pd.DataFrame(rows)
        return df.set_index("time") if not df.empty else df
    return pd.DataFrame()


def _parse_security_search(msgs: List[blpapi.Message]) -> pd.DataFrame:
    for msg in msgs:
        res = msg.getElement("results")
        rows = []
        for i in range(res.numValues()):
            el = res.getValueAsElement(i)
            rows.append({
                "security": el.getElementAsString("security"),
                "description": el.getElementAsString("description"),
                "yellowKey": el.getElementAsString("yellowKey") if el.hasElement("yellowKey") else None,
            })
        return pd.DataFrame(rows)
    return pd.DataFrame()


def _parse_field_info(msgs: List[blpapi.Message]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    def _row_from_el(el) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key in ("id", "mnemonic", "description", "datatype", "categoryName", "ftype"):
            if el.hasElement(key):
                out[key] = el.getElement(key).getValue()
        if not out:
            for i in range(el.numElements()):
                sub = el.getElement(i)
                try:
                    out[str(sub.name())] = sub.getValue()
                except Exception:
                    pass
        return out
    for msg in msgs:
        container = None
        for name in ("fieldData", "fieldInfo", "fieldResponse", "fieldResponses"):
            if msg.asElement().hasElement(name):
                container = msg.getElement(name)
                break
        if container is None:
            continue
        if hasattr(container, "numValues"):
            for i in range(container.numValues()):
                rows.append(_row_from_el(container.getValueAsElement(i)))
        else:
            rows.append(_row_from_el(container))

    return pd.DataFrame(rows)



# ────────────────────────────────────────────────────────────────────────────────
# Live feed helper
# ────────────────────────────────────────────────────────────────────────────────
class LiveFeed:
    """Very small helper that subscribes to real‑time mktdata in a background thread."""

    def __init__(
        self,
        session: blpapi.Session,
        securities: List[str],
        fields: List[str],
        on_data: Callable[[str, Dict[str, Any], dt.datetime], None],
        on_status: Optional[Callable[[str, str], None]] = None,
    ):
        self.session = session
        self.securities = securities
        self.fields = fields
        self.on_data = on_data
        self.on_status = on_status
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._subs: Optional[blpapi.SubscriptionList] = None

    def start(self) -> None:
        if self._running:
            return
        subs = blpapi.SubscriptionList()
        for s in self.securities:
            subs.add(s, self.fields, [], blpapi.CorrelationId(s))
        self._subs = subs                           # keep it
        self.session.subscribe(subs)
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self._running:
            return
        try:
            if self._subs is not None:
                self.session.unsubscribe(self._subs)
        finally:
            self._running = False
            if self._thread and self._thread.is_alive() and threading.current_thread() != self._thread:
                self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        while self._running:
            ev = self.session.nextEvent(500)
            et = ev.eventType()
            for msg in ev:
                cids = msg.correlationIds()
                sec = (cids[0].value() if cids else None)
                if isinstance(sec, (int, float)) or sec is None:
                    sec = str(sec)
                if et == blpapi.Event.SUBSCRIPTION_DATA:
                    payload: Dict[str, Any] = {}
                    for i in range(msg.numElements()):
                        el = msg.getElement(i)
                        try:
                            if hasattr(el, "isNull") and el.isNull():
                                continue
                        except Exception:
                            pass
                        name = str(el.name())
                        try:
                            dtp = el.datatype()  # blpapi.DataType
                            if dtp == blpapi.DataType.DATETIME:
                                v = el.getValueAsDatetime()
                                if isinstance(v, dt.date) and not isinstance(v, dt.datetime):
                                    v = dt.datetime(v.year, v.month, v.day)
                                else:
                                    v = v.replace(tzinfo=None)
                            elif dtp == blpapi.DataType.STRING:
                                v = el.getValueAsString()
                            elif dtp in (blpapi.DataType.FLOAT32, blpapi.DataType.FLOAT64):
                                v = el.getValueAsFloat()
                            elif dtp in (blpapi.DataType.INTEGER32, blpapi.DataType.INTEGER64):
                                v = el.getValueAsInteger()
                            elif dtp == blpapi.DataType.BOOL:
                                v = el.getValueAsBool()
                            else:
                                v = el.getValue()
                        except blpapi.exception.IndexOutOfRangeException:
                            continue
                        except Exception:
                            continue
                        payload[name] = v
                    if payload:
                        self.on_data(sec, payload, dt.datetime.now())

                elif et in (blpapi.Event.SUBSCRIPTION_STATUS, blpapi.Event.SESSION_STATUS):
                    if self.on_status:
                        self.on_status(sec or "", str(msg))

    # context manager sugar
    def __enter__(self):
        self.start(); return self
    def __exit__(self, exc_type, exc, tb):
        self.stop()


# ────────────────────────────────────────────────────────────────────────────────
# Small utilities
# ────────────────────────────────────────────────────────────────────────────────

def _as_yyyymmdd(x: Union[str, dt.date]) -> str:
    if isinstance(x, str):
        return x.replace("-", "")
    return x.strftime("%Y%m%d")


def _to_bbg_dt(t: dt.datetime):
    """Convert Python datetime to blpapi-compatible datetime."""
    if t.tzinfo is not None:
        t = t.astimezone(dt.timezone.utc).replace(tzinfo=None)
    # Bloomberg's Python API accepts a Python datetime directly
    return t


# ────────────────────────────────────────────────────────────────────────────────
# Examples
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    bbg = Bbg()

    # BDP: reference snapshot
    print("BDP example:\n", bbg.bdp(["IBM US Equity", "SPY US Equity"], ["PX_LAST", "CUR_MKT_CAP"]))

    # BDS: bulk field
    members = bbg.bds(["SPX Index"], "INDX_MEMBERS")
    print("\nBDS example (first rows):\n", members["SPX Index"].head())

    # BDH: historical
    h = bbg.bdh(["9127934G8 Govt"], ["PX_LAST"], start=dt.date(2025,1,1), end=dt.date(2025,3,31))
    print("\nBDH example:\n", h["9127934G8 Govt"].tail())

    # Intraday bars
    endt = dt.datetime.now().replace(microsecond=0) - dt.timedelta(minutes=1)
    startt = endt - dt.timedelta(hours=1)
    bars = bbg.intraday_bars("SPY US Equity", startt, endt, event_type="TRADE", interval_minutes=5)
    print("\nIntraday bars:\n", bars.tail())

    # Intraday ticks
    ticks = bbg.intraday_ticks("SPY US Equity", startt, endt, event_types=["TRADE", "BID", "ASK"])
    print("\nIntraday ticks (head):\n", ticks.head())

    # Discovery
    print("\nSecurity search for 'UST 10Y':\n", bbg.security_search("UST 10Y").head())
    print("\nField info for PX_* fields:\n", bbg.field_info(["PX_LAST", "PX_OPEN"]))

    # Live subscription (prints 5 updates then exits)
    ct = {"n": 0}
    def on_data(sec: str, data: Dict[str, Any], ts: dt.datetime) -> None:
        if {"BID", "ASK", "LAST_PRICE"} & set(data):
            print(f"{ts:%H:%M:%S} {sec} -> { {k:data[k] for k in data if k in ('BID','ASK','BID_SIZE','ASK_SIZE','LAST_PRICE')} }")
        ct["n"] += 1
        if ct["n"] >= 5:
            feed.stop()
    def on_status(sec: str, msg: str) -> None:
        print("[STATUS]", sec, msg)
    feed = bbg.subscribe(["TYA Index"], ["BID", "ASK", "BID_SIZE", "ASK_SIZE", "LAST_PRICE"], on_data=on_data, on_status=on_status)
    with feed:
        import time
        while feed._running:
            time.sleep(0.1)
