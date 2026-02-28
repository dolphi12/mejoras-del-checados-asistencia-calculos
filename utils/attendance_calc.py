from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, date
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import re

from utils.attendance_rules import AttendanceRules


TIME_FMT = "%H:%M"


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    s = str(x).strip()
    return "" if s.lower() in {"nan", "nat", "none"} else s


def parse_hhmm(s: str) -> Optional[Tuple[int, int]]:
    s = _safe_str(s)
    if not s:
        return None
    # Accept "YYYY-MM-DD HH:MM:SS" or "YYYY-MM-DDTHH:MM:SS"
    if len(s) >= 16 and (s[4] == "-" and (s[10] in (" ", "T"))):
        try:
            hh = int(s[11:13])
            mm = int(s[14:16])
            if 0 <= hh <= 23 and 0 <= mm <= 59:
                return hh, mm
        except Exception:
            return None
    # Accept "HH:MM"
    if len(s) >= 5 and s[2] == ":":
        try:
            hh = int(s[0:2])
            mm = int(s[3:5])
            if 0 <= hh <= 23 and 0 <= mm <= 59:
                return hh, mm
        except Exception:
            return None
    return None



def normalize_event_hhmm_list(times: List[str]) -> List[str]:
    """Normalize a list of event time strings to HH:MM and collapse consecutive duplicates.

    This is used to avoid placeholder/filler times (e.g., export columns filled with the last event)
    and to reduce repeated scans within the same minute.
    """
    out: List[str] = []
    prev: Optional[str] = None
    for t in (times or []):
        parsed = parse_hhmm(_safe_str(t))
        if not parsed:
            continue
        hh, mm = parsed
        norm = f"{hh:02d}:{mm:02d}"
        if prev is None or norm != prev:
            out.append(norm)
            prev = norm
    return out


def hhmm_to_minutes(s: str) -> Optional[int]:
    p = parse_hhmm(s)
    if not p:
        return None
    hh, mm = p
    return hh * 60 + mm


def minutes_to_hhmm(mins: int) -> str:
    mins = int(round(mins))
    if mins < 0:
        mins = 0
    hh = mins // 60
    mm = mins % 60
    return f"{hh:02d}:{mm:02d}"


def _parse_local_dt(s: str) -> Optional[datetime]:
    s = _safe_str(s)
    if not s:
        return None
    s2 = s.replace(" ", "T")
    try:
        return datetime.fromisoformat(s2)
    except Exception:
        return None


def _reconstruct_event_datetimes(
    start_local: Optional[datetime],
    event_hhmm: List[str],
) -> List[datetime]:
    """Rebuild a monotonic datetime series from HH:MM tokens.

    Uses `start_local.date()` as base. If a time goes backwards, assume next day.
    """
    if not start_local:
        # Fallback: assume today and keep monotonic by day wraps
        base = datetime.now().date()
        prev: Optional[datetime] = None
        out: List[datetime] = []
        day_offset = 0
        for t in event_hhmm:
            p = parse_hhmm(t)
            if not p:
                continue
            hh, mm = p
            dt = datetime(base.year, base.month, base.day, hh, mm) + timedelta(days=day_offset)
            if prev and dt < prev:
                day_offset += 1
                dt = dt + timedelta(days=1)
            out.append(dt)
            prev = dt
        return out

    base = start_local.date()
    prev: Optional[datetime] = None
    out: List[datetime] = []
    day_offset = 0
    for t in event_hhmm:
        p = parse_hhmm(t)
        if not p:
            continue
        hh, mm = p
        dt = datetime(base.year, base.month, base.day, hh, mm) + timedelta(days=day_offset)
        if prev and dt < prev:
            day_offset += 1
            dt = datetime(base.year, base.month, base.day, hh, mm) + timedelta(days=day_offset)
        out.append(dt)
        prev = dt
    return out


@dataclass
class AttendanceRow:
    employee_id: str
    fecha: str
    nombre: str

    entrada: str
    salida_comer: str
    regreso_comer: str
    salida_cenar: str
    regreso_cenar: str
    salida: str

    permiso_hhmm: str

    descuento_no_laborado_hhmm: str
    tiempo_trabajado_hhmm: str
    horas_extra_hhmm: str

    # Extras for UI/debug (not exported by default)
    _eventos: str = ""
    _alerta: str = ""


def compute_lunch_discount_minutes(lunch_out: Optional[datetime], lunch_in: Optional[datetime], rules: Optional[AttendanceRules] = None) -> Tuple[int, int]:
    """Return (lunch_duration, lunch_discount) according to configurable rules.

    Defaults preserve previous behavior:
      - If duration <= 60 minutes -> discount 30 minutes
      - If duration > 60 minutes -> discount full duration
    """
    rules = rules or AttendanceRules()
    if not lunch_out or not lunch_in:
        return 0, 0
    dur = int(round((lunch_in - lunch_out).total_seconds() / 60))
    if dur <= 0:
        return max(dur, 0), 0

    thr = max(int(rules.lunch_threshold_minutes), 0)
    fixed = max(int(rules.lunch_discount_minutes_if_within), 0)
    inclusive = bool(rules.lunch_threshold_inclusive)

    within = dur <= thr if inclusive else dur < thr

    mode = str(rules.lunch_mode or "").strip()
    if mode == "none":
        disc = 0
    elif mode == "full":
        disc = dur
    else:
        # fixed_if_within_else_full
        disc = fixed if within else dur

    return dur, max(int(disc), 0)


def compute_interval_minutes(a: Optional[datetime], b: Optional[datetime]) -> int:
    if not a or not b:
        return 0
    m = int(round((b - a).total_seconds() / 60))
    return max(m, 0)


_EDIT_KEY_CLEAN_RE = re.compile(r"[^A-Z0-9]+")


# Map various UI/editor keys to canonical internal keys.
_EDIT_KEY_MAP: Dict[str, str] = {
    # core
    "ENTRADA": "ENTRADA",
    "IN": "ENTRADA",
    "CHECKIN": "ENTRADA",

    "SALIDA": "SALIDA",
    "OUT": "SALIDA",
    "CHECKOUT": "SALIDA",

    # comida / lunch
    "SALIDA_A_COMER": "SALIDA_A_COMER",
    "SALIDA_COMER": "SALIDA_A_COMER",
    "SALIDA_A_LUNCH": "SALIDA_A_COMER",
    "LUNCH_OUT": "SALIDA_A_COMER",
    "COMIDA_OUT": "SALIDA_A_COMER",

    "REGRESO_DE_COMER": "REGRESO_DE_COMER",
    "REGRESO_COMER": "REGRESO_DE_COMER",
    "LUNCH_IN": "REGRESO_DE_COMER",
    "COMIDA_IN": "REGRESO_DE_COMER",

    # cena / dinner
    "SALIDA_A_CENAR": "SALIDA_A_CENAR",
    "SALIDA_CENAR": "SALIDA_A_CENAR",
    "DINNER_OUT": "SALIDA_A_CENAR",
    "CENA_OUT": "SALIDA_A_CENAR",

    "REGRESO_DE_CENAR": "REGRESO_DE_CENAR",
    "REGRESO_CENAR": "REGRESO_DE_CENAR",
    "DINNER_IN": "REGRESO_DE_CENAR",
    "CENA_IN": "REGRESO_DE_CENAR",

    # permisos
    "PERMISO": "PERMISO",
    "PERMISO_HHMM": "PERMISO",
    "PERMISOS": "PERMISO",

    "PERMISOS_DETALLE": "PERMISOS_DETALLE",
    "PERMISO_DETALLE": "PERMISOS_DETALLE",
    "PERMISO_DETAIL": "PERMISOS_DETALLE",
    "PERMISOS_DETAIL": "PERMISOS_DETALLE",
}


def _normalize_editable_times(editable_times: Optional[Dict[str, str]]) -> Dict[str, str]:
    """Normalize keys from different UIs (ENTRADA vs entrada vs lunch_out, etc.)."""
    out: Dict[str, str] = {}
    for k, v in (editable_times or {}).items():
        ks = _safe_str(k).upper()
        ks = _EDIT_KEY_CLEAN_RE.sub("_", ks).strip("_")
        ks = _EDIT_KEY_MAP.get(ks, ks)
        if not ks:
            continue
        out[ks] = _safe_str(v)
    return out


def _parse_op_date_to_date(op_date: str) -> date:
    """Parse a date string used by exports/UI into a `date`.

    Tries ISO first. If it can't parse, falls back to today's date (best effort).
    """
    s = _safe_str(op_date)
    if not s:
        return datetime.now().date()

    # Common shapes: "YYYY-MM-DD", "YYYY-MM-DD ..." (timestamp), etc.
    m = re.match(r"(\d{4}-\d{2}-\d{2})", s)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y-%m-%d").date()
        except Exception:
            pass

    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d/%m/%y", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except Exception:
            continue

    return datetime.now().date()


def _infer_start_local(
    op_date: str,
    event_times_hhmm: List[str],
    editable_times_norm: Dict[str, str],
) -> datetime:
    """Infer a stable start_local for historical recalculation.

    Uses:
      - op_date as the base day
      - first available time token (prefers ENTRADA if present)
    """
    base_d = _parse_op_date_to_date(op_date)

    first = _safe_str(editable_times_norm.get("ENTRADA"))
    if not first and event_times_hhmm:
        first = _safe_str(event_times_hhmm[0])

    p = parse_hhmm(first)
    if p:
        hh, mm = p
    else:
        hh, mm = 0, 0

    return datetime(base_d.year, base_d.month, base_d.day, hh, mm)


def compute_attendance_from_events(
    op_date: str,
    employee_id: str,
    employee_name: str,
    start_local: Optional[datetime] = None,
    end_local: Optional[datetime] = None,
    event_times_hhmm: List[str] = None,
    permiso_override_minutes: Optional[int] = None,
    editable_times: Optional[Dict[str, str]] = None,
    rules: Optional[AttendanceRules] = None,
    permiso_intervals_override: Optional[List[Tuple[str, str]]] = None,
) -> AttendanceRow:
    """Build AttendanceRow from exported jornada row.

    - `start_local` / `end_local` are optional for backwards compatibility.
    - `editable_times` can override individual HH:MM fields. Keys are normalized, so
      "entrada", "ENTRADA", "SALIDA A COMER", "lunch_out", etc. all work.
    """
    event_times_hhmm = event_times_hhmm or []
    editable_norm = _normalize_editable_times(editable_times)

    # --- Critical fix: if start_local is missing, rebuild it from FECHA instead of "today"
    if start_local is None:
        start_local = _infer_start_local(op_date, event_times_hhmm, editable_norm)

    events_dt = _reconstruct_event_datetimes(start_local, event_times_hhmm)
    eventos_str = ", ".join([dt.strftime(TIME_FMT) for dt in events_dt])

    alerta = ""
    if len(events_dt) < 2:
        alerta = "Menos de 2 eventos: no se puede calcular jornada completa."

    # Map events
    def _get_dt(i: int) -> Optional[datetime]:
        if i < 0 or i >= len(events_dt):
            return None
        return events_dt[i]

    entrada_dt = _get_dt(0)
    salida_dt = _get_dt(len(events_dt) - 1) if events_dt else None

    mid = events_dt[1:-1] if len(events_dt) > 2 else []
    lunch_out_dt = mid[0] if len(mid) >= 1 else None
    lunch_in_dt = mid[1] if len(mid) >= 2 else None
    dinner_out_dt = mid[2] if len(mid) >= 3 else None
    dinner_in_dt = mid[3] if len(mid) >= 4 else None

    # Permissions: events 6..(n-1) (1-indexed) => indices 5..(n-2) (0-indexed)
    permiso_minutes = 0
    perm = mid[4:] if len(mid) >= 5 else []
    if perm:
        for i in range(0, len(perm), 2):
            a = perm[i]
            b = perm[i + 1] if i + 1 < len(perm) else salida_dt
            permiso_minutes += compute_interval_minutes(a, b)

    # Apply editable overrides (HH:MM fields) on same reconstructed timeline
    # We keep dates aligned by placing edited times near entrada date and wrapping forward as needed.
    def _apply_override(base: Optional[datetime], new_hhmm: str) -> Optional[datetime]:
        if not base:
            return None
        p = parse_hhmm(new_hhmm)
        if not p:
            return base
        hh, mm = p
        dt = datetime(base.year, base.month, base.day, hh, mm)
        # ensure dt not earlier than base by wrapping
        while dt < base:
            dt += timedelta(days=1)
        return dt

    # NOTE: do NOT index directly like editable_times["ENTRADA"] to avoid KeyErrors/inconsistencies
    if entrada_dt and editable_norm.get("ENTRADA"):
        entrada_dt = _apply_override(entrada_dt, editable_norm.get("ENTRADA") or "")
    if entrada_dt and editable_norm.get("SALIDA"):
        # salida should be after entrada
        salida_dt = _apply_override(entrada_dt, editable_norm.get("SALIDA") or "")
    if entrada_dt and editable_norm.get("SALIDA_A_COMER"):
        lunch_out_dt = _apply_override(entrada_dt, editable_norm.get("SALIDA_A_COMER") or "")
    if lunch_out_dt and editable_norm.get("REGRESO_DE_COMER"):
        lunch_in_dt = _apply_override(lunch_out_dt, editable_norm.get("REGRESO_DE_COMER") or "")
    if entrada_dt and editable_norm.get("SALIDA_A_CENAR"):
        dinner_out_dt = _apply_override(entrada_dt, editable_norm.get("SALIDA_A_CENAR") or "")
    if dinner_out_dt and editable_norm.get("REGRESO_DE_CENAR"):
        dinner_in_dt = _apply_override(dinner_out_dt, editable_norm.get("REGRESO_DE_CENAR") or "")

    # Permission detail override from either explicit param or editable_times.
    permiso_detail_intervals: Optional[List[Tuple[str, str]]] = None
    if permiso_intervals_override:
        permiso_detail_intervals = permiso_intervals_override
    else:
        det_raw = _safe_str(editable_norm.get("PERMISOS_DETALLE"))
        if det_raw:
            permiso_detail_intervals = parse_permiso_intervals(det_raw)

    # Override permiso detail intervals if provided (advanced edit)
    if permiso_detail_intervals:
        permiso_minutes = 0
        # Interpret each (out, in) pair as a time interval; if "in" is missing, it will use salida.
        anchor = entrada_dt or (events_dt[0] if events_dt else start_local)
        if anchor:
            prev_base = anchor
            for out_s, in_s in permiso_detail_intervals:
                out_dt = _apply_override(prev_base, out_s) if _safe_str(out_s) else None
                if out_dt is None:
                    continue
                if _safe_str(in_s):
                    in_dt = _apply_override(out_dt, in_s)
                else:
                    in_dt = salida_dt
                if in_dt is None:
                    continue
                permiso_minutes += compute_interval_minutes(out_dt, in_dt)
                prev_base = out_dt

    # Override permiso total if provided (manual edit)
    if (not permiso_detail_intervals) and permiso_override_minutes is not None and permiso_override_minutes >= 0:
        permiso_minutes = int(permiso_override_minutes)

    # Back-compat: allow permiso total override via editable_times (e.g., editor sends "permiso_hhmm")
    if (not permiso_detail_intervals) and permiso_override_minutes is None:
        perm_h = _safe_str(editable_norm.get("PERMISO"))
        pm = hhmm_to_minutes(perm_h) if perm_h else None
        if pm is not None and pm >= 0:
            permiso_minutes = int(pm)

    lunch_dur, lunch_disc = compute_lunch_discount_minutes(lunch_out_dt, lunch_in_dt, rules=rules)
    dinner_dur = compute_interval_minutes(dinner_out_dt, dinner_in_dt)

    rules = rules or AttendanceRules()

    gross = compute_interval_minutes(entrada_dt, salida_dt)

    lunch_disc_eff = max(int(lunch_disc), 0) if rules.lunch_mode != "none" else 0
    dinner_disc_eff = max(int(dinner_dur), 0) if bool(rules.apply_dinner_deduction) else 0
    perm_disc_eff = max(int(permiso_minutes), 0) if bool(rules.apply_permissions_deduction) else 0

    descuento = lunch_disc_eff + dinner_disc_eff + perm_disc_eff
    if rules.clamp_negative_to_zero:
        net = max(gross - descuento, 0)
    else:
        net = gross - descuento

    if bool(rules.compute_overtime):
        thr = max(int(rules.overtime_threshold_minutes), 0)
        overtime = max(net - thr, 0)
    else:
        overtime = 0

    def _hhmm(dt: Optional[datetime]) -> str:
        return dt.strftime(TIME_FMT) if dt else ""

    row = AttendanceRow(
        employee_id=_safe_str(employee_id),
        fecha=_safe_str(op_date),
        nombre=_safe_str(employee_name),
        entrada=_hhmm(entrada_dt),
        salida_comer=_hhmm(lunch_out_dt),
        regreso_comer=_hhmm(lunch_in_dt),
        salida_cenar=_hhmm(dinner_out_dt),
        regreso_cenar=_hhmm(dinner_in_dt),
        salida=_hhmm(salida_dt),
        permiso_hhmm=minutes_to_hhmm(permiso_minutes),
        descuento_no_laborado_hhmm=minutes_to_hhmm(descuento),
        tiempo_trabajado_hhmm=minutes_to_hhmm(net),
        horas_extra_hhmm=minutes_to_hhmm(overtime),
        _eventos=eventos_str,
        _alerta=alerta,
    )
    return row


def build_attendance_table_from_export_xlsx(
    xlsx_path: str,
    sheet_name: str = "JORNADAS_CIERRE",
    rules: Optional[AttendanceRules] = None,
) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, dtype=str, engine="openpyxl")
    return build_attendance_table_from_export_df(df, rules=rules)


def build_attendance_table_from_export_df(df: pd.DataFrame, rules: Optional[AttendanceRules] = None) -> pd.DataFrame:
    # Normalize expected column names from collector export
    cols = {c: str(c).strip() for c in df.columns}
    df = df.rename(columns=cols)

    needed = ["fecha_registro", "employee_id", "employee_name", "start_local", "end_local"]
    for c in needed:
        if c not in df.columns:
            df[c] = ""

    ecols = [c for c in df.columns if re.match(r"^E\d\d$", str(c).strip())]
    ecols = sorted(ecols)

    out_rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        op_date = _safe_str(r.get("fecha_registro"))
        emp = _safe_str(r.get("employee_id"))
        name = _safe_str(r.get("employee_name"))
        start_dt = _parse_local_dt(_safe_str(r.get("start_local")))
        end_dt = _parse_local_dt(_safe_str(r.get("end_local")))

        ev = []
        for c in ecols:
            v = _safe_str(r.get(c))
            if v:
                # Keep only HH:MM tokens if present
                p = parse_hhmm(v)
                if p:
                    hh, mm = p
                    ev.append(f"{hh:02d}:{mm:02d}")

        ev = normalize_event_hhmm_list(ev)

        perm_detail = ""
        if len(ev) > 6:
            perms = ev[5:-1]
            pairs = []
            for k in range(0, len(perms), 2):
                a = perms[k]
                b = perms[k + 1] if k + 1 < len(perms) else ""
                if parse_hhmm(a) and (parse_hhmm(b) if b else True):
                    if b:
                        pairs.append(f"{a}-{b}")
                    else:
                        pairs.append(a)
            perm_detail = ";".join(pairs)

        perm_intervals = parse_permiso_intervals(perm_detail)

        row = compute_attendance_from_events(
            op_date=op_date,
            employee_id=emp,
            employee_name=name,
            start_local=start_dt,
            end_local=end_dt,
            event_times_hhmm=ev,
            rules=rules,
            permiso_intervals_override=perm_intervals if perm_intervals else None,
        )

        out_rows.append(
            {
                "ID": row.employee_id,
                "FECHA": row.fecha,
                "NOMBRE": row.nombre,
                "ENTRADA": row.entrada,
                "SALIDA A COMER": row.salida_comer,
                "REGRESO DE COMER": row.regreso_comer,
                "SALIDA A CENAR": row.salida_cenar,
                "REGRESO DE CENAR": row.regreso_cenar,
                "SALIDA": row.salida,
                "PERMISO": row.permiso_hhmm,
                "PERMISOS DETALLE": perm_detail,
                "DESCUENTO NO LABORADO": row.descuento_no_laborado_hhmm,
                "TIEMPO TRABAJADO": row.tiempo_trabajado_hhmm,
                "HORAS EXTRA": row.horas_extra_hhmm,
                "_EVENTOS": row._eventos,
                "_ALERTA": row._alerta,
            }
        )

    out = pd.DataFrame(out_rows)
    # Stable ordering
    if not out.empty:
        out = out.sort_values(["FECHA", "ID"], kind="stable")
    return out




def parse_permiso_intervals(s: str) -> List[Tuple[str, str]]:
    """Parse 'HH:MM-HH:MM;HH:MM-HH:MM' into list of (out,in)."""
    s = _safe_str(s)
    if not s:
        return []
    out: List[Tuple[str, str]] = []
    parts = [p.strip() for p in s.split(";") if p.strip()]
    for p in parts:
        if "-" in p:
            a, b = p.split("-", 1)
            a = a.strip()
            b = b.strip()
            if parse_hhmm(a) and parse_hhmm(b):
                out.append((a, b))
        else:
            # allow "HH:MM" single token (treated as open interval with no end)
            t = p.strip()
            if parse_hhmm(t):
                out.append((t, ""))
    return out


def autodetect_permiso_detail_from_eventos(eventos: str) -> str:
    """Auto-detect permission intervals from a string like "HH:MM, HH:MM, ...".

    Rule:
      - events[0] = entrada
      - events[1:3] = comida (out/in)
      - events[3:5] = cena (out/in)
      - events[-1] = salida
      - permissions = events[5:-1] paired (out/in)

    Returns "HH:MM-HH:MM;HH:MM-HH:MM". If a last permission has no pair, it returns "HH:MM".
    """
    eventos = _safe_str(eventos)
    if not eventos:
        return ""

    raw = [t.strip() for t in re.split(r"[;,]\s*|\s*,\s*", eventos) if t.strip()]
    times: List[str] = []
    for t in raw:
        p = parse_hhmm(t)
        if not p:
            continue
        hh, mm = p
        times.append(f"{hh:02d}:{mm:02d}")

    if len(times) <= 6:
        return ""

    perms = times[5:-1]
    parts: List[str] = []
    for i in range(0, len(perms), 2):
        a = perms[i]
        b = perms[i + 1] if i + 1 < len(perms) else ""
        if not parse_hhmm(a):
            continue
        if b and not parse_hhmm(b):
            b = ""
        parts.append(f"{a}-{b}" if b else a)
    return ";".join(parts)

def recalc_attendance_table(df: pd.DataFrame, rules: Optional[AttendanceRules] = None) -> pd.DataFrame:
    """Recalculate computed columns after user edits time fields.

    Expects df to contain at least these fields:
    ID, FECHA, NOMBRE, ENTRADA, SALIDA A COMER, REGRESO DE COMER, SALIDA A CENAR, REGRESO DE CENAR, SALIDA, PERMISO
    """
    out_rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        emp = _safe_str(r.get("ID"))
        op_date = _safe_str(r.get("FECHA"))
        name = _safe_str(r.get("NOMBRE"))
        # Build a minimal synthetic timeline anchored on op_date
        try:
            base = datetime.strptime(op_date, "%Y-%m-%d")
        except Exception:
            base = datetime.now()

        editable = {
            "ENTRADA": _safe_str(r.get("ENTRADA")),
            "SALIDA_A_COMER": _safe_str(r.get("SALIDA A COMER")),
            "REGRESO_DE_COMER": _safe_str(r.get("REGRESO DE COMER")),
            "SALIDA_A_CENAR": _safe_str(r.get("SALIDA A CENAR")),
            "REGRESO_DE_CENAR": _safe_str(r.get("REGRESO DE CENAR")),
            "SALIDA": _safe_str(r.get("SALIDA")),
        }

        perm_m = hhmm_to_minutes(_safe_str(r.get("PERMISO")))
        if perm_m is None:
            perm_m = 0

        perm_detail_raw = _safe_str(r.get("PERMISOS DETALLE") or r.get("_PERMISOS_DETALLE") or r.get("_PERMISOS_DETAIL"))
        perm_intervals = parse_permiso_intervals(perm_detail_raw)

        # Create pseudo events list for reference (entry..exit)
        ev = []
        for k in ["ENTRADA", "SALIDA A COMER", "REGRESO DE COMER", "SALIDA A CENAR", "REGRESO DE CENAR", "SALIDA"]:
            v = _safe_str(r.get(k))
            if parse_hhmm(v):
                ev.append(v)

        row = compute_attendance_from_events(
            op_date=op_date,
            employee_id=emp,
            employee_name=name,
            start_local=base,
            end_local=None,
            event_times_hhmm=ev,
            permiso_override_minutes=perm_m,
            editable_times=editable,
            rules=rules,
            permiso_intervals_override=perm_intervals if perm_intervals else None,
        )

        out_rows.append(
            {
                "ID": row.employee_id,
                "FECHA": row.fecha,
                "NOMBRE": row.nombre,
                "ENTRADA": row.entrada,
                "SALIDA A COMER": row.salida_comer,
                "REGRESO DE COMER": row.regreso_comer,
                "SALIDA A CENAR": row.salida_cenar,
                "REGRESO DE CENAR": row.regreso_cenar,
                "SALIDA": row.salida,
                "PERMISO": row.permiso_hhmm,
                "PERMISOS DETALLE": perm_detail_raw,
                "DESCUENTO NO LABORADO": row.descuento_no_laborado_hhmm,
                "TIEMPO TRABAJADO": row.tiempo_trabajado_hhmm,
                "HORAS EXTRA": row.horas_extra_hhmm,
                "_EVENTOS": _safe_str(r.get("_EVENTOS")),
                "_ALERTA": row._alerta,
            }
        )

    out = pd.DataFrame(out_rows)
    if not out.empty:
        out = out.sort_values(["FECHA", "ID"], kind="stable")
    return out

def recalc_attendance_rows(df: pd.DataFrame, keys: List[Tuple[str, str]], rules: Optional[AttendanceRules] = None) -> pd.DataFrame:
    """Recalculate only selected (ID, FECHA) rows.

    Designed for Streamlit's data_editor / grid editing:
    avoid recalculating the full table on each edit.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df
    if not keys:
        return df

    d2 = df.copy()
    rules = rules or AttendanceRules()

    # Ensure expected computed columns exist
    for c in ["DESCUENTO NO LABORADO", "TIEMPO TRABAJADO", "HORAS EXTRA", "_ALERTA"]:
        if c not in d2.columns:
            d2[c] = ""

    for emp_id, op_date in keys:
        emp_id_s = str(emp_id).strip()
        op_s = str(op_date).strip()
        if not emp_id_s or not op_s:
            continue

        mask = (d2.get("ID", "").astype(str) == emp_id_s) & (d2.get("FECHA", "").astype(str) == op_s)
        if not bool(mask.any()):
            continue

        # Stable anchor for historical dates (avoid "today" drift)
        try:
            base = datetime.strptime(op_s, "%Y-%m-%d")
        except Exception:
            base = datetime.now()

        for idx in d2[mask].index:
            row = d2.loc[idx]

            entrada = _safe_str(row.get("ENTRADA"))
            lunch_out = _safe_str(row.get("SALIDA A COMER"))
            lunch_in = _safe_str(row.get("REGRESO DE COMER"))
            dinner_out = _safe_str(row.get("SALIDA A CENAR"))
            dinner_in = _safe_str(row.get("REGRESO DE CENAR"))
            salida = _safe_str(row.get("SALIDA"))

            perm_detail = _safe_str(row.get("PERMISOS DETALLE"))
            permiso_total = _safe_str(row.get("PERMISO"))

            ev = [x for x in [entrada, lunch_out, lunch_in, dinner_out, dinner_in, salida] if x]
            ev = normalize_event_hhmm_list(ev)

            perm_m = hhmm_to_minutes(permiso_total) if permiso_total else None
            perm_intervals = parse_permiso_intervals(perm_detail) if perm_detail else []

            rec = compute_attendance_from_events(
                employee_id=emp_id_s,
                employee_name=_safe_str(row.get("NOMBRE")),
                op_date=op_s,
                start_local=base,
                end_local=None,
                event_times_hhmm=ev,
                rules=rules,
                permiso_override_minutes=perm_m if perm_m is not None else None,
                permiso_intervals_override=perm_intervals if perm_intervals else None,
                editable_times={
                    # send canonical keys; compute will also accept other shapes
                    "ENTRADA": entrada,
                    "SALIDA_A_COMER": lunch_out,
                    "REGRESO_DE_COMER": lunch_in,
                    "SALIDA_A_CENAR": dinner_out,
                    "REGRESO_DE_CENAR": dinner_in,
                    "SALIDA": salida,
                    "PERMISO": permiso_total,
                    "PERMISOS_DETALLE": perm_detail,
                },
            )

            d2.at[idx, "ENTRADA"] = rec.entrada
            d2.at[idx, "SALIDA A COMER"] = rec.salida_comer
            d2.at[idx, "REGRESO DE COMER"] = rec.regreso_comer
            d2.at[idx, "SALIDA A CENAR"] = rec.salida_cenar
            d2.at[idx, "REGRESO DE CENAR"] = rec.regreso_cenar
            d2.at[idx, "SALIDA"] = rec.salida
            d2.at[idx, "PERMISO"] = rec.permiso_hhmm
            d2.at[idx, "DESCUENTO NO LABORADO"] = rec.descuento_no_laborado_hhmm
            d2.at[idx, "TIEMPO TRABAJADO"] = rec.tiempo_trabajado_hhmm
            d2.at[idx, "HORAS EXTRA"] = rec.horas_extra_hhmm
            d2.at[idx, "_ALERTA"] = rec._alerta

    return d2
