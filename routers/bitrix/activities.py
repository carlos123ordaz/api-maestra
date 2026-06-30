import csv
from datetime import datetime

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from .shared import BITRIX_WEBHOOK, DATA_DIR, _fetch_users_batch

router = APIRouter(prefix="/bitrix")

ACTIVITIES_CSV = DATA_DIR / "bitrix_activities.csv"
ACTIVITIES_FIELDS = [
    "id", "type_id", "subject", "completed", "created", "end_time",
    "responsible_id", "responsible_name", "owner_type_id", "owner_id",
]

ACTIVITY_TYPE_MAP: dict[str, str] = {"1": "Reunion", "2": "Llamada", "8": "Visita"}


async def _fetch_all_activities() -> list[dict]:
    """Obtiene actividades CRM (OWNER_TYPE_ID=2) desde el 1 de enero del año anterior."""
    raw: list[dict] = []
    start = 0
    url = f"{BITRIX_WEBHOOK}/crm.activity.list"
    prev_year = datetime.now().year - 1
    cutoff = f"{prev_year}-01-01T00:00:00"

    async with httpx.AsyncClient(timeout=60) as client:
        while True:
            params = {
                "filter[OWNER_TYPE_ID]": "2",
                "filter[>CREATED]": cutoff,
                "select[]": [
                    "ID", "TYPE_ID", "SUBJECT", "COMPLETED", "CREATED",
                    "END_TIME", "RESPONSIBLE_ID", "OWNER_TYPE_ID", "OWNER_ID",
                ],
                "start": start,
            }
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            body = resp.json()
            page = body.get("result", [])
            raw.extend(page)
            total = body.get("total", 0)
            start += 50
            if start >= total:
                break

    user_ids = {str(a.get("RESPONSIBLE_ID", "")) for a in raw if a.get("RESPONSIBLE_ID")}
    user_map = await _fetch_users_batch(user_ids) if user_ids else {}

    normalized = []
    for a in raw:
        rid = str(a.get("RESPONSIBLE_ID", ""))
        normalized.append({
            "id":               str(a.get("ID", "")),
            "type_id":          str(a.get("TYPE_ID", "")),
            "subject":          a.get("SUBJECT", "") or "",
            "completed":        a.get("COMPLETED", "N") or "N",
            "created":          a.get("CREATED", "") or "",
            "end_time":         a.get("END_TIME", "") or "",
            "responsible_id":   rid,
            "responsible_name": user_map.get(rid, rid),
            "owner_type_id":    str(a.get("OWNER_TYPE_ID", "")),
            "owner_id":         str(a.get("OWNER_ID", "")),
        })
    return normalized


def _save_activities_csv(activities: list[dict]) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    with open(ACTIVITIES_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ACTIVITIES_FIELDS)
        writer.writeheader()
        writer.writerows(activities)


def _load_activities_csv() -> list[dict]:
    if not ACTIVITIES_CSV.exists():
        return []
    with open(ACTIVITIES_CSV, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _activities_response(activities: list[dict], source: str) -> dict:
    mtime = datetime.fromtimestamp(ACTIVITIES_CSV.stat().st_mtime).isoformat() \
        if ACTIVITIES_CSV.exists() else datetime.now().isoformat()
    return {
        "source":     source,
        "last_sync":  mtime,
        "total":      len(activities),
        "activities": activities,
    }


@router.get("/activities")
async def get_bitrix_activities():
    """Devuelve actividades CRM de Bitrix24. Usa caché CSV si existe."""
    rows = _load_activities_csv()
    if rows:
        return JSONResponse(content=_activities_response(rows, "cache"))
    activities = await _fetch_all_activities()
    _save_activities_csv(activities)
    return JSONResponse(content=_activities_response(activities, "bitrix"))


@router.post("/activities/sync")
async def sync_bitrix_activities():
    """Fuerza la sincronización de actividades CRM desde Bitrix24."""
    try:
        activities = await _fetch_all_activities()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"Error Bitrix24: {e}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
    _save_activities_csv(activities)
    return JSONResponse(content=_activities_response(activities, "bitrix"))
