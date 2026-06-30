import csv
import asyncio
from datetime import datetime

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from .shared import BITRIX_WEBHOOK, DATA_DIR, _fetch_users_batch
from .activities import _fetch_all_activities, _save_activities_csv

router = APIRouter(prefix="/bitrix")

DEALS_CSV = DATA_DIR / "bitrix_deals.csv"
DEALS_FIELDS = [
    "id", "title", "stage_id", "stage_name", "stage_semantic_id",
    "category_id", "category_name", "opportunity", "currency_id",
    "probability", "assigned_by_name", "date_create", "date_modify",
    "closedate", "source_id", "source_name", "is_closed",
    "alcance_pau", "alcance_pic", "dept_cisac", "company_id", "business_units",
]

PAU_MAP: dict[str, str]  = {"3904": "No llenado", "3906": "Si Incluye", "3908": "No incluye"}
PIC_MAP: dict[str, str]  = {"3898": "No llenado", "3900": "Si Incluye", "3902": "No incluye"}
CISAC_MAP: dict[str, str] = {"380": "Ventas", "382": "Servicios", "384": "Proyectos", "2087": "QHSE"}

UN_MAP: dict[str, str] = {
    "2033": "UNAU",
    "2037": "UNVA",
    "2035": "UNAI",
    "4552": "UNAP",
    "4734": "UNEPC",
    "3856": "PROSER",
    "3858": "PROSER",
    "3860": "PROSER",
}


def _map_cisac(raw) -> str:
    if not raw:
        return ""
    ids = raw if isinstance(raw, list) else [str(raw)]
    labels = [CISAC_MAP[str(i)] for i in ids if str(i) in CISAC_MAP]
    return ",".join(labels)


def _map_unidades(raw) -> str:
    if not raw:
        return ""
    ids = raw if isinstance(raw, list) else [str(raw)]
    seen: dict[str, None] = {}
    for i in ids:
        unit = UN_MAP.get(str(i))
        if unit:
            seen[unit] = None
    return ",".join(seen.keys())


async def _fetch_deal_categories() -> dict[str, str]:
    url = f"{BITRIX_WEBHOOK}/crm.dealcategory.list"
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(url)
            r.raise_for_status()
            items = r.json().get("result", [])
        result: dict[str, str] = {"0": "Pipeline Principal"}
        for cat in items:
            result[str(cat["ID"])] = cat["NAME"]
        return result
    except Exception:
        return {"0": "Pipeline Principal"}


async def _fetch_deal_stages(category_ids: set[str]) -> dict[str, dict]:
    stage_map: dict[str, dict] = {}
    async with httpx.AsyncClient(timeout=20) as client:
        for cat_id in category_ids:
            try:
                r = await client.get(
                    f"{BITRIX_WEBHOOK}/crm.dealcategory.stage.list",
                    params={"id": cat_id},
                )
                r.raise_for_status()
                for s in r.json().get("result", []):
                    stage_map[s["STATUS_ID"]] = {
                        "name":     s.get("NAME", s["STATUS_ID"]),
                        "semantic": s.get("SEMANTICS", ""),
                    }
            except Exception:
                pass
    return stage_map


async def _fetch_deal_sources() -> dict[str, str]:
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(
                f"{BITRIX_WEBHOOK}/crm.status.list",
                params={"FILTER[ENTITY_ID]": "SOURCE"},
            )
            r.raise_for_status()
            return {s["STATUS_ID"]: s["NAME"] for s in r.json().get("result", [])}
    except Exception:
        return {}


async def _fetch_all_deals() -> list[dict]:
    raw_deals: list[dict] = []
    start = 0
    url = f"{BITRIX_WEBHOOK}/crm.deal.list"

    async with httpx.AsyncClient(timeout=60) as client:
        while True:
            params = {
                "select[]": [
                    "ID", "TITLE", "STAGE_ID", "STAGE_SEMANTIC_ID",
                    "CURRENCY_ID", "OPPORTUNITY", "ASSIGNED_BY_ID",
                    "DATE_CREATE", "DATE_MODIFY", "CLOSEDATE",
                    "SOURCE_ID", "PROBABILITY", "CATEGORY_ID", "IS_CLOSED",
                    "UF_CRM_1714677550", "UF_CRM_1714677510",
                    "UF_CRM_1444152618", "COMPANY_ID", "UF_CRM_1579123522",
                ],
                "start": start,
            }
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            body = resp.json()
            page = body.get("result", [])
            raw_deals.extend(page)
            total = body.get("total", 0)
            start += 50
            if start >= total:
                break

    user_ids     = {str(d["ASSIGNED_BY_ID"]) for d in raw_deals if d.get("ASSIGNED_BY_ID")}
    category_ids = {str(d.get("CATEGORY_ID", "0")) for d in raw_deals} | {"0"}

    user_map, stage_map, cat_map, source_map = await asyncio.gather(
        _fetch_users_batch(user_ids),
        _fetch_deal_stages(category_ids),
        _fetch_deal_categories(),
        _fetch_deal_sources(),
    )

    normalized = []
    for d in raw_deals:
        stage_id   = d.get("STAGE_ID", "")
        stage_info = stage_map.get(stage_id, {})
        cat_id     = str(d.get("CATEGORY_ID", "0"))
        uid        = str(d.get("ASSIGNED_BY_ID", ""))
        src_id     = d.get("SOURCE_ID", "") or ""
        semantic   = d.get("STAGE_SEMANTIC_ID", "") or stage_info.get("semantic", "")
        normalized.append({
            "id":                str(d.get("ID", "")),
            "title":             d.get("TITLE", ""),
            "stage_id":          stage_id,
            "stage_name":        stage_info.get("name", stage_id),
            "stage_semantic_id": semantic,
            "category_id":       cat_id,
            "category_name":     cat_map.get(cat_id, f"Pipeline {cat_id}"),
            "opportunity":       d.get("OPPORTUNITY", "0") or "0",
            "currency_id":       d.get("CURRENCY_ID", "USD"),
            "probability":       d.get("PROBABILITY", "0") or "0",
            "assigned_by_name":  user_map.get(uid, uid),
            "date_create":       d.get("DATE_CREATE", "") or "",
            "date_modify":       d.get("DATE_MODIFY", "") or "",
            "closedate":         d.get("CLOSEDATE", "") or "",
            "source_id":         src_id,
            "source_name":       source_map.get(src_id, src_id),
            "is_closed":         d.get("IS_CLOSED", "N"),
            "alcance_pau":       PAU_MAP.get(str(d.get("UF_CRM_1714677550", "") or ""), "No llenado"),
            "alcance_pic":       PIC_MAP.get(str(d.get("UF_CRM_1714677510", "") or ""), "No llenado"),
            "dept_cisac":        _map_cisac(d.get("UF_CRM_1444152618")),
            "company_id":        str(d.get("COMPANY_ID", "") or ""),
            "business_units":    _map_unidades(d.get("UF_CRM_1579123522")),
        })
    return normalized


def _save_deals_csv(deals: list[dict]) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    with open(DEALS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DEALS_FIELDS)
        writer.writeheader()
        writer.writerows(deals)


def _load_deals_csv() -> list[dict]:
    if not DEALS_CSV.exists():
        return []
    with open(DEALS_CSV, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _deals_response(deals: list[dict], source: str) -> dict:
    mtime = datetime.fromtimestamp(DEALS_CSV.stat().st_mtime).isoformat() \
        if DEALS_CSV.exists() else datetime.now().isoformat()
    return {
        "source":    source,
        "last_sync": mtime,
        "total":     len(deals),
        "deals":     deals,
    }


@router.get("/deals")
async def get_bitrix_deals():
    """Devuelve los deals CRM de Bitrix24. Usa caché CSV si existe."""
    rows = _load_deals_csv()
    if rows:
        return JSONResponse(content=_deals_response(rows, "cache"))
    deals = await _fetch_all_deals()
    _save_deals_csv(deals)
    return JSONResponse(content=_deals_response(deals, "bitrix"))


@router.post("/deals/sync")
async def sync_bitrix_deals():
    """Fuerza la sincronización de deals y actividades CRM desde Bitrix24 en paralelo."""
    try:
        deals, activities = await asyncio.gather(
            _fetch_all_deals(),
            _fetch_all_activities(),
        )
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"Error Bitrix24: {e}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
    _save_deals_csv(deals)
    _save_activities_csv(activities)
    return JSONResponse(content=_deals_response(deals, "bitrix"))
