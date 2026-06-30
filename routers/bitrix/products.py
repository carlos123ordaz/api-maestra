import csv
from datetime import datetime

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from .shared import BITRIX_WEBHOOK, DATA_DIR

router = APIRouter(prefix="/bitrix")

PRODUCTOS_CSV = DATA_DIR / "bitrix_productos.csv"
PRODUCTOS_FIELDS = [
    "id", "name", "code", "active",
    "section_id", "section_name",
    "parent_section_id", "parent_section_name",
]


async def _fetch_all_sections() -> dict[str, dict]:
    url = f"{BITRIX_WEBHOOK}/crm.productsection.list"
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(url, params={"start": 0})
        r.raise_for_status()
        items = r.json().get("result", [])
    return {
        s["ID"]: {"name": s["NAME"], "parent_id": s.get("SECTION_ID") or ""}
        for s in items
    }


async def _fetch_all_crm_products() -> list[dict]:
    products: list[dict] = []
    start = 0
    url = f"{BITRIX_WEBHOOK}/crm.product.list"
    async with httpx.AsyncClient(timeout=30) as client:
        while True:
            params = {
                "select[]": ["ID", "NAME", "CODE", "ACTIVE", "SECTION_ID"],
                "start": start,
            }
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            body = resp.json()
            page = body.get("result", [])
            products.extend(page)
            total = body.get("total", 0)
            start += 50
            if start >= total:
                break
    return products


def _save_productos_csv(products: list[dict], sections: dict[str, dict]) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    with open(PRODUCTOS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PRODUCTOS_FIELDS)
        writer.writeheader()
        for p in products:
            sid = p.get("SECTION_ID", "")
            sec = sections.get(sid, {})
            parent_id = sec.get("parent_id", "")
            parent_sec = sections.get(parent_id, {})
            writer.writerow({
                "id":                  p.get("ID", ""),
                "name":                p.get("NAME", ""),
                "code":                p.get("CODE", ""),
                "active":              p.get("ACTIVE", "N"),
                "section_id":          sid,
                "section_name":        sec.get("name", ""),
                "parent_section_id":   parent_id,
                "parent_section_name": parent_sec.get("name", ""),
            })


def _load_productos_csv() -> list[dict]:
    if not PRODUCTOS_CSV.exists():
        return []
    with open(PRODUCTOS_CSV, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _productos_response(products: list[dict], source: str) -> dict:
    mtime = datetime.fromtimestamp(PRODUCTOS_CSV.stat().st_mtime).isoformat() \
        if PRODUCTOS_CSV.exists() else datetime.now().isoformat()
    return {
        "source":    source,
        "last_sync": mtime,
        "total":     len(products),
        "productos": products,
    }


@router.get("/productos")
async def get_bitrix_productos(seccion: str = ""):
    """Devuelve los productos CRM de Bitrix24. Usa caché CSV si existe."""
    rows = _load_productos_csv()
    if not rows:
        sections = await _fetch_all_sections()
        products = await _fetch_all_crm_products()
        _save_productos_csv(products, sections)
        rows = _load_productos_csv()

    if seccion:
        q = seccion.lower()
        rows = [
            r for r in rows
            if q in (r.get("section_name") or "").lower()
            or q in (r.get("parent_section_name") or "").lower()
        ]

    return JSONResponse(content=_productos_response(rows, "cache"))


@router.post("/productos/sync")
async def sync_bitrix_productos():
    """Fuerza resincronización de productos CRM desde Bitrix24."""
    try:
        sections = await _fetch_all_sections()
        products = await _fetch_all_crm_products()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"Error Bitrix24: {e}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
    _save_productos_csv(products, sections)
    return JSONResponse(content=_productos_response(_load_productos_csv(), "bitrix"))
