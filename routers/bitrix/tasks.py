import csv
import asyncio
from datetime import datetime

import httpx
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from .shared import BITRIX_WEBHOOK, DATA_DIR, _fetch_users_batch

router = APIRouter(prefix="/bitrix")

TASK_STATUS_MAP = {
    "1": "Nueva",
    "2": "Pendiente",
    "3": "En proceso",
    "4": "Por verificar",
    "5": "Completada",
    "6": "Rechazada",
    "7": "Diferida",
}
TASK_PRIORITY_MAP = {
    "0": "Baja",
    "1": "Normal",
    "2": "Alta",
}

CSV_FIELDS = [
    "id", "title", "status", "stage_id", "stage_name", "priority", "responsible_name",
    "created_by_name", "created_date", "deadline", "description",
]


def _csv_path(group_id: str):
    return DATA_DIR / f"bitrix_tareas_{group_id}.csv"


async def _fetch_group_name(group_id: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(
                f"{BITRIX_WEBHOOK}/sonet_group.get",
                params={"FILTER[ID]": group_id},
            )
            items = r.json().get("result", [])
            if items:
                return items[0].get("NAME", f"Grupo {group_id}")
    except Exception:
        pass
    return f"Grupo {group_id}"


async def _fetch_stages(group_id: str) -> list[dict]:
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(
                f"{BITRIX_WEBHOOK}/task.stages.get",
                params={"entityId": group_id, "isAdmin": "Y"},
            )
            result = r.json().get("result", {})
            items: list[dict] = []
            if isinstance(result, dict):
                for v in result.values():
                    items.append({
                        "id":    str(v.get("ID", "")),
                        "name":  v.get("TITLE", ""),
                        "color": v.get("COLOR", ""),
                        "sort":  int(v.get("SORT", 0)),
                    })
            elif isinstance(result, list):
                for v in result:
                    items.append({
                        "id":    str(v.get("ID", "")),
                        "name":  v.get("TITLE", ""),
                        "color": v.get("COLOR", ""),
                        "sort":  int(v.get("SORT", 0)),
                    })
            items.sort(key=lambda x: x["sort"])
            return items
    except Exception:
        pass
    return []


async def _fetch_all_bitrix_tasks(group_id: str):
    tasks: list[dict] = []
    start = 0
    url   = f"{BITRIX_WEBHOOK}/tasks.task.list"

    async with httpx.AsyncClient(timeout=30) as client:
        while True:
            params = {
                "filter[GROUP_ID]": group_id,
                "select[]": [
                    "ID", "TITLE", "STATUS", "STAGE_ID", "PRIORITY",
                    "RESPONSIBLE_ID", "CREATED_BY",
                    "CREATED_DATE", "DEADLINE", "DESCRIPTION",
                ],
                "start": start,
            }
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            body  = resp.json()
            page  = body.get("result", {}).get("tasks", [])
            tasks.extend(page)
            total = body.get("total", 0)
            start += 50
            if start >= total:
                break

    user_ids: set[str] = set()
    for t in tasks:
        if t.get("responsibleId"):
            user_ids.add(str(t["responsibleId"]))
        if t.get("createdBy"):
            user_ids.add(str(t["createdBy"]))

    async def _empty_map():
        return {}

    user_map, stages_list = await asyncio.gather(
        _fetch_users_batch(user_ids) if user_ids else _empty_map(),
        _fetch_stages(group_id),
    )
    stage_name_map = {s["id"]: s["name"] for s in stages_list}

    normalized = []
    for t in tasks:
        status_raw   = str(t.get("status", ""))
        priority_raw = str(t.get("priority", "1"))
        stage_id_raw = str(t.get("stageId", "") or "")
        rid = str(t.get("responsibleId", ""))
        cid = str(t.get("createdBy", ""))
        normalized.append({
            "id":               str(t.get("id", "")),
            "title":            t.get("title", ""),
            "status":           TASK_STATUS_MAP.get(status_raw, status_raw),
            "stage_id":         stage_id_raw,
            "stage_name":       stage_name_map.get(stage_id_raw, ""),
            "priority":         TASK_PRIORITY_MAP.get(priority_raw, priority_raw),
            "responsible_name": user_map.get(rid, rid),
            "created_by_name":  user_map.get(cid, cid),
            "created_date":     t.get("createdDate", ""),
            "deadline":         t.get("deadline", "") or "",
            "description":      (t.get("description") or "").strip(),
        })
    return normalized, stages_list


def _save_csv(tasks: list[dict], group_id: str) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    with open(_csv_path(group_id), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(tasks)


def _load_csv(group_id: str) -> list[dict]:
    p = _csv_path(group_id)
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _build_response(tasks, group_id, group_name, source, stages=None) -> dict:
    mtime = datetime.fromtimestamp(_csv_path(group_id).stat().st_mtime).isoformat() \
        if _csv_path(group_id).exists() else datetime.now().isoformat()
    clean_stages = [{"id": s["id"], "name": s["name"], "color": s.get("color", "")}
                    for s in (stages or [])]
    return {
        "source":     source,
        "group_id":   group_id,
        "group_name": group_name,
        "last_sync":  mtime,
        "total":      len(tasks),
        "stages":     clean_stages,
        "tareas":     tasks,
    }


@router.get("/tareas/{group_id}")
async def get_bitrix_tareas(group_id: str):
    """Devuelve las tareas de un grupo Bitrix24. Usa caché CSV si existe."""
    rows = _load_csv(group_id)
    if rows:
        group_name, stages = await asyncio.gather(
            _fetch_group_name(group_id),
            _fetch_stages(group_id),
        )
        return JSONResponse(content=_build_response(rows, group_id, group_name, "cache", stages))
    tasks, stages = await _fetch_all_bitrix_tasks(group_id)
    group_name    = await _fetch_group_name(group_id)
    _save_csv(tasks, group_id)
    return JSONResponse(content=_build_response(tasks, group_id, group_name, "bitrix", stages))


@router.get("/tareas/{group_id}/debug")
async def debug_bitrix_tareas(group_id: str):
    """Devuelve datos crudos de Bitrix24: etapas y primeras 5 tareas sin normalizar."""
    stages_raw: dict = {}
    tasks_raw: list  = []
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            sr = await client.get(
                f"{BITRIX_WEBHOOK}/task.stages.get",
                params={"entityId": group_id, "isAdmin": "Y"},
            )
            stages_raw = sr.json()

            tr = await client.get(
                f"{BITRIX_WEBHOOK}/tasks.task.list",
                params={
                    "filter[GROUP_ID]": group_id,
                    "select[]": ["ID", "TITLE", "STATUS", "STAGE_ID", "UF_KANBAN_STAGE_ID"],
                    "start": 0,
                },
            )
            body = tr.json()
            tasks_raw = body.get("result", {}).get("tasks", [])[:5]
    except Exception as e:
        return JSONResponse(content={"error": str(e)})
    return JSONResponse(content={"stages": stages_raw, "tasks_sample": tasks_raw})


@router.post("/tareas/{group_id}/sync")
async def sync_bitrix_tareas(group_id: str):
    """Fuerza la sincronización con Bitrix24 para un grupo y actualiza su CSV."""
    from fastapi import HTTPException
    try:
        tasks, stages = await _fetch_all_bitrix_tasks(group_id)
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"Error al consultar Bitrix24: {e}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
    group_name = await _fetch_group_name(group_id)
    _save_csv(tasks, group_id)
    return JSONResponse(content=_build_response(tasks, group_id, group_name, "bitrix", stages))
