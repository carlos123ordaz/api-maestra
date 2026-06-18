import os
import json
import io
import csv
import asyncio
import pathlib
from datetime import datetime
from typing import Optional

import fitz  # PyMuPDF
import httpx
import google.generativeai as genai
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from PIL import Image

# ── reportlab imports ─────────────────────────────────────────────────────────
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, HRFlowable,
)
from reportlab.lib.colors import HexColor

_ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(_ENV_PATH)

app = FastAPI(
    title="Extractor de Presupuesto",
    description="Extrae partidas e insumos desde imágenes o PDFs escaneados",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── prompts ───────────────────────────────────────────────────────────────────

EXTRACTION_PROMPT = """Extrae partidas de este presupuesto de construcción.

Columnas: CÓD.→codigo | PARTIDA→partida | UND→unidad | P.U.→precio_unitario | M.O.→mano_de_obra | MAT.→material | EQU.→equipo

edt_grupo: nombre textual de la sección (fila en negrita sin números que precede las partidas, ej: "MOVIMIENTO DE TIERRAS"). NUNCA el código numérico.

Reglas:
- Tablas de dos columnas lado a lado: extrae de ambas
- Omite encabezados de sección, títulos y totales; solo filas con UND y números
- Números: coma=miles punto=decimal (1,354.62→1354.62); vacío o guión→0
- Conserva asterisco en codigo; sin código→null

JSON únicamente, sin markdown:
{"partidas":[{"codigo":"OE.1.1.1.01","edt_grupo":"MOVIMIENTO DE TIERRAS","partida":"DESCRIPCIÓN","unidad":"M2","precio_unitario":104.70,"mano_de_obra":61.05,"material":40.60,"equipo":3.05}]}"""

LIBRO_PARTIDAS_PROMPT = """Extrae todas las partidas de construcción de este documento.

Columnas: CÓD.→codigo | PARTIDA→partida | UND→unidad | P.U.→precio_unitario | M.O.→mano_de_obra | MAT.→material | EQU.→equipo

edt_grupo: nombre textual del grupo/sección (fila en negrita sin números que precede las partidas, ej: "CIMIENTOS CORRIDOS"). NUNCA el código numérico. Sin sección→null. Sin código→null.

Reglas:
- Tablas de dos columnas lado a lado: extrae de ambas
- Omite encabezados, títulos de página y totales; solo filas con unidad y al menos un número
- Números: coma=miles punto=decimal (1,354.62→1354.62); vacío o guión→0; texto vacío→null
- Conserva asterisco en codigo si lo tiene

JSON únicamente, sin markdown:
{"partidas":[{"codigo":"OE.1.1.1.01","edt_grupo":"CIMIENTOS CORRIDOS","partida":"DESCRIPCIÓN","unidad":"M2","precio_unitario":104.70,"mano_de_obra":61.05,"material":40.60,"equipo":3.05}]}"""

LIBRO_INSUMOS_PROMPT = """Extrae todos los insumos/recursos de construcción de este documento.

Columnas: NOMBRE/Recurso/Insumo→nombre | CATEGORÍA/Tipo/Clase→categoria | UND→unidad | CANTIDAD→cantidad | P.U.→precio_unitario | TOTAL→total

Reglas:
- Tablas de dos columnas lado a lado: extrae de ambas
- Omite encabezados, títulos de página y totales generales; solo filas con al menos un número
- Números: coma=miles punto=decimal (1,354.62→1354.62); vacío o guión→0
- Sin columna CANTIDAD→usa 1; sin TOTAL→calcula cantidad×precio_unitario
- categoria: infiere del contexto del documento (ej: "Mano de Obra", "Material", "Equipo", "Subcontrato"); si no hay info→""

JSON únicamente, sin markdown:
{"insumos":[{"nombre":"CEMENTO PORTLAND TIPO I (42.5 KG)","unidad":"BOL","cantidad":1.00,"precio_unitario":25.00,"total":25.00}]}"""


# ── helpers ───────────────────────────────────────────────────────────────────

def get_model():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY no configurada en el entorno.")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        "gemini-2.5-flash-lite",
        generation_config=genai.GenerationConfig(max_output_tokens=65536),
    )


def parse_response(text: str | None) -> dict:
    if not text or not text.strip():
        raise ValueError("El modelo devolvió una respuesta vacía (posiblemente bloqueada por filtros de seguridad).")

    text = text.strip()

    # Strip markdown code block if present
    if text.startswith("```"):
        lines = text.split("\n")
        end_md = next((i for i in range(len(lines) - 1, 0, -1) if lines[i].strip() == "```"), len(lines))
        text = "\n".join(lines[1:end_md]).strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Extract JSON object by brace-matching (handles trailing text or minor truncation)
    start = text.find('{')
    if start == -1:
        raise ValueError("No se encontró JSON en la respuesta del modelo.")

    depth, json_end = 0, -1
    for i, ch in enumerate(text[start:], start):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                json_end = i + 1
                break

    if json_end == -1:
        raise ValueError(
            "La respuesta fue truncada antes de cerrar el JSON. "
            "El documento puede ser demasiado grande — intenta con menos páginas."
        )

    try:
        return json.loads(text[start:json_end])
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON malformado en la respuesta: {exc}") from exc


# Franjas horizontales por página — cada franja tiene ~50-60 ítems, bien dentro del límite de tokens.
# Subir el PDF completo al Files API no funciona para documentos con muchas partidas porque
# el modelo (gemini-2.5-flash-lite) usa parte del presupuesto de tokens en "thinking",
# dejando insuficiente espacio para el JSON completo.
STRIPS = 3   # franjas por página
SCALE  = 2.0 # resolución al renderizar (2× = mejor OCR)


def _render_page(pdf_bytes: bytes, page_num: int) -> Image.Image:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pix = doc[page_num].get_pixmap(matrix=fitz.Matrix(SCALE, SCALE))
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img


def _call_image(model, prompt: str, img: Image.Image) -> dict:
    response = model.generate_content([prompt, img])
    raw = response.text if response.candidates else None
    return parse_response(raw)


def _extract_strips(model, img: Image.Image, prompt: str) -> dict:
    """Divide una imagen en STRIPS franjas horizontales y extrae cada una por separado."""
    all_partidas: list = []
    all_insumos:  list = []
    strip_h = img.height // STRIPS
    for s in range(STRIPS):
        top    = s * strip_h
        bottom = img.height if s == STRIPS - 1 else (s + 1) * strip_h
        strip  = img.crop((0, top, img.width, bottom))
        try:
            result = _call_image(model, prompt, strip)
            all_partidas.extend(result.get("partidas", []))
            all_insumos.extend(result.get("insumos",  []))
        except ValueError:
            pass  # franja en blanco o sin ítems válidos
    return {"partidas": all_partidas, "insumos": all_insumos}


def run_extraction(model, pdf_bytes: bytes | None, image: Image.Image | None, prompt: str) -> dict:
    """Extrae datos renderizando cada página como imagen y procesándola en franjas."""
    if pdf_bytes is not None:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages = len(doc)
        doc.close()
        all_partidas: list = []
        all_insumos:  list = []
        for page_num in range(total_pages):
            img    = _render_page(pdf_bytes, page_num)
            result = _extract_strips(model, img, prompt)
            all_partidas.extend(result.get("partidas", []))
            all_insumos.extend(result.get("insumos",  []))
        return {"partidas": all_partidas, "insumos": all_insumos}
    else:
        return _extract_strips(model, image, prompt)


def detect_file(filename: str, content_type: str):
    nombre = (filename or "").lower()
    ct = (content_type or "").lower()
    es_pdf = nombre.endswith(".pdf") or "pdf" in ct
    es_img = (
        any(nombre.endswith(e) for e in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"))
        or "image" in ct
    )
    return es_pdf, es_img


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "servicio": "Extractor de Presupuesto v2",
        "endpoints": {
            "POST /extraer":                "Extrae partidas de un presupuesto",
            "POST /extraer/libro/partidas": "Extrae solo partidas de un libro",
            "POST /extraer/libro/insumos":  "Extrae solo insumos de un libro",
            "GET  /health":                 "Estado del servicio",
        },
    }


@app.get("/health")
def health():
    return {"status": "ok", "api_key_configurada": bool(os.getenv("GEMINI_API_KEY"))}


@app.post("/extraer")
async def extraer_presupuesto(archivo: UploadFile = File(...)):
    """Extrae partidas de un presupuesto escaneado (PDF o imagen)."""
    model = get_model()
    contenido = await archivo.read()
    es_pdf, es_img = detect_file(archivo.filename or "", archivo.content_type or "")

    if es_pdf:
        resultado = run_extraction(model, contenido, None, EXTRACTION_PROMPT)
    elif es_img:
        try:
            img = Image.open(io.BytesIO(contenido)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"No se pudo abrir la imagen: {e}")
        resultado = run_extraction(model, None, img, EXTRACTION_PROMPT)
    else:
        raise HTTPException(status_code=415, detail=f"Tipo no soportado: '{archivo.content_type}'. Usa PDF, JPG o PNG.")

    partidas = resultado.get("partidas", [])
    return JSONResponse(content={"archivo": archivo.filename, "total_partidas": len(partidas), "partidas": partidas})


async def _extraer_con_prompt(archivo: UploadFile, prompt: str) -> tuple[bytes, Image.Image | None]:
    contenido = await archivo.read()
    es_pdf, es_img = detect_file(archivo.filename or "", archivo.content_type or "")
    if es_pdf:
        return contenido, None
    if es_img:
        try:
            return b"", Image.open(io.BytesIO(contenido)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"No se pudo abrir la imagen: {e}")
    raise HTTPException(status_code=415, detail=f"Tipo no soportado: '{archivo.content_type}'. Usa PDF, JPG o PNG.")


@app.post("/extraer/libro/partidas")
async def extraer_libro_partidas(archivo: UploadFile = File(...)):
    """Extrae SOLO partidas desde un libro de precios (PDF o imagen)."""
    model = get_model()
    pdf_bytes, img = await _extraer_con_prompt(archivo, LIBRO_PARTIDAS_PROMPT)
    resultado = run_extraction(model, pdf_bytes if pdf_bytes else None, img, LIBRO_PARTIDAS_PROMPT)
    partidas = resultado.get("partidas", [])
    return JSONResponse(content={
        "archivo":        archivo.filename,
        "tipo":           "partidas",
        "total_partidas": len(partidas),
        "partidas":       partidas,
    })


# ── Bitrix24 ─────────────────────────────────────────────────────────────────

BITRIX_WEBHOOK = "https://corsusaint.bitrix24.com/rest/6238/dbakwyqx9fxrblp1"
DATA_DIR       = pathlib.Path(__file__).parent / "data"

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


def _csv_path(group_id: str) -> pathlib.Path:
    return DATA_DIR / f"bitrix_tareas_{group_id}.csv"


async def _fetch_group_name(group_id: str) -> str:
    """Intenta obtener el nombre del grupo de trabajo desde Bitrix24."""
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
    """Obtiene las etapas (kanban stages) de un grupo Bitrix24, ordenadas por SORT.
    Retorna lista de {id, name, color}."""
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


async def _fetch_all_bitrix_tasks(group_id: str) -> list[dict]:
    """Obtiene todas las tareas de un grupo Bitrix24 paginando de 50 en 50."""
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

    # Resolver nombres de usuarios y etapas en paralelo
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
    # Build id→name map from stages list
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


def _build_response(tasks: list[dict], group_id: str, group_name: str, source: str,
                    stages: list[dict] | None = None) -> dict:
    mtime = datetime.fromtimestamp(_csv_path(group_id).stat().st_mtime).isoformat() \
        if _csv_path(group_id).exists() else datetime.now().isoformat()
    # Strip internal 'sort' key before sending to client
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


@app.get("/bitrix/tareas/{group_id}")
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


@app.get("/bitrix/tareas/{group_id}/debug")
async def debug_bitrix_tareas(group_id: str):
    """Devuelve datos crudos de Bitrix24: etapas y primeras 3 tareas sin normalizar."""
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


@app.post("/bitrix/tareas/{group_id}/sync")
async def sync_bitrix_tareas(group_id: str):
    """Fuerza la sincronización con Bitrix24 para un grupo y actualiza su CSV."""
    try:
        tasks, stages = await _fetch_all_bitrix_tasks(group_id)
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"Error al consultar Bitrix24: {e}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
    group_name = await _fetch_group_name(group_id)
    _save_csv(tasks, group_id)
    return JSONResponse(content=_build_response(tasks, group_id, group_name, "bitrix", stages))


# ── Bitrix24 Productos CRM ────────────────────────────────────────────────────

PRODUCTOS_CSV = DATA_DIR / "bitrix_productos.csv"
PRODUCTOS_FIELDS = [
    "id", "name", "code", "active",
    "section_id", "section_name",
    "parent_section_id", "parent_section_name",
]


async def _fetch_all_sections() -> dict[str, dict]:
    """Devuelve mapa section_id → {name, parent_id} con todos los niveles."""
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
    """Obtiene todos los productos CRM paginando de 50 en 50."""
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


@app.get("/bitrix/productos")
async def get_bitrix_productos(seccion: str = ""):
    """Devuelve los productos CRM de Bitrix24. Usa caché CSV si existe.

    Params:
      seccion: filtra filas donde section_name o parent_section_name contengan este texto (case-insensitive).
               Ej: ?seccion=Proy
    """
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


@app.post("/bitrix/productos/sync")
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


# ── Bitrix24 CRM Deals ────────────────────────────────────────────────────────

DEALS_CSV = DATA_DIR / "bitrix_deals.csv"
DEALS_FIELDS = [
    "id", "title", "stage_id", "stage_name", "stage_semantic_id",
    "category_id", "category_name", "opportunity", "currency_id",
    "probability", "assigned_by_name", "date_create", "date_modify",
    "closedate", "source_id", "source_name", "is_closed",
    "alcance_pau", "alcance_pic", "dept_cisac",
]

PAU_MAP: dict[str, str]  = {"3904": "No llenado", "3906": "Si Incluye", "3908": "No incluye"}
PIC_MAP: dict[str, str]  = {"3898": "No llenado", "3900": "Si Incluye", "3902": "No incluye"}
CISAC_MAP: dict[str, str] = {"380": "Ventas", "382": "Servicios", "384": "Proyectos", "2087": "QHSE"}


def _map_cisac(raw) -> str:
    """Convierte el campo múltiple UF_CRM_1444152618 a una cadena 'Ventas,Proyectos'."""
    if not raw:
        return ""
    ids = raw if isinstance(raw, list) else [str(raw)]
    labels = [CISAC_MAP[str(i)] for i in ids if str(i) in CISAC_MAP]
    return ",".join(labels)


async def _fetch_deal_categories() -> dict[str, str]:
    """Retorna mapa category_id → name (incluye '0' para el pipeline por defecto)."""
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
    """Retorna mapa stage_id → {name, semantic} para todas las categorías."""
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
    """Retorna mapa source_id → name."""
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
    """Obtiene todos los deals CRM de Bitrix24 paginando de 50 en 50."""
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
                    "UF_CRM_1444152618",
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

    # IDs únicos para lookups paralelos
    user_ids     = {str(d["ASSIGNED_BY_ID"]) for d in raw_deals if d.get("ASSIGNED_BY_ID")}
    category_ids = {str(d.get("CATEGORY_ID", "0")) for d in raw_deals} | {"0"}

    # Lookups en paralelo
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
        })
    return normalized


async def _fetch_users_batch(user_ids: set[str]) -> dict[str, str]:
    """Retorna mapa user_id → nombre completo (reutilizable)."""
    user_map: dict[str, str] = {}
    async with httpx.AsyncClient(timeout=30) as client:
        for uid in user_ids:
            try:
                r = await client.get(
                    f"{BITRIX_WEBHOOK}/user.get",
                    params={"filter[ID]": uid},
                )
                data = r.json().get("result", [])
                if data:
                    u = data[0]
                    user_map[uid] = f"{u.get('NAME', '')} {u.get('LAST_NAME', '')}".strip()
            except Exception:
                pass
    return user_map


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


@app.get("/bitrix/deals")
async def get_bitrix_deals():
    """Devuelve los deals CRM de Bitrix24. Usa caché CSV si existe."""
    rows = _load_deals_csv()
    if rows:
        return JSONResponse(content=_deals_response(rows, "cache"))
    deals = await _fetch_all_deals()
    _save_deals_csv(deals)
    return JSONResponse(content=_deals_response(deals, "bitrix"))


@app.post("/bitrix/deals/sync")
async def sync_bitrix_deals():
    """Fuerza la sincronización de deals CRM desde Bitrix24."""
    try:
        deals = await _fetch_all_deals()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"Error Bitrix24: {e}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
    _save_deals_csv(deals)
    return JSONResponse(content=_deals_response(deals, "bitrix"))


# ══════════════════════════════════════════════════════════════════════════════
# PDF Generation — Cotizaciones
# ══════════════════════════════════════════════════════════════════════════════
from reportlab.pdfgen import canvas as pdf_canvas

class DisciplinaLinea(BaseModel):
    nombre: str
    precio_venta: float

class CotizacionPDFRequest(BaseModel):
    numero: str
    titulo: str
    cliente: Optional[str] = None
    proyecto: Optional[str] = None
    ubicacion: Optional[str] = None
    descripcion: Optional[str] = None
    tipo_cambio: float = 3.5
    plazo_semanas: Optional[int] = None
    garantia_meses: int = 12
    validez_dias: int = 30
    forma_pago: Optional[str] = None
    disciplinas: list[DisciplinaLinea] = []
    precio_gg: float = 0.0


def _fmt_usd(v: float) -> str:
    return f"$ {v:,.2f}"


def _build_cotizacion_pdf(req: CotizacionPDFRequest) -> bytes:  # noqa: C901
    """Genera el PDF usando canvas directo (control pixel-perfect)."""
    buf = io.BytesIO()
    PW, PH = A4                          # 595.27 x 841.89 pt
    MX = 1.8 * cm                        # margen horizontal
    CW = PW - 2 * MX                    # ancho útil

    c = pdf_canvas.Canvas(buf, pagesize=A4)
    c.setTitle(f"Cotización COT-{req.numero.zfill(3)}")

    # Paleta
    DARK        = HexColor("#111827")
    BRAND       = HexColor("#2563EB")
    BRAND_DARK  = HexColor("#1D4ED8")
    BRAND_LIGHT = HexColor("#DBEAFE")
    GRAY_L      = HexColor("#F3F4F6")
    GRAY_M      = HexColor("#6B7280")
    GRAY_D      = HexColor("#374151")
    GRAY_LINE   = HexColor("#E5E7EB")
    WHITE       = colors.white

    cot_num  = f"COT-{req.numero.zfill(3)}"
    today_s  = datetime.now().strftime("%d de %B de %Y")
    y = PH   # cursor vertical (desciende)

    # ── helpers ───────────────────────────────────────────────────────────────
    def fill(col):   c.setFillColor(col)
    def stroke(col): c.setStrokeColor(col)
    def font(name, size): c.setFont(name, size)
    def rect_f(x, y, w, h, col):
        c.setFillColor(col); c.setStrokeColor(col)
        c.rect(x, y, w, h, fill=1, stroke=0)
    def hrule(y_pos, col=GRAY_LINE, lw=0.5):
        c.setStrokeColor(col); c.setLineWidth(lw)
        c.line(MX, y_pos, PW - MX, y_pos)
    def label_val(label, val, x, y_pos, col_w):
        font("Helvetica-Bold", 6.5); fill(GRAY_M)
        c.drawString(x, y_pos, label.upper())
        font("Helvetica-Bold", 10); fill(DARK)
        # clip long text
        val_s = str(val or "—")
        c.drawString(x, y_pos - 13, val_s[:55])

    # ═══════════════════════════════════════════════════════════════════════
    # 1. HEADER (banda oscura full-width)
    # ═══════════════════════════════════════════════════════════════════════
    HDR_H = 74
    y -= HDR_H
    rect_f(0, y, PW, HDR_H, DARK)

    # Logo / empresa
    font("Helvetica-Bold", 15); fill(WHITE)
    c.drawString(MX, y + 44, "CORSUSA INTERNATIONAL S.A.C.")
    font("Helvetica", 7.5); fill(HexColor("#9CA3AF"))
    c.drawString(MX, y + 26, "INGENIERÍA  ·  AUTOMATIZACIÓN  ·  SUMINISTROS")

    # COT número (derecha)
    font("Helvetica", 7); fill(HexColor("#9CA3AF"))
    c.drawRightString(PW - MX, y + 52, "COTIZACIÓN")
    font("Helvetica-Bold", 20); fill(WHITE)
    c.drawRightString(PW - MX, y + 26, cot_num)

    # ═══════════════════════════════════════════════════════════════════════
    # 2. STRIPE AZUL
    # ═══════════════════════════════════════════════════════════════════════
    STRIPE_H = 22
    y -= STRIPE_H
    rect_f(0, y, PW, STRIPE_H, BRAND)
    font("Helvetica-Bold", 8.5); fill(WHITE)
    c.drawCentredString(PW / 2, y + 7, "PROPUESTA TÉCNICO-COMERCIAL")

    # ═══════════════════════════════════════════════════════════════════════
    # 3. DATOS DEL CLIENTE (2 columnas)
    # ═══════════════════════════════════════════════════════════════════════
    y -= 18
    half = CW / 2 - 6
    col1 = MX
    col2 = MX + CW / 2 + 6

    label_val("Nombre del Postor", "CORSUSA INTERNATIONAL S.A.C.", col1, y, half)
    label_val("Fecha",              today_s,                       col2, y, half)
    y -= 32

    if req.proyecto or req.cliente:
        label_val("Proyecto", req.proyecto, col1, y, half)
        label_val("Cliente",  req.cliente,  col2, y, half)
        y -= 32

    if req.ubicacion:
        label_val("Ubicación", req.ubicacion, col1, y, half)
        y -= 32

    if req.descripcion:
        label_val("Descripción", req.descripcion, col1, y, CW)
        y -= 32

    y -= 4
    hrule(y)
    y -= 14

    # ═══════════════════════════════════════════════════════════════════════
    # 4. TABLA DE PRECIOS
    # ═══════════════════════════════════════════════════════════════════════
    # anchos de columna: item | descripcion | und | cant | pv | total
    COL = [0.85*cm, 0, 1.4*cm, 1.4*cm, 3.6*cm, 3.8*cm]
    COL[1] = CW - sum(COL) + COL[1]          # descripción toma el resto
    col_x = [MX]
    for w in COL[:-1]:
        col_x.append(col_x[-1] + w)

    ROW_H = 22

    # Encabezado tabla
    rect_f(MX, y - ROW_H, CW, ROW_H, DARK)
    headers    = ["#",    "DESCRIPCIÓN",  "UND",   "CANT.", "P.V. UNIT.",  "PRECIO TOTAL"]
    h_aligns   = ["C",    "L",            "C",     "C",     "R",           "R"]
    font("Helvetica-Bold", 8); fill(WHITE)
    for h, hx, hw, ha in zip(headers, col_x, COL, h_aligns):
        ty = y - ROW_H + 7
        if ha == "C": c.drawCentredString(hx + hw/2, ty, h)
        elif ha == "R": c.drawRightString(hx + hw - 4, ty, h)
        else: c.drawString(hx + 5, ty, h)
    y -= ROW_H

    # Filas de datos
    subtotal = 0.0
    all_rows = list(req.disciplinas) + ([type("GG", (), {"nombre": "Gastos Generales", "precio_venta": req.precio_gg})()] if req.precio_gg > 0 else [])

    for i, disc in enumerate(all_rows):
        pv = disc.precio_venta
        subtotal += pv

        bg = WHITE if i % 2 == 0 else GRAY_L
        rect_f(MX,       y - ROW_H, CW - COL[-1], ROW_H, bg)
        rect_f(col_x[-1], y - ROW_H, COL[-1],      ROW_H, BRAND_LIGHT)

        # línea inferior
        c.setStrokeColor(GRAY_LINE); c.setLineWidth(0.3)
        c.line(MX, y - ROW_H, PW - MX, y - ROW_H)

        ty = y - ROW_H + 7
        font("Helvetica", 9); fill(GRAY_M)
        c.drawCentredString(col_x[0] + COL[0]/2, ty, str(i + 1))

        font("Helvetica-Bold", 9.5); fill(DARK)
        c.drawString(col_x[1] + 5, ty, disc.nombre[:52])

        font("Helvetica", 9); fill(GRAY_M)
        c.drawCentredString(col_x[2] + COL[2]/2, ty, "GLB")
        c.drawCentredString(col_x[3] + COL[3]/2, ty, "1")

        font("Helvetica", 9.5); fill(DARK)
        c.drawRightString(col_x[4] + COL[4] - 4, ty, _fmt_usd(pv))

        font("Helvetica-Bold", 9.5); fill(BRAND_DARK)
        c.drawRightString(col_x[5] + COL[5] - 4, ty, _fmt_usd(pv))

        y -= ROW_H

    # Fila TOTAL
    rect_f(MX, y - ROW_H - 2, CW, ROW_H + 2, DARK)
    ty = y - ROW_H + 5
    font("Helvetica-Bold", 9); fill(GRAY_L)
    c.drawRightString(col_x[4] + COL[4] - 4, ty, "COSTO SIN IGV")
    font("Helvetica-Bold", 12); fill(WHITE)
    c.drawRightString(col_x[5] + COL[5] - 4, ty, _fmt_usd(subtotal))
    y -= ROW_H + 2

    # ═══════════════════════════════════════════════════════════════════════
    # 5. BLOQUE IGV (alineado derecha)
    # ═══════════════════════════════════════════════════════════════════════
    igv       = subtotal * 0.18
    total_igv = subtotal * 1.18
    y -= 12

    igv_x  = PW - MX - 5.5*cm   # arranque del bloque
    igv_rw = 5.5*cm              # ancho del bloque

    def igv_row(label, value, is_total=False):
        nonlocal y
        if is_total:
            hrule(y + 2, DARK, 1)
            y -= 6
        size = 10.5 if is_total else 9
        font_name = "Helvetica-Bold" if is_total else "Helvetica"
        font(font_name, size)
        fill(DARK if is_total else GRAY_M)
        c.drawString(igv_x, y, label)
        fill(BRAND_DARK if is_total else DARK)
        c.drawRightString(PW - MX, y, value)
        y -= 16 if is_total else 13

    igv_row("Sub Total (sin IGV)", _fmt_usd(subtotal))
    igv_row("IGV (18%)",           _fmt_usd(igv))
    igv_row("TOTAL CON IGV",       _fmt_usd(total_igv), is_total=True)

    font("Helvetica", 7.5); fill(GRAY_M)
    c.drawRightString(PW - MX, y + 4, f"S/. {total_igv * req.tipo_cambio:,.0f}  (TC S/. {req.tipo_cambio})")
    y -= 20

    # ═══════════════════════════════════════════════════════════════════════
    # 6. CONDICIONES COMERCIALES
    # ═══════════════════════════════════════════════════════════════════════
    hrule(y)
    y -= 14

    font("Helvetica-Bold", 7); fill(GRAY_M)
    c.drawString(MX, y, "CONDICIONES COMERCIALES")
    y -= 10

    # Caja gris
    COND_H = 58
    c.setFillColor(GRAY_L); c.setStrokeColor(GRAY_LINE); c.setLineWidth(0.5)
    c.roundRect(MX, y - COND_H, CW, COND_H, 5, fill=1, stroke=1)

    plazo_str   = f"{req.plazo_semanas} Semanas" if req.plazo_semanas else "Por definir"
    garantia_s  = f"{req.garantia_meses} Mes{'es' if req.garantia_meses != 1 else ''}"
    forma_s     = req.forma_pago or "Por coordinar"

    PAD = 12
    cw4 = CW / 4

    def cond_field(lbl, val, cx, cy):
        font("Helvetica-Bold", 6.5); fill(GRAY_M)
        c.drawString(cx, cy, lbl.upper())
        font("Helvetica-Bold", 10); fill(DARK)
        c.drawString(cx, cy - 13, str(val)[:30])

    cy_top = y - 13
    cond_field("Plazo de Entrega",   plazo_str,              MX + PAD,          cy_top)
    cond_field("Garantía",           garantia_s,             MX + cw4 + PAD,    cy_top)
    cond_field("Validez de Oferta",  f"{req.validez_dias} Días", MX + 2*cw4 + PAD, cy_top)
    cond_field("Forma de Pago",      forma_s,                MX + 3*cw4 + PAD,  cy_top)

    y -= COND_H + 12

    # ═══════════════════════════════════════════════════════════════════════
    # 7. FOOTER
    # ═══════════════════════════════════════════════════════════════════════
    FOOTER_Y = 1.3 * cm
    hrule(FOOTER_Y + 14, GRAY_LINE, 0.4)
    font("Helvetica", 7); fill(GRAY_M)
    c.drawCentredString(PW/2, FOOTER_Y + 4,
        "Este documento es confidencial y preparado exclusivamente para el cliente indicado.")
    c.drawCentredString(PW/2, FOOTER_Y - 6,
        "CORSUSA INTERNATIONAL S.A.C.  ·  RUC 20601182553")

    c.showPage()
    c.save()
    buf.seek(0)
    return buf.read()


@app.post("/cotizaciones/pdf")
async def generar_cotizacion_pdf(req: CotizacionPDFRequest):
    """Genera un PDF de cotización para el cliente."""
    try:
        pdf_bytes = _build_cotizacion_pdf(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generando PDF: {e}")

    filename = f"COT-{req.numero.zfill(3)}.pdf"
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/extraer/libro/insumos")
async def extraer_libro_insumos(archivo: UploadFile = File(...)):
    """Extrae SOLO insumos desde un libro de precios (PDF o imagen)."""
    model = get_model()
    pdf_bytes, img = await _extraer_con_prompt(archivo, LIBRO_INSUMOS_PROMPT)
    resultado = run_extraction(model, pdf_bytes if pdf_bytes else None, img, LIBRO_INSUMOS_PROMPT)
    insumos = resultado.get("insumos", [])
    return JSONResponse(content={
        "archivo":       archivo.filename,
        "tipo":          "insumos",
        "total_insumos": len(insumos),
        "insumos":       insumos,
    })
