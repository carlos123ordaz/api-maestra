import os
import json
import io
import csv
import pathlib
from datetime import datetime

import fitz  # PyMuPDF
import httpx
import google.generativeai as genai
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from PIL import Image

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
    "id", "title", "status", "priority", "responsible_name",
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
                    "ID", "TITLE", "STATUS", "PRIORITY",
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

    # Resolver nombres de usuarios
    user_ids: set[str] = set()
    for t in tasks:
        if t.get("responsibleId"):
            user_ids.add(str(t["responsibleId"]))
        if t.get("createdBy"):
            user_ids.add(str(t["createdBy"]))

    user_map: dict[str, str] = {}
    if user_ids:
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

    normalized = []
    for t in tasks:
        status_raw   = str(t.get("status", ""))
        priority_raw = str(t.get("priority", "1"))
        rid = str(t.get("responsibleId", ""))
        cid = str(t.get("createdBy", ""))
        normalized.append({
            "id":               str(t.get("id", "")),
            "title":            t.get("title", ""),
            "status":           TASK_STATUS_MAP.get(status_raw, status_raw),
            "priority":         TASK_PRIORITY_MAP.get(priority_raw, priority_raw),
            "responsible_name": user_map.get(rid, rid),
            "created_by_name":  user_map.get(cid, cid),
            "created_date":     t.get("createdDate", ""),
            "deadline":         t.get("deadline", "") or "",
            "description":      (t.get("description") or "").strip(),
        })
    return normalized


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


def _build_response(tasks: list[dict], group_id: str, group_name: str, source: str) -> dict:
    mtime = datetime.fromtimestamp(_csv_path(group_id).stat().st_mtime).isoformat() \
        if _csv_path(group_id).exists() else datetime.now().isoformat()
    return {
        "source":     source,
        "group_id":   group_id,
        "group_name": group_name,
        "last_sync":  mtime,
        "total":      len(tasks),
        "tareas":     tasks,
    }


@app.get("/bitrix/tareas/{group_id}")
async def get_bitrix_tareas(group_id: str):
    """Devuelve las tareas de un grupo Bitrix24. Usa caché CSV si existe."""
    rows = _load_csv(group_id)
    if rows:
        group_name = await _fetch_group_name(group_id)
        return JSONResponse(content=_build_response(rows, group_id, group_name, "cache"))
    tasks      = await _fetch_all_bitrix_tasks(group_id)
    group_name = await _fetch_group_name(group_id)
    _save_csv(tasks, group_id)
    return JSONResponse(content=_build_response(tasks, group_id, group_name, "bitrix"))


@app.post("/bitrix/tareas/{group_id}/sync")
async def sync_bitrix_tareas(group_id: str):
    """Fuerza la sincronización con Bitrix24 para un grupo y actualiza su CSV."""
    try:
        tasks = await _fetch_all_bitrix_tasks(group_id)
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"Error al consultar Bitrix24: {e}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
    group_name = await _fetch_group_name(group_id)
    _save_csv(tasks, group_id)
    return JSONResponse(content=_build_response(tasks, group_id, group_name, "bitrix"))


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
async def get_bitrix_productos():
    """Devuelve los productos CRM de Bitrix24. Usa caché CSV si existe."""
    rows = _load_productos_csv()
    if rows:
        return JSONResponse(content=_productos_response(rows, "cache"))
    sections = await _fetch_all_sections()
    products = await _fetch_all_crm_products()
    _save_productos_csv(products, sections)
    return JSONResponse(content=_productos_response(_load_productos_csv(), "bitrix"))


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
