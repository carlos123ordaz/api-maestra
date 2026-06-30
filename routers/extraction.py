import io
import json

import google.generativeai as genai
import fitz  # PyMuPDF
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from config import GEMINI_API_KEY

router = APIRouter()

# ── Prompts ───────────────────────────────────────────────────────────────────

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

# ── Configuración del modelo ──────────────────────────────────────────────────

STRIPS = 3
SCALE  = 2.0


def get_model():
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY no configurada en el entorno.")
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel(
        "gemini-2.5-flash-lite",
        generation_config=genai.GenerationConfig(max_output_tokens=65536),
    )


def parse_response(text: str | None) -> dict:
    if not text or not text.strip():
        raise ValueError("El modelo devolvió una respuesta vacía (posiblemente bloqueada por filtros de seguridad).")

    text = text.strip()

    if text.startswith("```"):
        lines = text.split("\n")
        end_md = next((i for i in range(len(lines) - 1, 0, -1) if lines[i].strip() == "```"), len(lines))
        text = "\n".join(lines[1:end_md]).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

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
            pass
    return {"partidas": all_partidas, "insumos": all_insumos}


def run_extraction(model, pdf_bytes: bytes | None, image: Image.Image | None, prompt: str) -> dict:
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


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/extraer")
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


async def _extraer_con_prompt(archivo: UploadFile, prompt: str):
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


@router.post("/extraer/libro/partidas")
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


@router.post("/extraer/libro/insumos")
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
