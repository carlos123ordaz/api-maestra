import io
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas as pdf_canvas

router = APIRouter()


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
    buf = io.BytesIO()
    PW, PH = A4
    MX = 1.8 * cm
    CW = PW - 2 * MX

    c = pdf_canvas.Canvas(buf, pagesize=A4)
    c.setTitle(f"Cotización COT-{req.numero.zfill(3)}")

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
    y = PH

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
        val_s = str(val or "—")
        c.drawString(x, y_pos - 13, val_s[:55])

    # 1. HEADER
    HDR_H = 74
    y -= HDR_H
    rect_f(0, y, PW, HDR_H, DARK)
    font("Helvetica-Bold", 15); fill(WHITE)
    c.drawString(MX, y + 44, "CORSUSA INTERNATIONAL S.A.C.")
    font("Helvetica", 7.5); fill(HexColor("#9CA3AF"))
    c.drawString(MX, y + 26, "INGENIERÍA  ·  AUTOMATIZACIÓN  ·  SUMINISTROS")
    font("Helvetica", 7); fill(HexColor("#9CA3AF"))
    c.drawRightString(PW - MX, y + 52, "COTIZACIÓN")
    font("Helvetica-Bold", 20); fill(WHITE)
    c.drawRightString(PW - MX, y + 26, cot_num)

    # 2. STRIPE AZUL
    STRIPE_H = 22
    y -= STRIPE_H
    rect_f(0, y, PW, STRIPE_H, BRAND)
    font("Helvetica-Bold", 8.5); fill(WHITE)
    c.drawCentredString(PW / 2, y + 7, "PROPUESTA TÉCNICO-COMERCIAL")

    # 3. DATOS DEL CLIENTE
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

    # 4. TABLA DE PRECIOS
    COL = [0.85*cm, 0, 1.4*cm, 1.4*cm, 3.6*cm, 3.8*cm]
    COL[1] = CW - sum(COL) + COL[1]
    col_x = [MX]
    for w in COL[:-1]:
        col_x.append(col_x[-1] + w)

    ROW_H = 22
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

    subtotal = 0.0
    all_rows = list(req.disciplinas) + ([type("GG", (), {"nombre": "Gastos Generales", "precio_venta": req.precio_gg})()] if req.precio_gg > 0 else [])

    for i, disc in enumerate(all_rows):
        pv = disc.precio_venta
        subtotal += pv
        bg = WHITE if i % 2 == 0 else GRAY_L
        rect_f(MX,       y - ROW_H, CW - COL[-1], ROW_H, bg)
        rect_f(col_x[-1], y - ROW_H, COL[-1],      ROW_H, BRAND_LIGHT)
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

    rect_f(MX, y - ROW_H - 2, CW, ROW_H + 2, DARK)
    ty = y - ROW_H + 5
    font("Helvetica-Bold", 9); fill(GRAY_L)
    c.drawRightString(col_x[4] + COL[4] - 4, ty, "COSTO SIN IGV")
    font("Helvetica-Bold", 12); fill(WHITE)
    c.drawRightString(col_x[5] + COL[5] - 4, ty, _fmt_usd(subtotal))
    y -= ROW_H + 2

    # 5. BLOQUE IGV
    igv       = subtotal * 0.18
    total_igv = subtotal * 1.18
    y -= 12

    igv_x  = PW - MX - 5.5*cm
    igv_rw = 5.5*cm

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

    # 6. CONDICIONES COMERCIALES
    hrule(y)
    y -= 14
    font("Helvetica-Bold", 7); fill(GRAY_M)
    c.drawString(MX, y, "CONDICIONES COMERCIALES")
    y -= 10

    COND_H = 58
    c.setFillColor(GRAY_L); c.setStrokeColor(GRAY_LINE); c.setLineWidth(0.5)
    c.roundRect(MX, y - COND_H, CW, COND_H, 5, fill=1, stroke=1)

    plazo_str  = f"{req.plazo_semanas} Semanas" if req.plazo_semanas else "Por definir"
    garantia_s = f"{req.garantia_meses} Mes{'es' if req.garantia_meses != 1 else ''}"
    forma_s    = req.forma_pago or "Por coordinar"

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

    # 7. FOOTER
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


@router.post("/cotizaciones/pdf")
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
