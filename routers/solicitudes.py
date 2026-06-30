import io
import json
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, HRFlowable,
)

router = APIRouter()

_TIPO_LABEL: dict[str, str] = {
    "aereo": "Aéreo", "terrestre": "Terrestre", "aereo_terrestre": "Aéreo y Terrestre",
    "nacional": "Nacional", "internacional": "Internacional",
    "ida": "Compra de boleto de Ida",
    "regreso": "Compra de boleto de Regreso",
    "ida_regreso": "Compra de boleto de Ida y regreso",
    "cambio_fecha": "Cambio de fecha",
    "mochila": "Considerar solo mochila de mano",
    "10kg": "Considerar maleta de 10 kg. (cabina)",
    "23kg": "Considerar maleta de 23 kg. (bodega)",
}


class PassengerData(BaseModel):
    nombre: str = ""
    dni: str = ""
    nacimiento: str = ""
    celular: str = ""
    correo: str = ""


class SolicitudBoletoPDFRequest(BaseModel):
    submission_id: str
    submitter_name: str
    submitter_email: str
    fecha_solicitud: str
    answers: dict[str, str]


def _disp(key: str, val: str) -> str:
    return _TIPO_LABEL.get(val, val)


def _fmt_fecha(iso: str) -> str:
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.strftime("%d/%m/%Y %H:%M")
    except Exception:
        return iso[:10] if len(iso) >= 10 else iso


def _fmt_date_only(s: str) -> str:
    try:
        return datetime.strptime(s, "%Y-%m-%d").strftime("%d/%m/%Y")
    except Exception:
        return s


def _build_solicitud_boleto_pdf(req: SolicitudBoletoPDFRequest) -> bytes:  # noqa: C901
    buf = io.BytesIO()
    PW, PH = A4
    MX = 1.8 * cm
    CW = PW - 2 * MX

    BLACK    = colors.black
    WHITE    = colors.white
    GRAY_ROW = HexColor("#F5F5F5")
    GRAY_LBL = HexColor("#555555")

    a = req.answers

    def S(name, font_name="Helvetica", size=9, color=BLACK,
          align=TA_LEFT, leading=None, bold=False):
        return ParagraphStyle(
            name,
            fontName="Helvetica-Bold" if bold else font_name,
            fontSize=size,
            textColor=color,
            alignment=align,
            leading=leading or size * 1.3,
        )

    s_company  = S("company",  bold=True,  size=12)
    s_ruc      = S("ruc",      size=7.5,   color=GRAY_LBL)
    s_doctitle = S("doctitle", bold=True,  size=11, align=TA_RIGHT)
    s_docid    = S("docid",    size=7.5,   color=GRAY_LBL, align=TA_RIGHT)
    s_lbl      = S("lbl",      bold=True,  size=6.5, color=GRAY_LBL)
    s_val      = S("val",      size=9)
    s_val_b    = S("valb",     bold=True,  size=9)
    s_hdr      = S("hdr",      bold=True,  size=7.5, color=WHITE)
    s_sec      = S("sec",      bold=True,  size=8,   color=WHITE)
    s_body     = S("body",     size=9,     leading=13)
    s_foot     = S("foot",     size=7,     color=GRAY_LBL, align=TA_CENTER)

    BORDER = TableStyle([
        ("BOX",          (0, 0), (-1, -1), 0.5, BLACK),
        ("INNERGRID",    (0, 0), (-1, -1), 0.5, BLACK),
        ("TOPPADDING",   (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
        ("LEFTPADDING",  (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
    ])

    def with_header(extra: list) -> TableStyle:
        return TableStyle(BORDER.getCommands() + [
            ("BACKGROUND", (0, 0), (-1, 0), BLACK),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, GRAY_ROW]),
        ] + extra)

    def field_block(label: str, value: str) -> list:
        return [Paragraph(label, s_lbl), Paragraph(value or "—", s_val)]

    def field_block_b(label: str, value: str) -> list:
        return [Paragraph(label, s_lbl), Paragraph(value or "—", s_val_b)]

    def section_bar(title: str):
        t = Table([[Paragraph(f"  {title}", s_sec)]], colWidths=[CW])
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), BLACK),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING",   (0, 0), (-1, -1), 4),
        ]))
        return t

    def bordered_text(text: str):
        t = Table([[Paragraph(text, s_body)]], colWidths=[CW])
        t.setStyle(TableStyle([
            ("BOX",           (0, 0), (-1, -1), 0.5, BLACK),
            ("TOPPADDING",    (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ("LEFTPADDING",   (0, 0), (-1, -1), 8),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
        ]))
        return t

    SP = lambda n=6: Spacer(1, n)

    story = []

    # 1. HEADER
    hdr = Table([[
        [Paragraph("CORSUSA INTERNATIONAL S.A.C.", s_company),
         Paragraph("RUC 20601182553", s_ruc)],
        [Paragraph("SOLICITUD DE COMPRA DE BOLETO", s_doctitle),
         Paragraph(f"N° {req.submission_id[:8].upper()}", s_docid)],
    ]], colWidths=[CW * 0.55, CW * 0.45])
    hdr.setStyle(TableStyle([
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(hdr)
    story.append(HRFlowable(width="100%", thickness=1.2, color=BLACK, spaceAfter=2))
    story.append(HRFlowable(width="100%", thickness=0.3, color=BLACK, spaceAfter=8))

    # 2. SOLICITANTE + FECHA
    info = Table([[
        field_block_b("SOLICITANTE", req.submitter_name),
        field_block("CORREO ELECTRÓNICO", req.submitter_email),
        field_block("FECHA DE SOLICITUD", _fmt_fecha(req.fecha_solicitud)),
        field_block("DOCUMENTO", req.submission_id[:8].upper()),
    ]], colWidths=[CW * 0.28, CW * 0.32, CW * 0.24, CW * 0.16])
    info.setStyle(BORDER)
    story.append(info)
    story.append(SP(3))

    # 3. CENTRO COSTO / TASK / TIPO
    meta = Table([[
        field_block_b("CENTRO DE COSTO", a.get("centro_costo", "")),
        field_block_b("N° TASK / REFERENCIA", a.get("numero_task", "")),
        field_block("TIPO DE BOLETO", _disp("tipo_boleto", a.get("tipo_boleto", ""))),
    ]], colWidths=[CW / 3, CW / 3, CW / 3])
    meta.setStyle(BORDER)
    story.append(meta)
    story.append(SP(10))

    # 4. INFORMACIÓN DEL VIAJE
    story.append(section_bar("INFORMACIÓN DEL VIAJE"))
    story.append(SP(2))

    row1 = Table([[
        field_block("DESTINO DE VUELO",  _disp("destino_vuelo", a.get("destino_vuelo", ""))),
        field_block("TIPO DE SERVICIO",  _disp("tipo_servicio", a.get("tipo_servicio", ""))),
        field_block("N° PASAJEROS",      a.get("num_pasajeros", "")),
    ]], colWidths=[CW * 0.32, CW * 0.48, CW * 0.20])
    row1.setStyle(BORDER)
    story.append(row1)
    story.append(SP(2))

    row2 = Table([[
        field_block("CIUDAD DE SALIDA",  a.get("ciudad_salida", "")),
        field_block("CIUDAD DESTINO",    a.get("ciudad_destino", "")),
    ]], colWidths=[CW / 2, CW / 2])
    row2.setStyle(BORDER)
    story.append(row2)
    story.append(SP(2))

    row3 = Table([[
        field_block("FECHA DE SALIDA",            _fmt_date_only(a.get("fecha_salida", ""))),
        field_block("HORA MÁX. LLEGADA AL DESTINO", a.get("hora_llegada_destino", "")),
        field_block("FECHA DE REGRESO",           _fmt_date_only(a.get("fecha_regreso", ""))),
        field_block("HORA MÁX. SALIDA AEROPUERTO", a.get("hora_salida_aeropuerto", "")),
    ]], colWidths=[CW * 0.20, CW * 0.30, CW * 0.20, CW * 0.30])
    row3.setStyle(BORDER)
    story.append(row3)
    story.append(SP(2))

    row4 = Table([[
        field_block("EQUIPAJE", _disp("equipaje", a.get("equipaje", ""))),
    ]], colWidths=[CW])
    row4.setStyle(BORDER)
    story.append(row4)
    story.append(SP(10))

    # 5. PASAJEROS
    story.append(section_bar("DATOS DE LOS PASAJEROS"))
    story.append(SP(2))

    raw_pax = a.get("datos_pasajeros", "")
    passengers: list[dict] = []
    try:
        passengers = json.loads(raw_pax)
    except Exception:
        if raw_pax:
            passengers = [{"nombre": raw_pax}]

    if passengers:
        pax_rows = [[
            Paragraph(h, s_hdr) for h in
            ["N°", "NOMBRE COMPLETO", "DNI", "F. NACIMIENTO", "CELULAR", "CORREO"]
        ]]
        for i, pax in enumerate(passengers):
            pax_rows.append([
                Paragraph(str(i + 1), s_val),
                Paragraph(pax.get("nombre", "—"), s_val_b),
                Paragraph(pax.get("dni", "—"), s_val),
                Paragraph(_fmt_date_only(pax.get("nacimiento", "")) or "—", s_val),
                Paragraph(pax.get("celular", "—"), s_val),
                Paragraph(pax.get("correo", "—"), s_val),
            ])
        pax_tbl = Table(
            pax_rows,
            colWidths=[CW*0.05, CW*0.28, CW*0.12, CW*0.14, CW*0.14, CW*0.27],
        )
        pax_tbl.setStyle(with_header([
            ("ALIGN",  (0, 0), (0, -1), "CENTER"),
            ("FONTNAME", (1, 1), (1, -1), "Helvetica-Bold"),
        ]))
        story.append(pax_tbl)
    else:
        story.append(Paragraph("Sin datos de pasajeros registrados.", s_body))

    # 6. PERSONAL TERCERO (opcional)
    personal_tercero = a.get("personal_tercero", "").strip()
    if personal_tercero:
        story.append(SP(10))
        story.append(section_bar("PERSONAL TERCERO"))
        story.append(SP(2))
        story.append(bordered_text(personal_tercero))

    # 7. NOTA (opcional)
    nota = a.get("nota", "").strip()
    if nota:
        story.append(SP(10))
        story.append(section_bar("OBSERVACIONES"))
        story.append(SP(2))
        story.append(bordered_text(nota))

    # 8. FIRMA
    story.append(SP(28))
    firma = Table([[
        [
            HRFlowable(width=5*cm, thickness=0.6, color=BLACK, spaceAfter=3),
            Paragraph(req.submitter_name.upper(), S("fn", bold=True, size=8)),
            Paragraph("Firma del Solicitante", S("fl", size=7, color=GRAY_LBL)),
        ],
        "",
    ]], colWidths=[CW * 0.4, CW * 0.6])
    firma.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.append(firma)

    def draw_footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(BLACK)
        canvas.setLineWidth(0.4)
        canvas.line(MX, 1.8 * cm, PW - MX, 1.8 * cm)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(GRAY_LBL)
        canvas.drawCentredString(PW / 2, 1.4 * cm,
                                 "Documento generado automáticamente — uso interno Corsusa")
        canvas.drawCentredString(PW / 2, 1.1 * cm,
                                 "CORSUSA INTERNATIONAL S.A.C.  ·  RUC 20601182553")
        canvas.restoreState()

    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=MX, rightMargin=MX,
        topMargin=1.4 * cm, bottomMargin=2.4 * cm,
        title="Solicitud de Compra de Boleto",
    )
    doc.build(story, onFirstPage=draw_footer, onLaterPages=draw_footer)
    buf.seek(0)
    return buf.read()


@router.post("/solicitudes/boleto/pdf")
async def generar_solicitud_boleto_pdf(req: SolicitudBoletoPDFRequest):
    """Genera el PDF de una solicitud de compra de boleto."""
    try:
        pdf_bytes = _build_solicitud_boleto_pdf(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generando PDF: {e}")

    safe_name = req.submitter_name.replace(" ", "_")[:20]
    filename = f"Solicitud_Boleto_{safe_name}_{req.submission_id[:8]}.pdf"
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
