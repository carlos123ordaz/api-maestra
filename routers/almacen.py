from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config import MS_CLIENT_ID, MS_CLIENT_SECRET, MS_TENANT_ID

router = APIRouter(prefix="/almacen")

# NOTA: Para que el envío funcione, la app de Azure AD debe tener el permiso
# de aplicación "Mail.Send" concedido por un administrador en:
# Entra ID → Registros de aplicaciones → [tu app] → Permisos de API
#
# Correos placeholder — reemplazar con los reales antes de producción:
_FROM_EMAIL = "cordaz@corsusa.com"   # remitente (cuenta habilitada en M365)
_TO_EMAIL = "ltaipe@corsusa.com"   # destinatario


class ItemPedido(BaseModel):
    descripcion: str
    cantidad: float
    unidad: str
    precio_unitario: float = 0.0


class NotificacionPedidoRequest(BaseModel):
    numero: int
    solicitado_por: Optional[str] = None
    proveedor_sugerido: Optional[str] = None
    fecha_pedido: str
    fecha_requerida: Optional[str] = None
    observaciones: Optional[str] = None
    items: list[ItemPedido] = []


class NotificacionSeguimientoRequest(BaseModel):
    numero: int
    asunto: Optional[str] = None
    estado: str
    from_email: Optional[str] = None
    firma: Optional[str] = None
    destinatarios: list[str] = []


async def _get_graph_token() -> str:
    """Obtiene access token de Microsoft Graph usando client credentials."""
    url = f"https://login.microsoftonline.com/{MS_TENANT_ID}/oauth2/v2.0/token"
    payload = {
        "grant_type":    "client_credentials",
        "client_id":     MS_CLIENT_ID,
        "client_secret": MS_CLIENT_SECRET,
        "scope":         "https://graph.microsoft.com/.default",
    }
    async with httpx.AsyncClient() as client:
        r = await client.post(url, data=payload)
    if r.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Error al obtener token de Microsoft Graph: {r.text}",
        )
    return r.json()["access_token"]


def _build_email_html(req: NotificacionPedidoRequest) -> str:
    """Construye el cuerpo HTML del correo de notificación."""
    numero_str = str(req.numero).zfill(4)
    total = sum(it.cantidad * it.precio_unitario for it in req.items)

    filas_items = ""
    for it in req.items:
        subtotal = it.cantidad * it.precio_unitario
        filas_items += f"""
        <tr>
          <td style="padding:8px 12px;border-bottom:1px solid #e5e7eb;color:#374151;">{it.descripcion}</td>
          <td style="padding:8px 12px;border-bottom:1px solid #e5e7eb;text-align:center;color:#374151;">{it.cantidad:g}</td>
          <td style="padding:8px 12px;border-bottom:1px solid #e5e7eb;text-align:center;color:#374151;">{it.unidad}</td>
          <td style="padding:8px 12px;border-bottom:1px solid #e5e7eb;text-align:right;color:#374151;">S/ {subtotal:.2f}</td>
        </tr>"""

    if not filas_items:
        filas_items = '<tr><td colspan="4" style="padding:12px;text-align:center;color:#9ca3af;">Sin ítems registrados</td></tr>'

    proveedor = req.proveedor_sugerido or "—"
    f_pedido = req.fecha_pedido or "—"
    f_req = req.fecha_requerida or "—"
    solicitante = req.solicitado_por or "—"

    return f"""<!DOCTYPE html>
<html lang="es">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="margin:0;padding:0;background:#f3f4f6;font-family:'Segoe UI',Arial,sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background:#f3f4f6;padding:32px 0;">
    <tr><td align="center">
      <table width="600" cellpadding="0" cellspacing="0" style="background:#ffffff;border-radius:12px;overflow:hidden;box-shadow:0 4px 24px rgba(0,0,0,.08);">

        <!-- Header -->
        <tr>
          <td style="background:#0047BA;padding:28px 32px;">
            <p style="margin:0;font-size:22px;font-weight:700;color:#ffffff;letter-spacing:-0.5px;">
              Corsusa Internacional S.A.C.
            </p>
            <p style="margin:6px 0 0;font-size:13px;color:#93b5e5;">Gestión de Almacén</p>
          </td>
        </tr>

        <!-- Título -->
        <tr>
          <td style="padding:28px 32px 16px;">
            <table cellpadding="0" cellspacing="0" width="100%">
              <tr>
                <td>
                  <p style="margin:0;font-size:13px;font-weight:700;color:#0047BA;letter-spacing:.06em;text-transform:uppercase;">
                    Notificación de pedido
                  </p>
                  <h1 style="margin:4px 0 0;font-size:24px;font-weight:700;color:#111827;">
                    Pedido #{numero_str} — En camino
                  </h1>
                  <p style="margin:6px 0 0;font-size:13px;color:#6b7280;">
                    El siguiente pedido ha sido marcado como <strong style="color:#7c3aed;">Enviado</strong>
                    y se encuentra en camino a las instalaciones.
                  </p>
                </td>
                <td align="right" valign="top">
                  <span style="display:inline-block;padding:6px 16px;background:#f5f3ff;color:#7c3aed;font-size:12px;font-weight:700;border-radius:20px;border:1px solid #ddd6fe;">
                    ENVIADO
                  </span>
                </td>
              </tr>
            </table>
          </td>
        </tr>

        <!-- Datos del pedido -->
        <tr>
          <td style="padding:0 32px 20px;">
            <table width="100%" cellpadding="0" cellspacing="0" style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:8px;">
              <tr>
                <td style="padding:14px 18px;border-bottom:1px solid #e5e7eb;">
                  <p style="margin:0;font-size:11px;color:#9ca3af;font-weight:600;text-transform:uppercase;letter-spacing:.05em;">Solicitado por</p>
                  <p style="margin:2px 0 0;font-size:14px;font-weight:600;color:#111827;">{solicitante}</p>
                </td>
                <td style="padding:14px 18px;border-bottom:1px solid #e5e7eb;border-left:1px solid #e5e7eb;">
                  <p style="margin:0;font-size:11px;color:#9ca3af;font-weight:600;text-transform:uppercase;letter-spacing:.05em;">Proveedor</p>
                  <p style="margin:2px 0 0;font-size:14px;font-weight:600;color:#111827;">{proveedor}</p>
                </td>
              </tr>
              <tr>
                <td style="padding:14px 18px;">
                  <p style="margin:0;font-size:11px;color:#9ca3af;font-weight:600;text-transform:uppercase;letter-spacing:.05em;">Fecha del pedido</p>
                  <p style="margin:2px 0 0;font-size:14px;font-weight:600;color:#111827;">{f_pedido}</p>
                </td>
                <td style="padding:14px 18px;border-left:1px solid #e5e7eb;">
                  <p style="margin:0;font-size:11px;color:#9ca3af;font-weight:600;text-transform:uppercase;letter-spacing:.05em;">Fecha requerida</p>
                  <p style="margin:2px 0 0;font-size:14px;font-weight:600;color:#111827;">{f_req}</p>
                </td>
              </tr>
            </table>
          </td>
        </tr>

        <!-- Tabla de ítems -->
        <tr>
          <td style="padding:0 32px 24px;">
            <p style="margin:0 0 10px;font-size:13px;font-weight:700;color:#374151;">Ítems del pedido</p>
            <table width="100%" cellpadding="0" cellspacing="0" style="border:1px solid #e5e7eb;border-radius:8px;overflow:hidden;">
              <thead>
                <tr style="background:#f3f4f6;">
                  <th style="padding:9px 12px;text-align:left;font-size:11px;font-weight:700;color:#6b7280;letter-spacing:.05em;text-transform:uppercase;">Descripción</th>
                  <th style="padding:9px 12px;text-align:center;font-size:11px;font-weight:700;color:#6b7280;letter-spacing:.05em;text-transform:uppercase;">Cant.</th>
                  <th style="padding:9px 12px;text-align:center;font-size:11px;font-weight:700;color:#6b7280;letter-spacing:.05em;text-transform:uppercase;">Und.</th>
                  <th style="padding:9px 12px;text-align:right;font-size:11px;font-weight:700;color:#6b7280;letter-spacing:.05em;text-transform:uppercase;">Total</th>
                </tr>
              </thead>
              <tbody>{filas_items}</tbody>
              <tfoot>
                <tr style="background:#f9fafb;border-top:2px solid #e5e7eb;">
                  <td colspan="3" style="padding:10px 12px;font-weight:700;color:#374151;font-size:13px;">Total estimado</td>
                  <td style="padding:10px 12px;text-align:right;font-weight:700;font-size:15px;color:#0047BA;">S/ {total:.2f}</td>
                </tr>
              </tfoot>
            </table>
          </td>
        </tr>

        <!-- Observaciones -->
        {'<tr><td style="padding:0 32px 24px;"><div style="background:#fffbeb;border:1px solid #fde68a;border-radius:8px;padding:14px 18px;"><p style="margin:0 0 4px;font-size:11px;font-weight:700;color:#92400e;text-transform:uppercase;letter-spacing:.05em;">Observaciones</p><p style="margin:0;font-size:13px;color:#78350f;">' + req.observaciones + '</p></div></td></tr>' if req.observaciones else ''}

        <!-- Footer -->
        <tr>
          <td style="background:#f9fafb;border-top:1px solid #e5e7eb;padding:20px 32px;">
            <p style="margin:0;font-size:12px;color:#9ca3af;text-align:center;">
              Este correo fue generado automáticamente por el sistema de almacén de
              <strong style="color:#6b7280;">Corsusa Internacional S.A.C.</strong><br>
              Por favor no responder a este mensaje.
            </p>
          </td>
        </tr>

      </table>
    </td></tr>
  </table>
</body>
</html>"""


def _build_seguimiento_html(req: NotificacionSeguimientoRequest) -> str:
    numero_str = str(req.numero).zfill(4)
    asunto = req.asunto or f"Pedido #{numero_str}"
    firma = req.firma or "Área de Almacén"

    return f"""<!DOCTYPE html>
<html lang="es">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="margin:0;padding:0;background:#f3f4f6;font-family:'Segoe UI',Arial,sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background:#f3f4f6;padding:32px 0;">
    <tr><td align="center">
      <table width="600" cellpadding="0" cellspacing="0" style="background:#ffffff;border-radius:12px;overflow:hidden;box-shadow:0 4px 24px rgba(0,0,0,.08);">

        <!-- Header -->
        <tr>
          <td style="background:#0047BA;padding:28px 32px;">
            <p style="margin:0;font-size:22px;font-weight:700;color:#ffffff;letter-spacing:-0.5px;">Corsusa Internacional S.A.C.</p>
            <p style="margin:6px 0 0;font-size:13px;color:#93b5e5;">Gestión de Almacén · Pedido #{numero_str}</p>
          </td>
        </tr>

        <!-- Cuerpo -->
        <tr>
          <td style="padding:32px 32px 24px;font-size:14px;color:#374151;line-height:1.7;">
            <p style="margin:0 0 16px;">Estimados,</p>
            <p style="margin:0 0 16px;">
              Se comunica que la solicitud de pedido correspondiente al
              <strong style="color:#111827;">{asunto}</strong>
              se encuentra actualmente
              <span style="display:inline-block;padding:2px 10px;background:#eff6ff;color:#1d4ed8;font-weight:700;border-radius:4px;font-size:13px;">{req.estado}</span>.
            </p>
            <p style="margin:0 0 16px;">
              Se viene realizando el seguimiento correspondiente a los equipos, materiales y consumibles
              requeridos, a fin de asegurar su correcta atención dentro de los plazos establecidos.
            </p>
            <p style="margin:0 0 32px;">Cualquier actualización será comunicada oportunamente.</p>
            <p style="margin:0;color:#6b7280;font-size:13px;">Atentamente,</p>
            <p style="margin:4px 0 0;font-weight:700;color:#111827;font-size:14px;">{firma}</p>
          </td>
        </tr>

        <!-- Footer -->
        <tr>
          <td style="background:#f9fafb;border-top:1px solid #e5e7eb;padding:16px 32px;">
            <p style="margin:0;font-size:11.5px;color:#9ca3af;text-align:center;">
              Correo generado desde el sistema de almacén de
              <strong style="color:#6b7280;">Corsusa Internacional S.A.C.</strong>
            </p>
          </td>
        </tr>

      </table>
    </td></tr>
  </table>
</body>
</html>"""


@router.post("/pedidos/notificar-seguimiento")
async def notificar_seguimiento_pedido(req: NotificacionSeguimientoRequest):
    """Envía un correo de seguimiento de estado del pedido via Microsoft Graph."""
    if not req.destinatarios:
        raise HTTPException(
            status_code=400, detail="Se requiere al menos un destinatario.")

    numero_str = str(req.numero).zfill(4)

    try:
        token = await _get_graph_token()
    except HTTPException as e:
        raise e

    subject = f"[Pedido #{numero_str}] Seguimiento — {req.estado}"
    html = _build_seguimiento_html(req)

    payload = {
        "message": {
            "subject": subject,
            "body": {"contentType": "HTML", "content": html},
            "toRecipients": [
                {"emailAddress": {"address": d}} for d in req.destinatarios
            ],
        },
        "saveToSentItems": True,
    }

    sender = req.from_email or _FROM_EMAIL

    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"https://graph.microsoft.com/v1.0/users/{sender}/sendMail",
            headers={"Authorization": f"Bearer {token}",
                     "Content-Type": "application/json"},
            json=payload,
        )

    if r.status_code not in (200, 202):
        raise HTTPException(
            status_code=502,
            detail=f"Error al enviar correo: {r.status_code} — {r.text[:400]}",
        )

    return {"ok": True, "from": sender, "to": req.destinatarios, "subject": subject}


@router.post("/pedidos/notificar-enviado")
async def notificar_pedido_enviado(req: NotificacionPedidoRequest):
    """
    Envía un correo de notificación via Microsoft Graph cuando un pedido
    pasa al estado 'Enviado'.

    Requiere permiso de aplicación 'Mail.Send' en Azure AD.
    """
    numero_str = str(req.numero).zfill(4)

    try:
        token = await _get_graph_token()
    except HTTPException as e:
        raise e

    subject = f"[Pedido #{numero_str}] Material en camino — Corsusa Almacén"
    html = _build_email_html(req)

    payload = {
        "message": {
            "subject": subject,
            "body": {
                "contentType": "HTML",
                "content":     html,
            },
            "toRecipients": [
                {"emailAddress": {"address": _TO_EMAIL}},
            ],
        },
        "saveToSentItems": True,
    }

    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"https://graph.microsoft.com/v1.0/users/{_FROM_EMAIL}/sendMail",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type":  "application/json",
            },
            json=payload,
        )

    if r.status_code not in (200, 202):
        raise HTTPException(
            status_code=502,
            detail=f"Error al enviar correo via Graph: {r.status_code} — {r.text[:400]}",
        )

    return {"ok": True, "from": _FROM_EMAIL, "to": _TO_EMAIL, "subject": subject}
