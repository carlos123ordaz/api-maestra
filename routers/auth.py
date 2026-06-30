import base64
import urllib.parse

import httpx
from fastapi import APIRouter, Query
from fastapi.responses import RedirectResponse

from config import MS_CLIENT_ID, MS_CLIENT_SECRET, MS_TENANT_ID, FRONTEND_URL, API_URL

router = APIRouter(prefix="/auth")


@router.get("/microsoft/login")
async def ms_login(return_path: str = Query(default="/")):
    """Genera la URL de autorización de Microsoft y redirige al usuario."""
    state = base64.urlsafe_b64encode(return_path.encode()).decode().rstrip("=")
    params = urllib.parse.urlencode({
        "client_id":     MS_CLIENT_ID,
        "response_type": "code",
        "redirect_uri":  f"{API_URL}/auth/microsoft/callback",
        "scope":         "openid email profile User.Read",
        "state":         state,
        "response_mode": "query",
    })
    auth_url = f"https://login.microsoftonline.com/{MS_TENANT_ID}/oauth2/v2.0/authorize?{params}"
    return RedirectResponse(url=auth_url)


@router.get("/microsoft/callback")
async def ms_callback(
    code:  str = Query(default=""),
    error: str = Query(default=""),
    state: str = Query(default=""),
):
    """Recibe el código de Microsoft, lo intercambia por tokens y redirige al frontend con el usuario."""
    try:
        padding = "=" * (-len(state) % 4)
        return_path = base64.urlsafe_b64decode(state + padding).decode()
        if not return_path.startswith("/"):
            return_path = "/"
    except Exception:
        return_path = "/"

    error_redirect = f"{FRONTEND_URL}{return_path}?ms_error=auth_failed"

    if error or not code:
        return RedirectResponse(url=error_redirect)

    try:
        token_url = f"https://login.microsoftonline.com/{MS_TENANT_ID}/oauth2/v2.0/token"
        async with httpx.AsyncClient(timeout=15) as client:
            token_resp = await client.post(token_url, data={
                "client_id":     MS_CLIENT_ID,
                "client_secret": MS_CLIENT_SECRET,
                "code":          code,
                "grant_type":    "authorization_code",
                "redirect_uri":  f"{API_URL}/auth/microsoft/callback",
                "scope":         "openid email profile User.Read",
            })
            token_resp.raise_for_status()
            access_token = token_resp.json().get("access_token", "")

            graph_resp = await client.get(
                "https://graph.microsoft.com/v1.0/me",
                headers={"Authorization": f"Bearer {access_token}"},
            )
            graph_resp.raise_for_status()
            user_data = graph_resp.json()

        name  = user_data.get("displayName") or ""
        email = user_data.get("mail") or user_data.get("userPrincipalName") or ""

        qs = urllib.parse.urlencode({"ms_name": name, "ms_email": email})
        return RedirectResponse(url=f"{FRONTEND_URL}{return_path}?{qs}")

    except Exception:
        return RedirectResponse(url=error_redirect)
