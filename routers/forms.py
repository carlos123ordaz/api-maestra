import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config import SUPABASE_URL, SUPABASE_SERVICE_KEY

router = APIRouter()


class FormSubmitRequest(BaseModel):
    form_id:         str
    submitter_name:  str
    submitter_email: str
    answers:         dict
    status:          str = "Pendiente"


@router.post("/forms/submit")
async def submit_form(req: FormSubmitRequest):
    """Inserta una form_submission usando la service key de Supabase (sin RLS)."""
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise HTTPException(status_code=500, detail="Supabase no configurado en el servidor")
    url = f"{SUPABASE_URL}/rest/v1/form_submissions"
    payload = {
        "form_id":         req.form_id,
        "submitted_by":    None,
        "submitter_name":  req.submitter_name,
        "submitter_email": req.submitter_email,
        "answers":         req.answers,
        "status":          req.status,
    }
    headers = {
        "apikey":        SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type":  "application/json",
        "Prefer":        "return=representation",
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, json=payload, headers=headers)
    if resp.status_code not in (200, 201):
        raise HTTPException(status_code=502, detail=f"Supabase error {resp.status_code}: {resp.text}")
    return resp.json()
