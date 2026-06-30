import os
import pathlib
from dotenv import load_dotenv

_ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(_ENV_PATH)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers.extraction import router as extraction_router
from routers.auth import router as auth_router
from routers.cotizaciones import router as cotizaciones_router
from routers.solicitudes import router as solicitudes_router
from routers.forms import router as forms_router
from routers.almacen import router as almacen_router
from routers.bitrix.tasks import router as bitrix_tasks_router
from routers.bitrix.products import router as bitrix_products_router
from routers.bitrix.deals import router as bitrix_deals_router
from routers.bitrix.activities import router as bitrix_activities_router

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

app.include_router(extraction_router)
app.include_router(auth_router)
app.include_router(cotizaciones_router)
app.include_router(solicitudes_router)
app.include_router(forms_router)
app.include_router(almacen_router)
app.include_router(bitrix_tasks_router)
app.include_router(bitrix_products_router)
app.include_router(bitrix_deals_router)
app.include_router(bitrix_activities_router)


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
