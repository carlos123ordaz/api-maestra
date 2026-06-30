import os
import pathlib
from dotenv import load_dotenv

load_dotenv(pathlib.Path(__file__).parent / ".env")

MS_CLIENT_ID      = os.getenv("MS_CLIENT_ID", "")
MS_CLIENT_SECRET  = os.getenv("MS_CLIENT_SECRET", "")
MS_TENANT_ID      = os.getenv("MS_TENANT_ID", "")
FRONTEND_URL      = os.getenv("FRONTEND_URL", "https://project-tracker-prod.vercel.app")
API_URL           = os.getenv("API_URL", "")
GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY", "")
SUPABASE_URL      = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")
