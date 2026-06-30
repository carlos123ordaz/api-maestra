import pathlib
import httpx

BITRIX_WEBHOOK = "https://corsusaint.bitrix24.com/rest/6238/dbakwyqx9fxrblp1"
DATA_DIR       = pathlib.Path(__file__).parent.parent.parent / "data"


async def _fetch_users_batch(user_ids: set[str]) -> dict[str, str]:
    """Retorna mapa user_id → nombre completo (reutilizable por todos los módulos Bitrix)."""
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
