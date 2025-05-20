import sys; print(f"SYS.PATH IN APP.PY: {sys.path}")

from fastapi import FastAPI
from service.registry import list_supported_env_types, register_environment
from service.core_routes import api_router

# Register environments at import time (so registry is populated immediately)
import examples.sokoban.environment as sok
register_environment("Sokoban", sok.SokobanEnvironment)
import examples.crafter_classic.environment as cc
register_environment("CrafterClassic", cc.CrafterClassicEnvironment)
import examples.math.environment as me
register_environment("HendryksMath", me.HendryksMathEnv)
import examples.swe_bench.environment as sb
register_environment("SWE-bench-Env", sb.SweBenchEnvironment)

app = FastAPI(title="Environment Service")

@app.on_event("startup")
async def startup_event():
    # Ready to serve
    pass

# Mount the main API router under /env
app.include_router(api_router, prefix="/env", tags=["env"])
