from fastapi import FastAPI
from app.api.generate import router

app = FastAPI(title="Valorant Voice AI")

app.include_router(router)

@app.get("/")
def health_check():
    return {"status": "running"}
