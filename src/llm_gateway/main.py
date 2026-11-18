from fastapi import FastAPI

app = FastAPI(title="LLM Gateway")


@app.get("/healthz")
def healthz():
    return {"status": "ok"}
