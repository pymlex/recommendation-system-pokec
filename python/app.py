import os
import yaml
import subprocess
import time
import threading
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pyngrok import ngrok


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
CONFIG_PATH = os.path.join(ROOT, "config.yaml")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

NGROK_TOKEN = cfg.get("ngrok_token") or os.environ.get("NGROK_AUTH_TOKEN") or ""
LOAD_USERS = cfg.get("load_users", 100000)
HOST = cfg.get("server", {}).get("host", "0.0.0.0")
PORT = cfg.get("server", {}).get("port", 8000)

API_CLI_PATH = os.path.join(ROOT, "build", "api_cli.exe")
if not os.path.exists(API_CLI_PATH):
    raise RuntimeError(f"api_cli.exe not found at {API_CLI_PATH}. Build C++ first.")

class APICLI:
    def __init__(self, exe_path, load_users):
        cmd = [exe_path, str(int(load_users))] if load_users else [exe_path]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        ready = False
        start = time.time()
        while True:
            line = self.p.stdout.readline()
            if not line:
                if (time.time() - start) > 120:
                    raise RuntimeError("api_cli did not start or produce READY")
                time.sleep(0.01)
                continue
            line = line.strip()
            if line == "READY":
                ready = True
                break
            # forward stderr lines to Python stderr
            print("[api_cli]", line)
            if (time.time() - start) > 120:
                break
        if not ready:
            raise RuntimeError("api_cli did not signal READY")
        self.lock = threading.Lock()

    def send(self, cmd, timeout=10.0):
        with self.lock:
            try:
                self.p.stdin.write(cmd.rstrip("\n") + "\n")
                self.p.stdin.flush()
            except Exception as e:
                raise RuntimeError("failed write to api_cli: " + str(e))
            start = time.time()
            out = ""
            while True:
                line = self.p.stdout.readline()
                if line is None:
                    raise RuntimeError("api_cli closed output")
                if line.strip() == "":
                    continue
                out = line.strip()
                break
                if (time.time() - start) > timeout:
                    raise TimeoutError("timeout waiting for api_cli response")
            return out

    def close(self):
        try:
            self.send("EXIT")
        except:
            pass
        try:
            self.p.kill()
        except:
            pass

api_cli = APICLI(API_CLI_PATH, LOAD_USERS)

app = FastAPI(title="Pokec Recommender API (C++ backend)")

templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
#app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

@app.on_event("shutdown")
def shutdown_event():
    api_cli.close()

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "loaded_users": LOAD_USERS})

@app.get("/health")
async def health():
    return {"status":"ok", "load_users": LOAD_USERS}

@app.get("/api/user/{uid}")
async def api_user(uid: int):
    cmd = f"USER {uid}"
    try:
        out = api_cli.send(cmd)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    try:
        import json
        j = json.loads(out)
        return JSONResponse(j)
    except Exception as e:
        raise HTTPException(status_code=500, detail="invalid JSON from backend: " + str(e) + " output: " + out)

@app.get("/api/recommend/graph/{uid}")
async def api_recommend_graph(uid: int, topk: int = 20):
    j = await api_user(uid)
    recs = j.get("recommendations", {}).get("graph", [])
    return recs[:topk]

@app.get("/api/recommend/collab/{uid}")
async def api_recommend_collab(uid: int, topk: int = 20):
    j = await api_user(uid)
    recs = j.get("recommendations", {}).get("collaborative", [])
    return recs[:topk]

@app.get("/api/recommend/interest/{uid}")
async def api_recommend_interest(uid: int, topk: int = 20):
    j = await api_user(uid)
    recs = j.get("recommendations", {}).get("interest", [])
    return recs[:topk]

@app.get("/api/recommend/clubs/{uid}")
async def api_recommend_clubs(uid: int, topk: int = 20):
    j = await api_user(uid)
    recs = j.get("recommendations", {}).get("clubs", [])
    return recs[:topk]

if __name__ == "__main__":
    """
    if NGROK_TOKEN:
        ngrok.set_auth_token(NGROK_TOKEN)
        public_url = ngrok.connect(PORT)
        print("ngrok public url:", public_url)
    """
    import uvicorn
    print("Server starting: FastAPI wrapper (C++ backend). Loaded users:", LOAD_USERS)
    uvicorn.run("app:app", host=HOST, port=PORT, log_level="info")
