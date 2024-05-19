from fastapi import FastAPI
from fastapi import Request
from fastapi.responses import JSONResponse
from ml import pipeline
from db import Database
app = FastAPI()
database = Database()
database.automigrate()
#database.update_all_documents()

@app.post("/register")
async def register(req: Request):
    data = await req.json()
    username = data["username"]
    password = data["password"]
    res = database.register(username, password)
    if res is not None:
        return JSONResponse(content={"message": 'ok'}, status_code=200)
    else:
        return JSONResponse(content={"message": 'error'}, status_code=500)

@app.post("/login")
async def login(req: Request):
    data = await req.json()
    username = data["username"]
    password = data["password"]
    res = database.login(username, password)
    if res is not None:
        return JSONResponse(content={"api_key": res.api_key}, status_code=200)
    else:
        return JSONResponse(content={"message": 'error'}, status_code=500)

@app.get("/get_nearest_documents")
async def get_nearest_documents(req: Request):
    data = await req.json()
    text = data["text"]
    if type(text) is not list:
        text = [text]
    res = pipeline(text)
    if res is not None:
        return JSONResponse(content={"documents": res}, status_code=200)
    else:
        return JSONResponse(content={"message": 'error'}, status_code=500)

@app.get('/get_api_key')
async def get_api_key(req: Request):
    data = await req.json()
    username = data["username"]
    res = database.get_api_key(username)
    if res is not None:
        return JSONResponse(content={"api_key": res}, status_code=200)
    else:
        return JSONResponse(content={"message": 'error'}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)