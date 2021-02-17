import uvicorn
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

# Python 直接執行 python main.py
if __name__ == "__main__":
    uvicorn.run(app, port=8000, host='0.0.0.0')