# Put the code for your API here.

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def say_hello():
    return {"greeting": "hello world"}
