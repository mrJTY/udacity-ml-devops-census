from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel


# Declare the data object with its components and type
class TaggedItem(BaseModel):
    name: str
    tags: Union[str, list]  # Can be a string or list
    item_id: int


app = FastAPI()


@app.get("/")
async def say_hello():
    return {"greeting": "hello world"}


# This allows the creation of data using the TaggedItem class via a POST
@app.post("/items")
async def create_item(item: TaggedItem):
    return item
