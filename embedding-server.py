#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2022 Austin Seipp
# SPDX-License-Identifier: MIT OR Apache-2.0

## ---------------------------------------------------------------------------------------------------------------------

import time
import base64

import click

import asyncio
from hypercorn.asyncio import serve
from hypercorn.config import Config

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

## ---------------------------------------------------------------------------------------------------------------------

loaded_models = {}
model_list = [
    ('all-MiniLM-L6-v2', SentenceTransformer, ('sentence-transformers/all-MiniLM-L6-v2',)),
]

app = FastAPI()

## ---------------------------------------------------------------------------------------------------------------------

class EmbeddingRequest(BaseModel):
    user: str | None = None
    model: str
    input: str | list[str]

class EmbeddingObject(BaseModel):
    index: int
    object: str
    embedding: list[float]
    dims: int

class EmbeddingResponse(BaseModel):
    model: str
    object: str
    data: list[EmbeddingObject]

@app.get("/v1/encode")
async def encode(request: EmbeddingRequest) -> EmbeddingResponse:
    m = request.model
    if not m in loaded_models:
        raise HTTPException(
            status_code=400,
            detail="Model '{}' does not exist".format(m),
        )

    if type(request.input) == list:
        inputs = request.input
    else:
        inputs = [request.input]

    vectors = []
    for i, s in enumerate(inputs):
        # XXX FIXME (aseipp): respect .max_seq_length here!
        v = loaded_models[m].encode(s, device="cpu")
        vectors.append(EmbeddingObject(
            object="embedding",
            index=i,
            dims=len(v),
            embedding=v.tolist()
        ))

    return EmbeddingResponse(
        model=m,
        object="list",
        data=vectors,
    )

class ModelListResponse(BaseModel):
    object: str
    data: list[str]

@app.get("/v1/models")
async def models() -> ModelListResponse:
    return ModelListResponse(
        object="list",
        data=list(loaded_models.keys()),
    )

## ---------------------------------------------------------------------------------------------------------------------

@app.on_event("startup")
async def start():
    for (name, ctor, args) in model_list:
        loaded_models[name] = ctor(*args)

@app.on_event("shutdown")
async def stop():
    loaded_models.clear()

## ---------------------------------------------------------------------------------------------------------------------

@click.command()
@click.option("--host", default="127.0.0.1", help="Host address to bind to")
@click.option("--port", default=5000, help="Port to bind to")
def main(host, port):
    config = Config()
    config.bind = [ "{}:{}".format(host, port) ]
    asyncio.run(serve(app, config))

if __name__ == "__main__":
    main()
