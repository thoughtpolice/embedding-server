#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2022 Austin Seipp
# SPDX-License-Identifier: MIT OR Apache-2.0

## ---------------------------------------------------------------------------------------------------------------------

import time
import base64

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

## ---------------------------------------------------------------------------------------------------------------------

loaded_models = {}
model_list = [
    ('all-MiniLM-L6-v2', SentenceTransformer, ('sentence-transformers/all-MiniLM-L6-v2',)),
]

## ---------------------------------------------------------------------------------------------------------------------

app = FastAPI()

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

@app.get("/encode")
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

## ---------------------------------------------------------------------------------------------------------------------

@app.on_event("startup")
async def start():
    for (name, ctor, args) in model_list:
        loaded_models[name] = ctor(*args)

@app.on_event("shutdown")
async def stop():
    loaded_models.clear()

## ---------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("embedding-server:app", host="127.0.0.1", port=5000, log_level="info")