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
all_model_names = []
all_models_list = []

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
        data=all_model_names,
    )

## ---------------------------------------------------------------------------------------------------------------------

@app.on_event("startup")
async def start():
    for (name, display_name, ctor) in all_models_list:
        loaded_models[display_name] = ctor(name)
        all_model_names.append(display_name)

@app.on_event("shutdown")
async def stop():
    loaded_models.clear()

## ---------------------------------------------------------------------------------------------------------------------

@click.command()
@click.option("--host", default="127.0.0.1", help="Host address to bind to")
@click.option("--port", default=5000, help="Port to bind to")
@click.option("--reload/--no-reload", default=False, help="Enable auto-reload during development")
@click.option("--save-models-to", default=None, help="Save models to this directory, then exit")
@click.option("--load-models-from", default=None, help="Load models from this directory")
def main(host, port, reload, save_models_to, load_models_from):
    global all_models_list

    models = [
        ('all-MiniLM-L6-v2', 'all-MiniLM-L6-v2', SentenceTransformer)
    ]

    if save_models_to != None:
        print("Saving models to '{}' and exiting...\n".format(save_models_to))
        for (name, display_name, ctor) in models:
            print("  : {} -> {}/{}".format(display_name, save_models_to, display_name))
            m = ctor(name)
            m.save("{}/{}".format(save_models_to, display_name))
            print()
    else:
        if load_models_from == None:
            print("Loading models from $TRANSFORMERS_CACHE and/or network...")
            all_models_list = models
        else:
            print("Loading models from '{}'...".format(load_models_from))
            for (name, display_name, ctor) in models:
                all_models_list.append(('{}/{}'.format(load_models_from, display_name), display_name, ctor))

        config = Config()
        config.bind = [ "{}:{}".format(host, port) ]
        config.use_reloader = reload
        asyncio.run(serve(app, config))

if __name__ == "__main__":
    main()
