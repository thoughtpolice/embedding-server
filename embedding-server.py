#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2022 Austin Seipp
# SPDX-License-Identifier: MIT OR Apache-2.0

## ---------------------------------------------------------------------------------------------------------------------

import base64
import time

import click

import asyncio
from hypercorn.asyncio import serve
from hypercorn.config import Config

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

import prometheus_client

## ---------------------------------------------------------------------------------------------------------------------

loaded_models = {}
all_model_names = []
all_models_list = []

## ---------------------------------------------------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    for (name, display_name, ctor, kwargs) in all_models_list:
        loaded_models[display_name] = ctor(name, **kwargs)
        all_model_names.append(display_name)
    yield
    loaded_models.clear()

app = FastAPI(lifespan=lifespan)

## ---------------------------------------------------------------------------------------------------------------------

class EmbeddingRequest(BaseModel):
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

ENCODE_REQUEST_TIME = prometheus_client.Summary("encode_request_processing_seconds", "Time spent processing embedding request")

@app.post("/v1/embeddings")
async def embeddings(request: EmbeddingRequest) -> EmbeddingResponse:
    m = request.model
    if not m in loaded_models:
        raise HTTPException(
            status_code=400,
            detail="Model '{}' does not exist".format(m),
        )

    nomic_matryoshka = False
    if m in [ "nomic-embed-text-v1.5" ]:
        nomic_matryoshka = True

    inputs = request.input if type(request.input) == list else [request.input]

    embeddings = []
    with ENCODE_REQUEST_TIME.time():
        # XXX FIXME (aseipp): respect .max_seq_length here!
        if not nomic_matryoshka:
            v = loaded_models[m].encode(inputs, device="cpu")
        else:
            # nomic matryoshka models use a custom design and require a
            # layer_norm to be applied after encoding. they may be freely
            # truncated to the desired dimensionality after this step
            v = loaded_models[m].encode(inputs, device="cpu", convert_to_tensor=True)
            v = F.layer_norm(v, normalized_shape=(v.shape[1],))
        [ embeddings.append(x) for x in v.tolist() ]

    vectors = []
    for i, s in enumerate(embeddings):
        vectors.append(EmbeddingObject(
            object="embedding",
            index=i,
            dims=len(s),
            embedding=s,
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

prom_app = prometheus_client.make_asgi_app()
app.mount("/metrics", prom_app)

@click.command()
@click.option("--host", default="127.0.0.1", help="Host address to bind to")
@click.option("--port", default=5000, help="Port to bind to")
@click.option("--reload/--no-reload", default=False, help="Enable auto-reload during development")
@click.option("--save-models-to", default=None, help="Save models to this directory, then exit")
@click.option("--load-models-from", default=None, help="Load models from this directory")
def main(host, port, reload, save_models_to, load_models_from):
    global all_models_list

    models = [
        ('all-MiniLM-L6-v2', 'all-MiniLM-L6-v2', SentenceTransformer, {
            'revision': 'e4ce9877abf3edfe10b0d82785e83bdcb973e22e',
        }),
        ('nomic-ai/nomic-embed-text-v1', 'nomic-embed-text-v1', SentenceTransformer, {
            'trust_remote_code': True,
            'revision': '02d96723811f4bb77a80857da07eda78c1549a4d',
        }),
        ('nomic-ai/nomic-embed-text-v1.5', 'nomic-embed-text-v1.5', SentenceTransformer, {
            'trust_remote_code': True,
            'revision': '7a5549b77c439ed64573143699547131d4218046',
        })
    ]

    if save_models_to != None:
        print("Saving models to '{}' and exiting...\n".format(save_models_to))
        for (name, display_name, ctor, kwargs) in models:
            print("  : {} -> {}/{}".format(display_name, save_models_to, display_name))
            m = ctor(name, **kwargs)
            m.save("{}/{}".format(save_models_to, display_name))
            print()
    else:
        if load_models_from == None:
            print("Loading models from $TRANSFORMERS_CACHE and/or network...")
            all_models_list = models
        else:
            print("Loading models from '{}'...".format(load_models_from))
            for (name, display_name, ctor, kwargs) in models:
                all_models_list.append(('{}/{}'.format(load_models_from, display_name), display_name, ctor, kwargs))

        config = Config()
        config.bind = [ "{}:{}".format(host, port) ]
        config.use_reloader = reload
        asyncio.run(serve(app, config))

if __name__ == "__main__":
    main()
