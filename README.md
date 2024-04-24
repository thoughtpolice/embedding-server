# Sentence embeddings over HTTP

> **NB**: This is an experiment but I welcome feedback if it's useful for you!

This is a stateless server that can compute a "sentence embedding" from given
English words or sentences; the request/response model uses a simple JSON-based
HTTP API.

A sentence embedding is a way of encoding words into a vector space, such that
similar words or ontological phrases are "close together" as defined by the
distance metric of the vector space. For example, the terms "iPhone" and
"popular smartphone" can each be transformed into a real-valued vector of `N`
entries; they will be considered "close together", in such a system, while
"iPhone" and "Roku" would be further apart. This is useful for certain forms of
semantic search, for instance.

Internally this server is written in Python, and uses the
**[sentence-transformers]** library and (transitively) **[PyTorch]** to compute
embeddings. The API is served over **[FastAPI]** by way of **[hypercorn]**. The
system is packaged and developed with **[Nix]**.

Because this server is completely stateless, it can be scaled out vertically
with more workers &mdash; though, Python will likely always imply some level of
compute/latency overhead versus a more optimized solution. However, it is easy,
simple to extend, and simple to understand.

There are probably other various clones and/or copies of this idea; but this one
is mine.

[sentence-transformers]: https://www.sbert.net
[PyTorch]: https://pytorch.org
[FastAPI]: https://fastapi.tiangolo.com
[hypercorn]: https://hypercorn.readthedocs.io
[Nix]: https://nixos.org/nix

## Running the server + HTTPie Demo

You have two options to run the server, in general:

- Use **[Nix]** with `nix run` (hacking, quick ease of use)
- Docker/podman/OCI runtime (probably everywhere else)

```bash
docker run --rm -it -p 5000:5000 ghcr.io/thoughtpolice/embedding-server:latest
# OR
nix run --tarball-ttl 0 github:thoughtpolice/embedding-server
```

The server is now bound to port 5000 (the default.)

Now, you can query the model list, and then encode two independent words with
one request:

```bash
http get localhost:5000/v1/models
http get localhost:5000/v1/encode \
  model=all-MiniLM-L6-v2 \
  input:='["iPhone","Samsung"]'
```

## Prometheus support

Use the `/metrics` endpoint.

## API endpoints

The API is loosely inspired by the **[OpenAI Embedding API][openai-api]**, used
with `text-embedding-ada-002`.

[openai-api]: https://platform.openai.com/docs/guides/embeddings/what-are-embeddings

### Endpoint: `GET /v1/models`

The request is a GET request, with no body. The response is a JSON object like
follows, listing all possible models you can use with the `v1/encode` endpoint:

```json
{
  "data": [
    "all-MiniLM-L6-v2",
    "nomic-embed-text-v1"
  ],
  "object": "list"
}
```

### Endpoint: `GET /v1/encode`

The request is a GET request, with a JSON object body, containing two fields:

- `input: list[string]`
- `model: string`

The `input` can simply be a list of words or phrases; the `model` is the
supported text embedding model to use, which must be one of the options returned
from `v1/models`.

Given a JSON request:

```json
{
  "model": "all-MiniLM-L6-v2",
  "input": ["iPhone", "Samsung"]
}
```

The response JSON will look like the following:

```json
{
    "data": [
        {
            "dims": 384,
            "embedding": [
                -0.02878604456782341,
                0.024760600179433823,
                0.06652576476335526,
                ...
            ],
            "index": 0,
            "object": "embedding"
        },
        {
            "dims": 384,
            "embedding": [
                -0.13341815769672394,
                0.049686793237924576,
                0.043825067579746246,
                ...
            ],
            "index": 1,
            "object": "embedding"
        }
    ],
    "model": "all-MiniLM-L6-v2",
    "object": "list"
}
```

This is fairly self explanatory, and effectively the only possible response;
though the `object` fields will help the schema evolve in the future. The `data`
list will have a list of objects, each containing the dimensions of the vector
as well as the `index` referring to which input this embedding is for.

## Hacking

Install **[direnv]** into your shell of choice, then move into this repository,
`direnv allow` the `.envrc` file, and you can just run the
`./embedding-server.py` script directly as you wish.

The **[flake.nix](./flake.nix)** file describes the actual Nix packages that are
exported/built from this repo. In short:

```bash
nix build '.#embedding-server'
nix build '.#docker-image'
```

[direnv]: https://direnv.net

You need `impure-derivations` enabled in `experimental-features` and,
practically speaking, Nix 2.15 or later, probably, since that's what I test
with.

## Notes

This package tries using `impure-derivations` to package model data. This
feature allows us to write a Nix expression which, in its body, downloads the
model data from huggingface.co over the network; this data is then "purified"
with a fixed-output derivation.

This allows us to have a single source of truth &mdash; the
`embedding-server.py` script itself &mdash; as the source of truth for all model
data, so we don't have to manually replicate all of the downloads of each model
inside a `.nix` file. However, we _do_ have to update the `hash` of the
fixed-output derivation, and it isn't clear if the hugging face libraries can be
configured to download stable model versions. We may have to use another
approach, eventually.

## License

MIT or Apache-2.0; see the corresponding `LICENSE-*` file for details.
