{
  description = "Stateless HTTP server for computing sentence embeddings";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system: let
      pkgs = import nixpkgs { inherit system; };
      python = pkgs.python3;

      /* needed libraries in all environments */
      pythonLibs = with python.pkgs; [ 
        sentence-transformers
        fastapi
        torch
        hypercorn
        click
        prometheus-client
        einops
      ];

      pythonLibsPath = python.pkgs.makePythonPath pythonLibs;

    in {
      devShells.default = pkgs.mkShell {
        /* Default shell environment. */
        packages = [ pkgs.httpie python ] ++ pythonLibs;
      };

      apps = rec {
        /* Used for 'nix run' */
        default = embedding-server;
        embedding-server = flake-utils.lib.mkApp { drv = self.packages.${system}.embedding-server; };
      };

      packages = let
        /* NB: don't export this under 'packages', no need */
        embedding-server-py = pkgs.runCommand "embedding-server" {} ''
          mkdir -p $out/libexec
          cp ${./embedding-server.py} $out/libexec/embedding-server.py
        '';
      in rec {
        /* Actual packages. 'default' is used for 'nix build' */
        default = embedding-server;

        embedding-server = pkgs.writeShellScriptBin "embedding-server" ''
          export PYTHONPATH="${pythonLibsPath}"
          exec ${python}/bin/python \
            ${embedding-server-py}/libexec/embedding-server.py \
            --load-models-from ${model-data} \
            "$@"
        '';

        docker-image = pkgs.dockerTools.buildLayeredImage {
          name = "embedding-server";
          tag = "latest";

          contents = [ model-data embedding-server ];
          config = {
            Cmd = [ "${embedding-server}/bin/embedding-server" "--host" "0.0.0.0" ];
            ExposedPorts = {
              "5000/tcp" = {};
            };
          };
        };

        model-data = let
          /* NB: model data uses the 'impure-derivations' feature to download
             all model datasets to the store. This is necessary because we don't
             want them downloading from the container. However, impure drvs are
             purely volatile and must be built in a separate drv from the main
             drv, and must have a fixed-output drv in-between them and any
             consumers.
          */
          real-data = pkgs.runCommand "model-data-dl" {
            __impure = true;
            impureEnvVars=pkgs.lib.fetchers.proxyImpureEnvVars; /* NB: needed for network */

            buildInputs = [
              python
              embedding-server-py
            ] ++ pythonLibs;
          } ''
            mkdir -p $out tmp
            export HOME=$PWD/tmp
            export HF_HOME=$PWD/tmp
            export PYTHONPATH="${pythonLibsPath}"

            ${python}/bin/python \
              ${embedding-server-py}/libexec/embedding-server.py \
              --save-models-to $out
            
            # FIXME: why doesn't the nomic model include this file?
            cp -v \
              $HF_HOME/hub/models--nomic-ai--nomic-embed-text-v1/snapshots/02d96723811f4bb77a80857da07eda78c1549a4d/configuration_hf_nomic_bert.py \
              $HF_HOME/hub/models--nomic-ai--nomic-embed-text-v1/snapshots/02d96723811f4bb77a80857da07eda78c1549a4d/modeling_hf_nomic_bert.py \
              $out/nomic-embed-text-v1
          '';

        /* finally, just re-package the data with a fixed-output sha256 hash */
        in pkgs.runCommand "model-data" {
          outputHashMode = "recursive";
          outputHashAlgo = "sha256";
          outputHash = "sha256-nMRQtxrJdLxAnNLGHP61nAePGWJ/ZVG6ad0v9wUBTeY=";
          passthru = { inherit real-data; };
        } "mkdir -p $out && cp -r ${real-data}/* $out";
      };
    });
}
