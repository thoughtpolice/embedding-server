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
      ];

      pythonLibsPath = python.pkgs.makePythonPath pythonLibs;

    in {
      devShells.default = pkgs.mkShell {
        packages = [ python ] ++ pythonLibs;
      };

      apps = rec {
        default = embedding-server;
        embedding-server = flake-utils.lib.mkApp { drv = self.packages.${system}.embedding-server; };
      };

      packages = rec {
        default = embedding-server;

        embedding-server-py = pkgs.runCommand "embedding-server" {} ''
          mkdir -p $out/libexec
          cp ${./embedding-server.py} $out/libexec/embedding-server.py
        '';

        embedding-server = pkgs.writeShellScriptBin "embedding-server" ''
          export PYTHONPATH="${pythonLibsPath}"
          exec ${embedding-server-py}/libexec/embedding-server.py "$@"
        '';
      };
    });
}
