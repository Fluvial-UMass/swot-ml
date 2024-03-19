{
  description = "Forest Flows scientific machine learning";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    poetry2nix = {
      url = "github:nix-community/poetry2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, poetry2nix }:
    flake-utils.lib.eachDefaultSystem (system:
    let
      pkgs = nixpkgs.legacyPackages.${system};
      p2n = import poetry2nix { inherit pkgs; };
    in rec
        {
          packages.default = p2n.mkPoetryApplication {
            projectDir = self;
          };

          devShells.default = pkgs.mkShell {
            packages = [
              (p2n.mkPoetryEnv { projectDir = self; preferWheels = true; })
              pkgs.poetry
              pkgs.python3Packages.pyqt6
              pkgs.python3Packages.jedi-language-server
              pkgs.python3Packages.isort
              pkgs.python3Packages.pyflakes
            ];
          };
      });
}
