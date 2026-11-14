{
  description = "Kestrel: small conversational language model experiments";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  outputs = { self, nixpkgs }:
    let
      systems = [ "aarch64-darwin" "x86_64-darwin" "aarch64-linux" "x86_64-linux" ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in {
      devShells = forAllSystems (pkgs:
        let
          pythonEnv = pkgs.python3.withPackages (ps: [
            ps.torch ps.transformers ps.tokenizers ps.numpy ps.pytest
          ]);
        in {
          default = pkgs.mkShell {
            packages = [ pythonEnv ];
            shellHook = ''
              # Editors need a stable interpreter path across input updates.
              ln -sfn ${pythonEnv} .python-env
              export TOKENIZERS_PARALLELISM=false
            '';
          };
        });
    };
}
