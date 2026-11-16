{
  description = "Kestrel: small conversational language model experiments";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  # The browser playground (`python -m kestrel playground`). To try local changes to it:
  #   nix develop --override-input suzunaan path:../suzunaan
  inputs.suzunaan.url = "path:/Users/lcham/devel/suzunaan";
  inputs.suzunaan.inputs.nixpkgs.follows = "nixpkgs";
  outputs = { self, nixpkgs, suzunaan }:
    let
      systems = [ "aarch64-darwin" "x86_64-darwin" "aarch64-linux" "x86_64-linux" ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system:
        f (import nixpkgs { inherit system; overlays = [ suzunaan.overlays.default ]; }));
    in {
      devShells = forAllSystems (pkgs:
        let
          pythonEnv = pkgs.python3.withPackages (ps: [
            ps.torch ps.transformers ps.tokenizers ps.numpy ps.pytest ps.suzunaan
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
