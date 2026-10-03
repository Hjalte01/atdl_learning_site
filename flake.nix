{
  description = "ATDL authoring and headless browser verification";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  outputs = { nixpkgs, ... }: {
    devShells = nixpkgs.lib.genAttrs [ "x86_64-linux" "aarch64-linux" ] (system:
      let pkgs = nixpkgs.legacyPackages.${system}; in {
        default = pkgs.mkShell {
          packages = [ pkgs.nodejs_22 pkgs.chromium pkgs.poppler-utils
            (pkgs.python3.withPackages (ps: [ ps.playwright ])) ];
          ATDL_CHROMIUM = "${pkgs.chromium}/bin/chromium";
          PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD = "1";
        };
      });
  };
}
