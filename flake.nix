{
  description = "GLB Text Extractor distributed by Nix.";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        pythonWithPkgs =
          pkgs.python3.withPackages (ps: with ps; [ pygltflib ]);

        appName = "glb2text";
        appVersion = "0.1.1";
      in {
        packages = {
          glb2text = pkgs.stdenv.mkDerivation {
            pname = appName;
            version = appVersion;
            src = self;

            nativeBuildInputs = [ pkgs.makeWrapper ];
            buildInputs = [ pythonWithPkgs ];

            dontBuild = true;

            installPhase = ''
              mkdir -p $out/bin $out/lib
              if [ -f "$src/glb2text.py" ]; then
                cp $src/glb2text.py $out/lib/glb2text.py
                makeWrapper ${pythonWithPkgs}/bin/python $out/bin/glb2text \
                  --add-flags "$out/lib/glb2text.py" \
                  --set GLB2TEXT_APP_NAME "${appName}" \
                  --set GLB2TEXT_APP_VERSION "${appVersion}"
              else
                echo "ERROR: glb2text.py not found in source directory" >&2
                exit 1
              fi
            '';
          };

          default = self.packages.${system}.glb2text;
        };

        apps = {
          glb2text = {
            type = "app";
            program = "${self.packages.${system}.glb2text}/bin/glb2text";
            meta = with pkgs.lib; {
              description = "Extract text reports from GLB/GLTF 3D model files";
              homepage = "https://github.com/willyrgf/glb2text";
              license = licenses.mit;
              platforms = platforms.all;
            };
          };

          default = self.apps.${system}.glb2text;
        };

        devShells = {
          default = pkgs.mkShell {
            name = "glb2text-dev-env";
            packages = [ pythonWithPkgs ];

            shellHook = ''
              export HISTFILE=$HOME/.history_nix
              export PYTHONPATH=${builtins.toString ./.}:$PYTHONPATH
              export PATH=${pythonWithPkgs}/bin:$PATH
              export GLB2TEXT_APP_NAME="${appName}"
              export GLB2TEXT_APP_VERSION="${appVersion}"
              alias glb2text="python ${builtins.toString ./.}/glb2text.py"
              echo "GLB2Text development environment activated"
              echo "Type 'glb2text path/to/model.glb' to run the application"
            '';
          };
        };
      });
}
