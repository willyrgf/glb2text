# glb2text

A command-line tool to extract human-readable information from GLB/GLTF 3D model files.

## Features

- Extract scenes, nodes, meshes, materials, textures, skins/skeletons, and animations
- Output formats: plain text, Markdown, or JSON
- No dependencies on Blender or other 3D software
- Works with both `.glb` and `.gltf` files

## Installation

### Via Nix Profile

```bash
nix profile install github:willyrgf/glb2text
```

### Run without installing

```bash
nix run github:willyrgf/glb2text -- <file.glb> [options]
```

## Usage

```bash
glb2text <file.glb> [options]
```

### Options

- `-o, --output FILE` - Write output to file instead of stdout
- `--markdown` - Output in Markdown format
- `--json` - Output in JSON format (great for piping to `jq`)
- `--max-depth N` - Maximum depth for node hierarchy (default: 3)
- `--verbose` - Print extra information to stderr

### Examples

#### Basic usage
```bash
glb2text model.glb
```

#### Save to file
```bash
glb2text model.glb -o report.txt
```

#### Markdown output
```bash
glb2text model.glb --markdown > report.md
```

#### JSON output with jq
```bash
# Get all animations
glb2text model.glb --json | jq .animations

# Get skeleton/skins
glb2text model.glb --json | jq .skeleton

# Get meshes
glb2text model.glb --json | jq .meshes

# Get animation names
glb2text model.glb --json | jq '.animations[].name'
```

## Development

```bash
# Run from source
nix run .#default -- <file.glb>

# Build
nix build
```
