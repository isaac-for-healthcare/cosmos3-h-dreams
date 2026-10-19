# Cosmos3-H-Dreams development and packaging

## Versions

The product version is declared in `cosmos3_h_dreams/pyproject.toml` under
`project.version`. The included FlashDreams framework has its own version in
`flashdreams/flashdreams/_version.py`; it is a separate package.

To change the product version, edit its pyproject and regenerate `uv.lock` with
`uv lock`. Review and commit both files together. There is no version-sync hook
or automated publishing workflow in this tree.

## Workspace packages

| Path | Distribution | Python import |
| --- | --- | --- |
| `cosmos3_h_dreams/` | `flashdreams-cosmos3-h-dreams` | `cosmos3_h_dreams` |
| `flashdreams/` | `flashdreams` | `flashdreams` |

Cosmos3-H-Dreams uses the workspace FlashDreams package with its `serving` extra.
Use the workspace checkout for development:

```bash
uv sync --package flashdreams-cosmos3-h-dreams --group test --locked
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for CPU and browser checks. Full dependency
installation may require CUDA build tools; checks in an existing environment
can run without resynchronizing it.

## Local distribution builds

From the workspace root:

```bash
uv lock --check
uv build --package flashdreams-cosmos3-h-dreams
```

This writes a wheel and source archive to `dist/`; it does not publish them.
The product package includes its Python modules, browser resources, LICENSE,
and NOTICE. Session YAML presets live outside the Python module, and the root
THIRD-PARTY-NOTICES is repository collateral; these are not included in the
product distributions by the current packaging configuration. Model checkpoints
and scene assets are supplied separately.

The product dependency on FlashDreams is resolved locally by the workspace.
Before distributing the product wheel on its own, verify that its unpinned
`flashdreams` dependency resolves to a compatible framework release and
that users receive the configuration examples and attribution files they need.
A local build alone does not establish that a standalone installation works.
