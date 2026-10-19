---
name: maintaining-oss-state
description: Maintains Cosmos3-H-Dreams release attribution, SPDX headers, REUSE metadata, dependency notices, and lockfile consistency. Use when changing dependencies or licensing collateral, adding third-party source or example data, or reviewing release attribution.
---

# Maintaining Cosmos3-H-Dreams OSS state

Keep release attribution consistent with the files and dependencies actually
shipped. Follow the repository's `AGENTS.md` and the user's scope. The included
FlashDreams framework is a separate dependency; this skill does not authorize
changes to its code or metadata.

Paths below are relative to the repository root.

## Release files

| File | Purpose |
| --- | --- |
| `LICENSE` | Project license and full Apache-2.0 text |
| `LICENSES/` | Included license texts; retain texts required by remaining material |
| `NOTICE` | Concise project attribution carried with distributions |
| `THIRD-PARTY-NOTICES` | Runtime dependency attribution, selected optional/development components, and source redistribution disclosures |
| `REUSE.toml` | Copyright and license annotations for files without inline headers and any upstream-specific overrides |
| `CONTRIBUTING.md` | Contribution terms, verbatim DCO text, and sign-off instructions |
| `cosmos3_h_dreams/LICENSE` and `cosmos3_h_dreams/NOTICE` | Product package licensing files included by its build configuration |
| `cosmos3_h_dreams/pyproject.toml` and `uv.lock` | Product dependencies and resolved workspace versions |

Read current contents before changing them. Do not describe the notices as a
complete dependency inventory unless they have been checked as such. Preserve
canonical license texts: the project LICENSE may have a preamble, so it need
not be byte-identical to `LICENSES/Apache-2.0.txt`.

## Dependencies

For a product dependency change:

1. Check its actual use and place it in runtime dependencies, an optional extra,
   or a development group accordingly. Avoid changing the framework for a
   product-only requirement.
2. Verify the license against the upstream license or the resolved distribution's
   license files. Record the package name, license identifier, and public upstream
   URL in the appropriate section of `THIRD-PARTY-NOTICES`.
3. Update the owning pyproject and regenerate `uv.lock` from the repository root.
   Review source indexes and resolved versions; keep declarations and lockfile
   changes together.
4. For removals, check remaining consumers before deleting an attribution or
   license text. For upgrades, check whether terms or bundled components changed.

Distinguish dependencies required by Cosmos3-H-Dreams from optional FlashDreams
extras. The product uses MJPEG and declares aiohttp directly; it does not request
WebRTC dependencies or the framework serving extra.

Routine dependency changes belong in THIRD-PARTY-NOTICES. Update NOTICE when
project or redistributed-source attribution changes, rather than copying the
whole dependency table there. Flag uncertain provenance or license compatibility
for the maintainer; do not claim clearance that the available evidence does not
establish.

## Source, models, and example data

For bundled third-party material, record its public source, version or dataset
subset where available, copyright holder, license, and any modifications.
Preserve required upstream headers and attribution. Include required license
texts and update THIRD-PARTY-NOTICES, NOTICE, and REUSE annotations as applicable.

The default REUSE annotation labels files NVIDIA-authored Apache-2.0. Do not
silently apply that default to external code, images, trajectories, or other
dataset samples. Use an appropriate path-specific annotation when their terms
or authorship differ. Public availability alone does not establish the right
to redistribute under the project's license.

Distinguish bundled material from separately downloaded model assets; identify
the latter's upstream terms without implying they inherit the code license.
For an independent implementation of a published method, cite the paper in the
relevant documentation. Do not label it as vendored upstream source. If source
was copied or adapted, separately account for that source's license and notices.

## Headers and contribution terms

Use the source-header pattern in `CONTRIBUTING.md` for new first-party files.
Keep SPDX identifiers near the start of the file. Preserve existing copyright
years and upstream notices; do not replace another author's attribution with
the project's header.

Use REUSE annotations for non-source files where inline headers are unsuitable.
An `aggregate` annotation combines with existing information; an `override`
annotation takes precedence. Select the behavior deliberately and keep it
scoped to the relevant paths.

Preserve the Developer Certificate of Origin text verbatim and retain its
section anchor and commit sign-off instructions when editing CONTRIBUTING.

## Validation

Select checks relevant to the change from `AGENTS.md` and `CONTRIBUTING.md`.
For metadata changes, use:

```bash
uv lock --check
git diff --check
```

If REUSE tooling is available, run `reuse lint`; report when it was not run.
For packaging changes, build the affected distribution and inspect the wheel
and source archive for required notices, licenses, and resources. A file at the
repository root is not automatically included in a package. The current product
build declares LICENSE and NOTICE as license files; verify any additional
collateral explicitly.

Do not assume a CI workflow or publication process exists: inspect the current
tree. Report which checks passed, which could not run, and unresolved attribution
questions. A passing syntax, build, or metadata check does not prove complete
license coverage or a working GPU installation.
