# Contributing to Cosmos3-H-Dreams

Cosmos3-H-Dreams provides action replay and headset browser inference on
FlashDreams. Contributions are licensed under [Apache-2.0](LICENSE).

## Submitting a pull request

1. Create a branch and keep the change focused. Discuss substantial features
   in an issue before implementing them.
2. Describe the problem, the resulting behavior, and how you validated it.
3. Add a regression check for non-trivial behavior changes and run the relevant
   checks below. Report GPU and headset validation separately from CPU tests.
4. Sign off your commits with `git commit -s` and open a pull request in this
   repository.

## Repository layout

| Path | Purpose |
| --- | --- |
| `cosmos3_h_dreams/cosmos3_h_dreams/` | Product application, model runtime, and HTTPS browser flow |
| `cosmos3_h_dreams/configs/` | Session presets and example configuration |
| `cosmos3_h_dreams/tests/` | Product Python and browser tests |
| `flashdreams/` | Included framework dependency |
| `tests/` | Repository checks and framework test utilities |
| `scripts/benchmark/` | Framework benchmark and plotting tools |
| `skills/` | Task-specific agent guidance |

Put product behavior and tests in `cosmos3_h_dreams/`. The product depends on
FlashDreams; the framework must not depend on the product. Keep framework
changes separate and agree their scope with the maintainer before making them.

The release name is Cosmos3-H-Dreams. The distribution remains
`flashdreams-cosmos3-h-dreams`, the Python import is `cosmos3_h_dreams`, and the
application entry point is `cosmos3-h-dreams`.

## Development and testing

Use Python 3.12, selected by the repository's `.python-version` file. The product
declares `>=3.12,<3.15`; Python 3.15 is excluded because the locked PyTorch release
has no compatible wheel. Install the product and test tooling from the workspace
root:

```bash
uv sync --package flashdreams-cosmos3-h-dreams --group test --locked
```

The browser JavaScript checks require Node.js 14.18.0 or newer; use a currently
supported LTS release. They load built-in modules with `require("node:...")`,
which Node.js 12 does not support. Check your version with `node --version`.
Node.js is test tooling; inference runs in Python.

With the environment installed, run the product and repository CPU checks:

```bash
PYTHONPATH=cosmos3_h_dreams:flashdreams .venv/bin/python -m pytest \
  cosmos3_h_dreams/tests tests/test_dependency_policy.py tests/test_agent_skills.py \
  -m ci_cpu -q
node cosmos3_h_dreams/tests/test_headset_status.cjs
node cosmos3_h_dreams/tests/test_scene_selector.cjs
uv lock --check
```

Browser transport tests use local loopback sockets and fake model sessions.
GPU generation, live headset sessions, and checkpoint downloads are separate
validation steps that require appropriate hardware and assets.

Use `test_*.py` files and `test_*` functions. Every pytest test must carry exactly
one tier marker: `ci_cpu` for CPU-safe checks, `ci_gpu` for GPU checks, or `manual`
for environment-specific or expensive checks. For a CPU-only module, use:

```python
import pytest

pytestmark = pytest.mark.ci_cpu
```

The repository's pytest plugin enforces tier markers. Keep
`--import-mode=importlib` in the pytest configuration so duplicate test filenames
in workspace packages remain distinct. Plain `pytest` also collects manual tests;
select the intended tier explicitly.

## Coding conventions

Type-annotate new code and follow the existing patterns. Prefer small changes,
existing helpers, and standard-library facilities over new dependencies.
Formatting and type-checking tools are configured in `.pre-commit-config.yaml`:

```bash
uv run --locked --group lint pre-commit run -a
```

New source files must carry the project's SPDX header, preserving any required
upstream attribution:

  ```python
  # SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  # SPDX-License-Identifier: Apache-2.0
  #
  # Licensed under the Apache License, Version 2.0 (the "License");
  # you may not use this file except in compliance with the License.
  # You may obtain a copy of the License at
  #
  # http://www.apache.org/licenses/LICENSE-2.0
  #
  # Unless required by applicable law or agreed to in writing, software
  # distributed under the License is distributed on an "AS IS" BASIS,
  # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  # See the License for the specific language governing permissions and
  # limitations under the License.
  ```

## Dependencies and release metadata

Declare product runtime dependencies in `cosmos3_h_dreams/pyproject.toml`. Keep
test tooling in development extras or dependency groups. Update `uv.lock` and
[THIRD-PARTY-NOTICES](THIRD-PARTY-NOTICES) alongside dependency changes.
The locked versions define the reproducible environment; minimum version bounds
are not a claim that every allowed combination has been tested.

See [DEV.md](DEV.md) for versioning and local package builds. Do not commit
checkpoints, recordings, credentials, or private keys.

## Developer Certificate of Origin (DCO)

This project accepts contributions under Apache-2.0. All contributions require
sign-off under the [Developer Certificate of Origin](https://developercertificate.org/).

### Signing Your Work

We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

Any contribution which contains commits that are not Signed-Off will not be accepted.

To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:

```bash
git commit -s -m "Add cool feature."
```

This will append the following to your commit message:

```
Signed-off-by: Your Name <your@email.com>
```

### Full text of the DCO

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

## Licensing of contributions

Preserve upstream copyright and license notices. Record the origin and terms
of contributed third-party source in [REUSE.toml](REUSE.toml) and
[THIRD-PARTY-NOTICES](THIRD-PARTY-NOTICES), and include required license texts.
Resolve licensing questions before submitting third-party code.

## Reporting issues and conduct

Report bugs and feature requests in this repository's issue tracker. Include
reproduction steps and relevant configuration without credentials or private data.
For security reports, follow [SECURITY.md](SECURITY.md). Participation is subject
to the [Code of Conduct](CODE_OF_CONDUCT.md).
