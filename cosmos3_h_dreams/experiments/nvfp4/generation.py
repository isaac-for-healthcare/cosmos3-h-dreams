# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the existing generation/lifecycle harness with a scoped attention choice."""
import argparse
import json
import sys
from pathlib import Path
import attention_variants as variants
from cosmos3_h_dreams import cache_refresh_probe as probe
from cosmos3_h_dreams.session import Cosmos3HDreamsSession

p=argparse.ArgumentParser(add_help=False)
p.add_argument('--attention-variant',choices=['default','cudnn','refresh-split'],required=True)
a,rest=p.parse_known_args()
factory=variants.model_factory(a.attention_variant)
def session(**kwargs):return Cosmos3HDreamsSession(**kwargs,model_factory=factory)
probe.Cosmos3HDreamsSession=session
out=Path(rest[rest.index('--output')+1])
sys.argv=[sys.argv[0],*rest]
try:
    probe.main()
finally:
    settings=dict(variant=a.attention_variant,records=variants.RECORDS,benchmark_only=True)
    if out.exists():(out/'attention-settings.json').write_text(json.dumps(settings,indent=2)+'\n')
