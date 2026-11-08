# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned settings for the matched Colossus control-to-browser measurement."""
import dataclasses,hashlib,json,os,platform,subprocess,sys
from pathlib import Path
import torch
from cosmos3_h_dreams.config import Cosmos3HDreamsConfig,COSMOS3_H_DREAMS_PRESETS


def config(mode,actions=None):
 values=dict(COSMOS3_H_DREAMS_PRESETS['wan-taehv-chunk2-2steps-fademem-compile'])
 values.update(checkpoint=os.environ['DREAMS_CHECKPOINT'],initial_frame=Path('assets/example_data/scenes/example_frame.png'),actions=actions,seed=11,profile=False)
 if mode=='latest':
  values.update(compile_cache_refresh=True,reuse_cache_rope=True,defer_cache_refresh=True,fp8_weights=Path(os.environ['DREAMS_FP8_WEIGHTS']) if os.environ.get('DREAMS_FP8_WEIGHTS') else None)
 values['nvfp4_weights']=Path(os.environ['DREAMS_NVFP4_WEIGHTS']) if os.environ.get('DREAMS_NVFP4_WEIGHTS') else None
 result=Cosmos3HDreamsConfig(**values)
 assert result.chunk_size==2 and result.denoise_steps==2 and result.fps==10
 return result


def metadata(mode,cfg):
 protocol=json.loads(Path(os.environ['DREAMS_BENCH_PROTOCOL']).read_text())
 commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
 assert not subprocess.check_output(['git','diff','HEAD','--'],text=True), 'Tracked source changed during the matrix'
 assert commit==protocol['commits'][mode]
 return dict(mode=mode,nvfp4_weights=os.environ.get("DREAMS_NVFP4_WEIGHTS"),completed=False,source_commit=commit,config=dataclasses.asdict(cfg),command=sys.argv,python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(),gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),driver=subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip(),compiler_cache={k:os.environ.get(k) for k in ['TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR']},assets_sha256={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in ['assets/example_data/scenes/example_frame.png','assets/example_data/scenes/trajectory.npy']},checkpoint_sha256=protocol['checkpoint_sha256'],torch_threads=torch.get_num_threads())


def assert_runtime(session,mode):
 model,rollout=session._require_runtime()
 enabled=mode=='latest'
 assert getattr(rollout,'defer_cache_refresh',False) is enabled
 assert getattr(rollout,'compile_cache_refresh',False) is enabled
 assert getattr(rollout.fademem,'reuse_cache_rope',False) is enabled
 expected_fp8=bool(os.environ.get('DREAMS_FP8_WEIGHTS') or os.environ.get('DREAMS_NVFP4_WEIGHTS'))
 assert bool(getattr(model,'quantization_info',None))==expected_fp8
 if os.environ.get("DREAMS_NVFP4_WEIGHTS"):assert "nvfp4" in model.quantization_info["recipe"]
 return rollout
