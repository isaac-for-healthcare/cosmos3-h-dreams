# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark-only attention choices; preserve the existing seed/denoise/fill paths."""
import functools
import hashlib
import types

import torch
import torch.nn.functional as F

RECORDS = []


def split_attention(q, k, v, cached, *, block_tokens, enable_gqa=False):
    count = q.shape[-2]
    if block_tokens <= 0 or count % block_tokens:
        raise ValueError('Temporal block size must divide the query sequence')
    return torch.cat([
        F.scaled_dot_product_attention(
            q[..., start:start+block_tokens, :],
            k[..., :cached+start+block_tokens, :],
            v[..., :cached+start+block_tokens, :], enable_gqa=enable_gqa)
        for start in range(0, count, block_tokens)
    ], dim=2)


def refresh_function(original, original_attention, block_tokens, frames):
    def attention(q, k, v, cached, within, *, cached_mask=None, enable_gqa=False):
        if (within is None or cached_mask is not None or
                q.shape[-2] != block_tokens*frames or
                within.shape != (block_tokens*frames, block_tokens*frames)):
            return original_attention(q, k, v, cached, within,
                cached_mask=cached_mask, enable_gqa=enable_gqa)
        return split_attention(q, k, v, cached,
            block_tokens=block_tokens, enable_gqa=enable_gqa)
    # Reuse the frozen product's exact layer computation. Only its attention
    # symbol differs, and only the compiled saturated-refresh callable uses it.
    # This avoids a global monkeypatch that also changes text/seed/cache fill.
    assert original.__closure__ is None
    fn = types.FunctionType(original.__code__,
        {**original.__globals__, 'cached_joint_attention': attention},
        name=original.__name__, argdefs=original.__defaults__)
    fn.__kwdefaults__ = original.__kwdefaults__
    fn.__module__ = original.__module__
    return fn


def model_factory(variant):
    from cosmos3_h_dreams.model import Cosmos3HDreamsModel
    if variant in ('default', 'cudnn'):
        return functools.partial(Cosmos3HDreamsModel,
            sdpa_backend=variant, native_gqa_attention=False)
    if variant != 'refresh-split':
        raise ValueError(variant)
    from cosmos3_h_dreams._runtime._network import _MoTLayer, cached_joint_attention
    class RefreshSplitModel(Cosmos3HDreamsModel):
        def _new_session(self, scene):
            rollout = super()._new_session(scene)
            if rollout.fademem is None or not rollout.compile_cache_refresh:
                raise ValueError('This benchmark requires compiled FadeMem refresh')
            block, frames = rollout._per_frame_tokens, rollout.chunk_size
            # Validate the actual runtime mask once outside compilation/timing.
            actual = rollout._causal_within(frames).cpu()
            order = torch.arange(frames*block)//block
            expected = order[:, None] >= order[None, :]
            assert torch.equal(actual, expected), 'Unsupported refresh visibility rule'
            if not hasattr(self, '_probe_refresh_functions'):
                self._probe_refresh_functions = {}
            if (block, frames) not in self._probe_refresh_functions:
                fn = refresh_function(_MoTLayer.gen_compute_with_kv,
                    cached_joint_attention, block, frames)
                self._probe_refresh_functions[block, frames] = torch.compile(
                    fn, fullgraph=True, dynamic=False, mode='default')
            rollout._compiled_refresh_layer = self._probe_refresh_functions[block, frames]
            RECORDS.append(dict(variant=variant,block_tokens=block,frames=frames,
                text_tokens=rollout.n_text,mask_sha256=hashlib.sha256(actual.numpy().tobytes()).hexdigest(),
                scope='saturated compiled FadeMem refresh only',native_gqa=False))
            return rollout
    return functools.partial(RefreshSplitModel,
        sdpa_backend='default', native_gqa_attention=False)


def check_cpu():
    torch.set_num_threads(2)
    torch.manual_seed(11)
    from cosmos3_h_dreams._runtime._attention import cached_joint_attention
    # Real product layer source is reused; check the changed attention operation
    # for multiple geometries, histories, frame counts and grouped head counts.
    for block, frames, history in ((5,1,0),(7,2,13),(11,3,17),(394,2,43)):
        count = block*frames
        q = torch.randn(1,4,count,8)
        k = torch.randn(1,2,history+count,8);v=torch.randn_like(k)
        order = torch.arange(count)//block
        within = order[:,None]>=order[None,:]
        for gqa in (False,True):
            keys = k if gqa else k.repeat_interleave(2,1)
            vals = v if gqa else v.repeat_interleave(2,1)
            ref = cached_joint_attention(q,keys,vals,history,within,enable_gqa=gqa)
            got = split_attention(q,keys,vals,history,block_tokens=block,enable_gqa=gqa)
            torch.testing.assert_close(got,ref,atol=2e-6,rtol=2e-5)
    # Rebinding affects this callable alone; fallback preserves padded masks.
    from cosmos3_h_dreams._runtime._network import _MoTLayer
    fn=refresh_function(_MoTLayer.gen_compute_with_kv,cached_joint_attention,5,2)
    assert fn.__code__ is _MoTLayer.gen_compute_with_kv.__code__
    assert _MoTLayer.gen_compute_with_kv.__globals__['cached_joint_attention'] is cached_joint_attention
    routed=fn.__globals__['cached_joint_attention']
    pad=torch.ones(history,dtype=torch.bool);pad[-3:]=False
    for mask,valid in ((within,pad),(within,None),(None,None)):
        got=routed(q,k,v,history,mask,cached_mask=valid,enable_gqa=True)
        ref=cached_joint_attention(q,k,v,history,mask,cached_mask=valid,enable_gqa=True)
        torch.testing.assert_close(got,ref,atol=0,rtol=0)
    print('Temporal mask equivalence and scoped callable checks passed.')


if __name__ == '__main__':
    check_cpu()
