# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reconcile matched source/settings, full-work timing and browser lineage."""
import json
import math
import os
import statistics
import sys
from pathlib import Path


def percentile(values, p):
    if not values:return None
    values=sorted(values);at=(len(values)-1)*p/100;lo=math.floor(at);hi=math.ceil(at)
    return values[lo]+(values[hi]-values[lo])*(at-lo)


def stats(values):
    assert all(math.isfinite(v) for v in values)
    return dict(n=len(values),p50=percentile(values,50),p90=percentile(values,90),
                p95=percentile(values,95),p99=percentile(values,99),max=max(values) if values else None)


def browser_summary(client, server, start_s, end_s):
    assert not client['errors']
    sent={r['seq']:r for r in client['sent'] if start_s<=r['tick']<end_s}
    outputs={r['id']:r for r in server['outputs']}
    observed_ids=[r['id'] for r in client['observed']]
    assert all(b>a for a,b in zip(observed_ids,observed_ids[1:])), 'nonmonotonic frame IDs'
    first={}
    for row in client['observed']:first.setdefault(outputs[row['id']]['seq'],row['ms'])
    received={r['seq'] for r in server['receives']} & sent.keys()
    consumed={r['seq'] for r in server['captures']} & sent.keys()
    seen=first.keys() & sent.keys()
    latency=[first[seq]-sent[seq]['ms'] for seq in sorted(seen)]
    assert all(value>=0 for value in latency)
    selected=[r for r in client['observed'] if client['startedMs']+start_s*1000<=r['ms']<client['startedMs']+end_s*1000]
    gaps=[b['ms']-a['ms'] for a,b in zip(selected,selected[1:])]
    missing=sum(b['id']-a['id']-1 for a,b in zip(selected,selected[1:]))
    publications=[r['id'] for r in server['publications']]
    assert publications==list(range(publications[0],publications[-1]+1)), 'publication sequence skipped frames'
    return dict(sent=len(sent),received=len(received),consumed=len(consumed),observed=len(seen),
        unobserved_controls=sorted(sent.keys()-seen),latency_ms=stats(latency),
        frame_gap_ms=stats(gaps),gaps_over100ms=sum(g>100 for g in gaps),missing_frame_ids=missing,
        observed_fps=(len(selected)-1)*1000/(selected[-1]['ms']-selected[0]['ms']),
        raw_latency_ms=latency,raw_gaps_ms=gaps)


def stages(client,server,start,end):
    sent={r['seq']:r for r in client['sent'] if start<=r['tick']<end}
    outputs={r['id']:r for r in server['outputs']}
    receives={}
    for row in server['receives']:receives.setdefault(row['seq'],row['time_s'])
    publications={r['id']:r['time_s'] for r in server['publications']}
    seen={}
    for row in client['observed']:seen.setdefault(outputs[row['id']]['seq'],row)
    records=[]
    for seq in sorted(sent.keys() & seen.keys()):
        obs=seen[seq];output=outputs[obs['id']];receive=receives[seq]
        assert receive<=output['capture_s']<=output['prepared_s']<=publications[obs['id']]
        total=obs['ms']-sent[seq]['ms']
        parts=dict(server_receive_to_capture_ms=(output['capture_s']-receive)*1000,
            capture_to_rgb_ms=(output['prepared_s']-output['capture_s'])*1000,
            rgb_to_publication_ms=(publications[obs['id']]-output['prepared_s'])*1000)
        residual=total-sum(parts.values())
        assert residual>=-1,('inconsistent client/server duration sum',seq,residual)
        records.append(dict(seq=seq,keys=sent[seq]['keys'],frame_id=obs['id'],frame_index=output['frame_index'],
            total_ms=total,**parts,transport_browser_and_receive_overhead_ms=residual))
    return records


def main(root):
    read=lambda p:json.loads(p.read_text())
    protocol=read(root/'protocol.json')
    cases=[]
    for index,variant in enumerate(protocol['order']):
        tag=f'{index:02d}-{variant}';path=root/'browser'/tag
        meta=read(path/'runtime.json');client=read(path/'client.json');server=read(path/'server.json')
        assert meta['completed'] and not meta['fake']
        expected=protocol['variants'][variant]
        assert meta['attention_variant']==expected['attention_variant']
        assert meta['mode']==expected['mode'] and meta['generation_lead_ms']==expected['lead_ms']
        assert bool(meta['config'].get('fp8_weights'))==expected['fp8']
        assert bool(meta['nvfp4_weights'])==expected['nvfp4']
        assert meta['source_commit']==protocol['commits'][expected['mode']]
        assert meta['display_fps']==expected['display_fps']
        assert meta['seconds']==protocol['seconds_per_trial']
        assert meta['keyboard_velocity_scale']==30/expected['display_fps']
        summary=browser_summary(client,server,*protocol['measurement_window_s'])
        expected_count=protocol['measured_controls_per_trial']
        assert summary['sent']==summary['received']==summary['consumed']==summary['observed']==expected_count,(tag,summary)
        records=stages(client,server,*protocol['measurement_window_s'])
        result=dict(case=tag,**meta,**summary,stage_records=records,
            rAF_interval_ms=stats(client['raf']),canvas_read_ms=stats(client['readCost']),ping_roundtrip_ms=stats(client['pingMs']),marker_ms=stats(server['marker_ms']))
        (path/'summary.json').write_text(json.dumps(result,indent=2)+'\n');cases.append(result)
    first=cases[0]
    for r in cases:
        for key in ['gpu','gpu_uuid','driver','python','torch','cuda','cudnn','assets_sha256','checkpoint_sha256','browser','torch_threads']:
            assert r[key]==first[key],(r['case'],key)
        for key in ['seed','initial_frame','actions','prompt','domain_id','height','width','fps','chunk_size','denoise_steps','decoder','encoder','window_frames','cache_mode','fademem_slots','fademem_beta','fademem_merge_selection','compile_denoise','compile_denoise_mode','cuda_graph_denoise']:
            assert r['config'][key]==first['config'][key],(r['case'],key)
    combined={}
    for variant in protocol['variants']:
        selected=[x for x in cases if x['variant']==variant];assert len(selected)==2
        records=[r for x in selected for r in x['stage_records']]
        combined[variant]=dict(source_commit=selected[0]['source_commit'],generation_lead_ms=selected[0]['generation_lead_ms'],
            controls_observed=len(records),latency_ms=stats([r['total_ms'] for r in records]),
            per_trial_latency_ms=[x['latency_ms'] for x in selected],
            server_stage_ms={key:stats([r[key] for r in records]) for key in ['server_receive_to_capture_ms','capture_to_rgb_ms','rgb_to_publication_ms','transport_browser_and_receive_overhead_ms']},
            control_press_ms=stats([r['total_ms'] for r in records if r['keys']]),control_release_ms=stats([r['total_ms'] for r in records if not r['keys']]),
            frame_gap_ms=stats([v for x in selected for v in x['raw_gaps_ms']]),observed_fps=[x['observed_fps'] for x in selected],
            missing_frame_ids=sum(x['missing_frame_ids'] for x in selected),gaps_over100ms=sum(x['gaps_over100ms'] for x in selected),
            canvas_read_median_ms=[x['canvas_read_ms']['p50'] for x in selected],marker_median_ms=[x['marker_ms']['p50'] for x in selected],ping_roundtrip_median_ms=[x['ping_roundtrip_ms']['p50'] for x in selected])
    comparisons=[]
    for a,b in protocol['comparisons']:
        base,new=combined[a]['latency_ms'],combined[b]['latency_ms']
        comparisons.append(dict(baseline=a,candidate=b,median_reduction_ms=base['p50']-new['p50'],median_reduction_pct=100*(1-new['p50']/base['p50']),p95_reduction_ms=base['p95']-new['p95']))
    summary=dict(completed=True,protocol=protocol,combined=combined,comparisons=comparisons,all_controls_received_consumed_observed=True,
        hardware=first['gpu'],physical_display_measured=False,semantic_motion_onset_measured=False,remote_network_measured=False,
        endpoint='Browser control-state WebSocket send to first corresponding decoded MJPEG image observed at browser requestAnimationFrame; common adapter, explicit per-variant presentation target, and wall-time keyboard velocity normalization.',
        notes=['Headless Chromium on Colossus loopback, GPU rendering disabled.','Frame IDs tag the chunk conditioned on each control; they do not establish the onset of visible tool motion.','Same wall-time input schedule; generated action trajectories may differ because capture timing differs.','Eight-frame chunks, two denoising steps, one seed/scene.', 'When presentation rates differ, action sampling and rollout depth per wall-time second also differ; this is not a quality-equivalence test.', 'Headless Chromium rAF is near 60Hz; 45 FPS pacing does not establish smoothness on a physical display.','BF16 source immutable; explicit separate FP8/NVFP4 overlays; instrumentation outside the product.'])
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))

if __name__=='__main__':main(Path(sys.argv[1]))
