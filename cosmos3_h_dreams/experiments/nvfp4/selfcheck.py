# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runnable CPU checks for analysis and the shared input adapter."""
import json,os
from pathlib import Path
from analyze import percentile, stats, browser_summary, stages

assert percentile([4,1,3,2],50)==2.5
assert abs(percentile([4,1,3,2],95)-3.85)<1e-10
assert stats([])['n']==0
client=dict(errors=[],startedMs=0,sent=[dict(seq=1,tick=0,ms=0)],
            observed=[dict(id=1,ms=100),dict(id=2,ms=134)])
server=dict(outputs=[dict(id=1,seq=1),dict(id=2,seq=1)],
            receives=[dict(seq=1)],captures=[dict(seq=1)],publications=[dict(id=1),dict(id=2)])
s=browser_summary(client,server,0,1)
assert s['observed']==s['sent']==1 and s['latency_ms']['p50']==100
assert s['missing_frame_ids']==s['gaps_over100ms']==0
client['observed'].append(dict(id=1,ms=150))
try:browser_summary(client,server,0,1)
except AssertionError:pass
else:raise AssertionError('duplicate/reordered IDs accepted')

checkroot=Path(os.environ.get('DREAMS_SELFCHECK_DIR','/nonexistent'))
if checkroot.exists():
    for mode in ('latest',):
        path=checkroot/mode
        read=lambda name:json.loads((path/name).read_text())
        assert read('runtime.json')['completed'] and read('runtime.json')['fake']
        s=browser_summary(read('client.json'),read('server.json'),0,4)
        assert s['sent']==s['consumed']==s['observed']==4,(mode,s)
print('Analysis self-checks passed; both fake browser paths verified when present.')

client=dict(sent=[dict(seq=1,tick=0,ms=0,keys=['d'])],observed=[dict(id=1,ms=100)])
server=dict(outputs=[dict(id=1,seq=1,capture_s=1.01,prepared_s=1.03,frame_index=0)],receives=[dict(seq=1,time_s=1.0)],publications=[dict(id=1,time_s=1.06)])
r=stages(client,server,0,1)[0]
assert abs(r['server_receive_to_capture_ms']-10)<1e-9
assert abs(r['capture_to_rgb_ms']-20)<1e-9
assert abs(r['rgb_to_publication_ms']-30)<1e-9
assert abs(r['transport_browser_and_receive_overhead_ms']-40)<1e-9
print('Stage accounting self-check passed.')

# Protocol wiring is separate from GPU timing. Every matrix retains the same
# measured control schedule, two repetitions and explicit per-variant pacing.
from matrix import protocol, worker_commands
for name, count in [('latency30', 6), ('rate', 4), ('attention45', 4)]:
    plan = protocol(name, 'test-commit')
    commands = list(worker_commands(plan, Path('/unused'), python='python'))
    assert plan['commits'] == {'latest': 'test-commit'}
    assert len(commands) == count
    assert sum(15 <= t < 70 for t in plan['control_times_s']) == 71
    for variant, command in zip(plan['order'], commands, strict=True):
        case = plan['variants'][variant]
        assert plan['order'].count(variant) == 2
        assert command[command.index('--attention-variant') + 1] == case['attention_variant']
        assert float(command[command.index('--display-fps') + 1]) == case['display_fps']
        assert float(command[command.index('--lead-ms') + 1]) == case['lead_ms']
    assert all(a in plan['variants'] and b in plan['variants'] for a, b in plan['comparisons'])
print('Three matrix protocols and worker arguments verified.')
