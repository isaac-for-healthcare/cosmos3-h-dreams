# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio, time
import numpy as np


def stamp(frame,number):
    frame=frame.copy()
    values=[(0xa5,8),(number,24),(number^0xffffff,24)]
    bits=[(v>>b)&1 for v,n in values for b in reversed(range(n))]
    frame[:8,:448,:]=np.repeat(np.array(bits,dtype=np.uint8)*255,8)[None,:,None]
    return frame

def read_stamp(frame):
    bits=frame[4,4:448:8,0]>127
    values=[]
    for start,end in [(0,8),(8,32),(32,56)]:
        v=0
        for b in bits[start:end]:v=(v<<1)|int(b)
        values.append(v)
    assert values[0]==0xa5 and values[1]^values[2]==0xffffff
    return values[1]

async def proxy_connection(reader,writer,target_port,delay_ms,mbps):
    upstream_r,upstream_w=await asyncio.open_connection('127.0.0.1',target_port)
    async def pipe(src,dst,rate):
        queue=asyncio.Queue(maxsize=64)
        async def receive():
            i=0
            while data:=await src.read(32768):
                # Deterministic ±10 ms arrival variation; enforce TCP ordering below.
                jitter=([0,5,-5,10,-10][i%5] if delay_ms else 0)/1000
                await queue.put((time.perf_counter()+delay_ms/1000+jitter,data));i+=1
            await queue.put((0,None))
        async def transmit():
            next_send=0
            while True:
                ready,data=await queue.get()
                if data is None:break
                await asyncio.sleep(max(0,max(ready,next_send)-time.perf_counter()))
                dst.write(data);await dst.drain()
                next_send=time.perf_counter()+len(data)*8/(rate*1e6) if rate else 0
        try:await asyncio.gather(receive(),transmit())
        finally:dst.close()
    try:await asyncio.gather(pipe(reader,upstream_w,1 if mbps else 0),pipe(upstream_r,writer,mbps))
    except (ConnectionError,asyncio.CancelledError):pass
    finally:
        writer.close();upstream_w.close()
