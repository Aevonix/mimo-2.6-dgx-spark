"""Bounded mixed-prefill DiffKV witness; run only on an idle SM121 GPU.

Arguments: mixed candidate path, existing C1 candidate path.
No model weights, KV writes, scheduler, inference endpoint, or runtime mutation.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import torch
from vllm.v1.attention.ops import triton_attention_helpers as helpers

def load(name,path,expected):
    path=Path(path)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==expected
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    return module

assert hashlib.sha256(Path(helpers.__file__).read_bytes()).hexdigest()=='35aa993d01345f535d2641cf9dcabb8b64c51b20a19707bc5987e616aaa87ddd'
candidate=load('_mixed_short_candidate',sys.argv[1],'d3bcd747fe99720bf2b28feccd6fed3491be083088ad7ff83dc4bfcebccb7d67')
baseline=load('_c1_candidate',sys.argv[2],'89777ec11eeb5a3e413718f2553e0424247a9a6ec767e14437d6afaf41f5c2bb')
assert torch.cuda.get_device_capability()==(12,1)
assert not candidate.is_batch_invariant
torch.manual_seed(20260922)
torch.backends.cuda.matmul.allow_tf32=False
torch.set_float32_matmul_precision('highest')
BS,H,K,V,SEG=16,16,192,128,16

def emit(obj):print(json.dumps(obj,allow_nan=False),flush=True)

def compare(a,b):
    a,b=a.float(),b.float()
    finite=bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all())
    return dict(finite=finite,byte_equal=bool(torch.equal(a,b)),
        relative_l2=float((a-b).square().mean().sqrt()/b.square().mean().sqrt().clamp_min(1e-8)) if finite else None,
        max_abs=float((a-b).abs().max()) if finite else None)

def scratch(cap):
    return [torch.full(shape,float('nan'),device='cuda',dtype=torch.float32)
        for shape in ((cap,H,SEG,V),(cap,H,SEG),(cap,H,SEG))]

def make_case(lengths,window):
    # Short sequences end immediately after the first token of logical page100.
    key_lengths=[1601 if n<=8 else n for n in lengths]
    page_counts=[(n+BS-1)//BS for n in key_lengths]
    total_pages=sum(page_counts)
    cache=torch.full((total_pages+1,1,BS,K+V),float('nan'),device='cuda',dtype=torch.bfloat16).transpose(1,2)
    physical=torch.randperm(total_pages,device='cuda')
    table=torch.full((len(lengths),max(page_counts)+2),total_pages,device='cuda',dtype=torch.int32)
    logical=[];off=0
    for seq,(n,pages) in enumerate(zip(key_lengths,page_counts)):
        values=torch.randn((n,K+V),device='cuda',dtype=torch.bfloat16)
        logical.append(values)
        mapping=physical[off:off+pages];off+=pages
        table[seq,:pages]=mapping.to(torch.int32)
        for page in range(pages):
            size=min(BS,n-page*BS)
            cache[mapping[page],:size,0]=values[page*BS:page*BS+size]
    starts=[0]
    for n in lengths:starts.append(starts[-1]+n)
    # Explicit padded Q/output rows must not become compact scratch indexes.
    q=torch.full((sum(lengths)+3,H,K),float('nan'),device='cuda',dtype=torch.bfloat16)
    q[:sum(lengths)]=torch.randn((sum(lengths),H,K),device='cuda',dtype=torch.bfloat16)
    for seq,n in enumerate(lengths):
        if n<=8:q[starts[seq+1]-1]=logical[seq][-1,:K].unsqueeze(0).expand(H,K)
    sinks=torch.randn(H,device='cuda',dtype=torch.float32) if window[0]>=0 else None
    args=dict(q=q,k=cache[...,:K],v=cache[...,K:],
        cu_seqlens_q=torch.tensor(starts,device='cuda',dtype=torch.int32),
        seqused_k=torch.tensor(key_lengths,device='cuda',dtype=torch.int32),
        max_seqlen_q=max(lengths),softmax_scale=K**-0.5,causal=True,
        softcap=0.0,window_size=window,block_table=table,sinks=sinks)
    return args,starts,key_lengths,logical

def reference(q,logical,key_length,qlen,window,sinks):
    positions=torch.arange(key_length-qlen,key_length,device='cuda')
    keys=torch.arange(key_length,device='cuda')
    mask=keys[None,:]<=positions[:,None]
    if window[0]>=0:mask&=positions[:,None]-keys[None,:]<=window[0]
    scores=torch.einsum('qhd,kd->hqk',q.float(),logical[:,:K].float())*K**-0.5
    scores.masked_fill_(~mask[None],float('-inf'))
    if sinks is not None:scores=torch.cat((scores,sinks[:,None,None].expand(H,qlen,1)),dim=-1)
    return torch.einsum('hqk,kd->qhd',scores.softmax(-1)[...,:key_length],logical[:,K:].float())

def execute(module,args,out,buffers,split=True):
    out.fill_(float('nan'))
    for b in buffers:b.fill_(float('nan'))
    extra={}
    if split:extra=dict(seq_threshold_3D=128,num_par_softmax_segments=SEG,
        softmax_segm_output=buffers[0],softmax_segm_max=buffers[1],softmax_segm_expsum=buffers[2])
    module.unified_attention_diffkv(**args,out=out,**extra)

def isolated(args,seq,q,key_length):
    local=dict(args,q=q,cu_seqlens_q=torch.tensor([0,q.shape[0]],device='cuda',dtype=torch.int32),
        seqused_k=torch.tensor([key_length],device='cuda',dtype=torch.int32),
        block_table=args['block_table'][seq:seq+1],max_seqlen_q=q.shape[0])
    out=torch.empty((q.shape[0],H,V),device='cuda',dtype=torch.bfloat16)
    execute(baseline,local,out,scratch(8))
    return out

cases=[((small,512),window) for small in (1,8) for window in ((-1,-1),(127,0))]
cases += [((512,small),window) for small in (1,8) for window in ((-1,-1),(127,0))]
cases += [((512,8,1,512),(-1,-1)),((8,512,512,1),(127,0))]
all_ok=True
for case,(lengths,window) in enumerate(cases):
    args,starts,key_lengths,logical=make_case(lengths,window)
    out=torch.empty((args['q'].shape[0],H,V),device='cuda',dtype=torch.bfloat16)
    buffers=scratch(len(lengths)*8)
    execute(candidate,args,out,buffers)
    original=out.clone()
    checks=[]
    old_out=torch.empty_like(out)
    execute(baseline,args,old_out,scratch(128))
    unused=[seq*8+i for seq,n in enumerate(lengths) for i in range(8) if n>8 or i>=n]
    unused_untouched=all(bool(torch.isnan(b[unused]).all()) for b in buffers)
    padded_untouched=bool(torch.isnan(out[sum(lengths):]).all())
    for seq,n in enumerate(lengths):
        row=out[starts[seq]:starts[seq+1]]
        ref=reference(args['q'][starts[seq]:starts[seq+1]],logical[seq],key_lengths[seq],n,window,args['sinks'])
        result={'seq':seq,'queries':n,'reference':compare(row,ref)}
        if n<=8:
            separate=torch.cat([isolated(args,seq,args['q'][starts[seq]+i:starts[seq]+i+1],key_lengths[seq]-n+i+1) for i in range(n)])
            result['q1']=compare(row,separate)
            result['previous_mixed_path']=compare(old_out[starts[seq]:starts[seq+1]],separate)
        else:
            result['long_unchanged']=compare(row,old_out[starts[seq]:starts[seq+1]])
        checks.append(result)
    # Warm and capture unchanged launch shapes. Replay twice with poisoned
    # scratch each time, then change actual Q values and compare fresh eager.
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):execute(candidate,args,out,buffers)
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):execute(candidate,args,out,buffers)
    graph.replay();graph.replay()
    replay=compare(out[:sum(lengths)],original[:sum(lengths)])
    args['q'][:sum(lengths)].add_(0.125)
    graph.replay();changed_graph=out[:sum(lengths)].clone()
    execute(candidate,args,out,buffers)
    changed=compare(changed_graph,out[:sum(lengths)])
    ok=unused_untouched and padded_untouched and replay['byte_equal'] and changed['byte_equal']
    for result in checks:
        ok &= result['reference']['finite'] and result['reference']['relative_l2']<0.02
        if 'q1' in result:ok &= result['q1']['byte_equal']
        else:ok &= result['long_unchanged']['byte_equal']
    all_ok &= bool(ok)
    emit(dict(event='case',case=case,lengths=lengths,window=window,key_lengths=key_lengths,
        compact_slots=len(lengths)*8,total_query_rows=sum(lengths),checks=checks,
        unused_scratch_untouched=unused_untouched,padded_output_untouched=padded_untouched,
        graph_replay=replay,graph_changed_input=changed,passed=bool(ok)))
emit(dict(event='complete',passed=all_ok,cases=len(cases),scope='Synthetic mixed-prefill attention only; no scheduler, KV writes, or model correctness claim.'))
raise SystemExit(0 if all_ok else 1)
