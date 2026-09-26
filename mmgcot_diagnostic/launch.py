"""Launch one deterministic image shard per explicitly selected free GPU."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from mmgcot_diagnostic.protocol import frozen_protocol, file_hash


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--gpus",required=True)
    args=p.parse_args()
    gpus=args.gpus.split(",")
    if len(set(gpus))!=len(gpus):
        p.error("GPU IDs must be unique")
    available=subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used",
                                      "--format=csv,noheader,nounits"],text=True)
    memory={i.strip():int(m.strip()) for i,m in (line.split(",") for line in available.splitlines())}
    if any(memory[g]>=500 for g in gpus):
        raise RuntimeError(f"requested GPU is occupied: {memory}")
    args.output_dir.mkdir(parents=True,exist_ok=False)
    (args.output_dir/"protocol.json").write_text(json.dumps(frozen_protocol(),ensure_ascii=False,indent=2)+"\n")
    snapshots=args.output_dir/"source_snapshot"
    snapshots.mkdir()
    root=Path(__file__).resolve().parents[1]
    dependencies=[*Path(__file__).resolve().parent.glob("*.py"),
        root/"live_kv_probe_prototype/run_hf_fork.py",root/"reasoning_checkpoints/extractor.py"]
    for path in dependencies:
        target=snapshots/path.relative_to(root)
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(path.read_bytes())
    receipt=dict(manifest=str(args.manifest.resolve()),manifest_sha256=file_hash(args.manifest),
                 gpus=gpus,started_at=time.time(),workers=[])
    workers=[]
    for shard,gpu in enumerate(gpus):
        cmd=[sys.executable,"-u","-m","mmgcot_diagnostic.run","--manifest",str(args.manifest.resolve()),
            "--output-dir",str(args.output_dir.resolve()),"--device",gpu,
            "--shard-index",str(shard),"--num-shards",str(len(gpus))]
        log=(args.output_dir/f"shard{shard}.log").open("x")
        env=dict(os.environ,PYTHONNOUSERSITE="1",PYTHONHASHSEED="20260920",CUDA_VISIBLE_DEVICES=gpu,
                 CUBLAS_WORKSPACE_CONFIG=":4096:8")
        process=subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
        workers.append((process,log))
        receipt["workers"].append(dict(shard=shard,gpu=gpu,pid=process.pid,command=cmd))
        print(f"START shard={shard} gpu={gpu} pid={process.pid}",flush=True)
    (args.output_dir/"launch.json").write_text(json.dumps(receipt,indent=2)+"\n")
    running=set(range(len(workers)))
    while running:
        for i in list(running):
            process,log=workers[i]
            code=process.poll()
            if code is not None:
                log.close();running.remove(i)
                receipt["workers"][i]["exit_code"]=code
                print(f"EXIT shard={i} code={code}",flush=True)
                # A systemic worker error invalidates concurrent comparisons.
                # Stop only the child processes owned by this launch.
                if code!=0:
                    for j in running:
                        workers[j][0].terminate()
                    for j in running:
                        receipt["workers"][j]["exit_code"]=workers[j][0].wait()
                        workers[j][1].close()
                    running.clear()
                    break
        if running:
            time.sleep(2)
    receipt["finished_at"]=time.time()
    (args.output_dir/"completion.json").write_text(json.dumps(receipt,indent=2)+"\n")
    if any(w.get("exit_code")!=0 for w in receipt["workers"]):
        raise SystemExit(1)


if __name__=="__main__":
    main()
