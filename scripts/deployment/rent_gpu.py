import json, sys, subprocess

res = subprocess.run(
    ["vastai", "search", "offers",
     "gpu_name=RTX_4090 num_gpus=1 rentable=True verified=true cpu_ram>=64 disk_space>=100 dph_total<=0.5",
     "--on-demand", "--raw"],
    capture_output=True, text=True
)

data = json.loads(res.stdout)
print(f"Found {len(data)} GPUs available:")
for d in data[:5]:
    print(f"  ID:{d['id']}  ${d['dph_total']:.3f}/hr  {d['gpu_name']}  RAM:{d.get('gpu_ram',0)//1024}GB")

if data:
    best = data[0]
    print(f"\nBest: ID {best['id']} at ${best['dph_total']:.3f}/hr")
    print(f"Renting...")
    
    res2 = subprocess.run(
        ["vastai", "create", "instance", str(best["id"]),
         "--image", "pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime",
         "--disk", "100"],
        capture_output=True, text=True
    )
    print(res2.stdout)
    print(res2.stderr if res2.stderr else "")
