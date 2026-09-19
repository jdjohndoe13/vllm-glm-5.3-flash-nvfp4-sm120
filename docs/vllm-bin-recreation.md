# How `vllm-bin` was created — recreating the bare-metal runtime tree on a new machine

`vllm-bin/` is the runtime tree used by the **no-docker launchers**
(`vllm-glm-5.3-flash-nvfp4-no-docker.sh`,
`vllm-glm-5.3-flash-nvfp4-mtp-no-docker.sh`). It is a byte-identical
extraction of the pinned Docker image's vLLM installation, overlaid with the
kit's `patched-files/`. The **docker launchers do NOT need it** — they
run the image directly and bind-mount `patched-files/` per file (README §4).

Provenance (recorded in `vllm-bin/EXTRACTION_INFO.txt`):

| fact | value |
|---|---|
| extraction date | 2026-09-15T07:27:03+00:00 |
| source container | `vllm-glm-5.3-flash-nvfp4` |
| source image | `cstechdev/vllm:glm53-flash-nope-sm120-cu130-20260826-r1` |
| source image pin | `sha256:0bd709e80b8ff13ae5de8f7d7f708a499fade3a26970d56afb1be2ff3860fde5` |
| vllm version | `0.1.dev20051+g487ecf187` (fork PR vllm-project/vllm#54743) |
| container python | 3.12.3 (must match an existing HOST `/usr/bin/python3` for the thin venv) |
| layout | `dist-packages/` (pip site dir, ~16 GB, full cu13 wheel stack: torch 2.13.0+cu130, cublas 13.1.1.3, cudnn 9.20.0.48, nccl 2.30.7 …), `bin/vllm` (console script), `venv/` (thin wrapper) |
| patched overlay | the files in `patched-files/manifest.txt` overlaid at extraction time, byte-identical to the docker launcher's per-file bind mounts |

## Step 0 — prerequisites

Same as README §2 for the docker path (driver, docker, nvidia-container-toolkit
only if you plan to run the docker launchers too). For the bare-metal path you
need:

- NVIDIA driver 590.x class device nodes on the host (the engine runs outside
  docker; no container toolkit needed).
- HOST python **3.12.x** (`/usr/bin/python3`) — the thin venv wraps the host
  interpreter and its site-packages are a symlink into the extracted
  `dist-packages/`, which contains **cp312** wheels. A different host python
  minor version will not work.

## Path A — extract from the pinned image (docker present)

1. Get the image (either way is equivalent; the launchers verify the hash):
   ```bash
   docker pull cstechdev/vllm@sha256:0bd709e80b8ff13ae5de8f7d7f708a499fade3a26970d56afb1be2ff3860fde5
   # or restore the 29 GB backup tarball made by backup-image.sh:
   #   backup-image.sh OUT.tar   (on the old machine)
   #   docker load -i OUT.tar    (on the new machine)
   docker image inspect --format '{{.Id}}' cstechdev/vllm:glm53-flash-nope-sm120-cu130-20260826-r1
   # must print the pin above
   ```
2. Create a container **without needing GPUs or running it** (docker cp reads
   the filesystem), then copy the two elements out read-only:
   ```bash
   KIT=/mnt/data/shared/models/vllm-glm-5.3-flash-nvfp4
   docker create --name vllm-extract-src \
     cstechdev/vllm:glm53-flash-nope-sm120-cu130-20260826-r1
   docker cp vllm-extract-src:/usr/local/lib/python3.12/dist-packages "$KIT/vllm-bin/"
   mkdir -p "$KIT/vllm-bin/bin"
   docker cp vllm-extract-src:/usr/local/bin/vllm "$KIT/vllm-bin/bin/vllm"
   docker rm vllm-extract-src
   ```
   (The original extraction was `docker cp` from the **running**
   `vllm-glm-5.3-flash-nvfp4` container — same result; a stopped container is
   cleaner and never touches the live engine.)
3. Build the thin venv and finish the layout:
   ```bash
   python3 -m venv --without-pip "$KIT/vllm-bin/venv"
   rm -rf "$KIT/vllm-bin/venv/lib/python3.12/site-packages"
   ln -s "$KIT/vllm-bin/dist-packages" \
         "$KIT/vllm-bin/venv/lib/python3.12/site-packages"
   # rewrite the console-script shebang to the venv python:
   sed -i "1s|.*|#!$KIT/vllm-bin/venv/bin/python|" "$KIT/vllm-bin/bin/vllm"
   chmod +x "$KIT/vllm-bin/bin/vllm"
   ```

## Path B — untar the fast-path tarball (docker NOT required)

The repository ships `vllm-bin-20260915.tar.xz` (4.6 GB; xz of the ~16 GB
tree extracted on 2026-09-15):

```bash
KIT=/mnt/data/shared/models/vllm-glm-5.3-flash-nvfp4
mkdir -p "$KIT" && tar -xJf vllm-bin-20260915.tar.xz -C "$KIT/"
ls "$KIT/vllm-bin"        # -> bin dist-packages venv EXTRACTION_INFO.txt
```

The tarball is dated **2026-09-15**, so it predates newer `patched-files`
revisions (2026-09-16 chunked `cudaHostRegister` + MoE shared-experts fix;
2026-09-18 stale-hit livelock fixes; 2026-09-19 A7 KVC-PIN). The overlay step
below is therefore **mandatory**, on BOTH paths.

## Patched-files overlay (always, on both paths)

`patched-files/manifest.txt` lists paths relative to the vllm package:

```bash
cd <repo>/patched-files
KIT=/mnt/data/shared/models/vllm-glm-5.3-flash-nvfp4   # same as above
while IFS= read -r rel; do
  [ -n "$rel" ] || continue
  cp "$rel" "$KIT/vllm-bin/dist-packages/vllm/$rel"
done < manifest.txt
```

The no-docker launchers re-verify every boot: each manifest file is
`cmp`-checked against the extracted package and any drift refuses startup
loudly (they also refuse to start when the tree is missing `venv/bin/python`,
`bin/vllm`, or the package `__init__.py` — the error message then prints the
complete extraction command list as a reminder, see the launcher's runtime-tree
guard).

## Verify

1. Spot-check the scheduler patch digest (2026-09-19 A7 KVC-PIN state):
   ```bash
   md5sum "$KIT/vllm-bin/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py"
   # expect 0ab8601e095a5e402e4a5c06519d1caa (matches patched-files/ and the last deploy)
   ```
2. Start the launcher (`MAX_MODEL_LEN`/`CPU_TIER_GB` etc. as needed; the
   parity gate runs on every start):
   ```bash
   bash vllm-glm-5.3-flash-nvfp4-mtp-no-docker.sh        # or -no-docker.sh
   bash vllm-glm-5.3-flash-nvfp4-mtp-no-docker.sh stop   # when done
   ```
3. Full needle battery: `bash test.sh` — expect `KV OFFLOADING WORKS`
   (T5 repeat-after-eviction ≥ ~10× faster than T1 cold).

## Notes

- `venv/` is deliberately thin and pip-less; installing things goes through
  the host python into the symlinked dir only if you actually need it — the
  engine has never needed anything beyond the wheels shipped inside the image.
- `EXTRACTION_INFO.txt` records only PROVENANCE (dates, pins, layout); the
  launcher hard-guards (tree completeness + manifest parity at start) are what
  actually prevent drift.
- For a fully docker-first machine you may skip `vllm-bin` entirely: the
  docker launchers (`vllm-glm-5.3-flash-nvfp4.sh`, `-mtp.sh`) consume the
  image + `patched-files/` directly and never touch this tree.
