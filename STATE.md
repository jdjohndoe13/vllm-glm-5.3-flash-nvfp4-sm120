# STATE.md — LLM-server ops workspace

Repo = orchestration home only (not the engine code). Ops brief: `prompt.md` (§5–§12 contract; deliberately UNTRACKED). Read §9 before dispatching any relay cycle.

## Live snapshot (2026-09-22)
- **Engine**: GLM-5.3-Flash-NVFP4 TP=8, 8× RTX 5090, Supermicro X14DBI, native launcher (NO docker), port 1025, model aliases `glm-5.3-flash` + `qwen-3.8-flash-next`. Client 192.168.2.208 (iterative GGP make-context with ~106k token prefills, ~0.7/s).
- **Boot 11:22:32**: clean GPUs (0 Xid), docker fully healthy, D-class clean. Engine relaunched natively by user (pid 11468). Bootlog `engine-20260922-112818.log` (via `latest.log` symlink in `/mnt/data/shared/models/vllm-glm-5.3-flash-nvfp4/logs/`).
- **ESC211** (KV-LOOKUP wedge escape, scheduler+connector patches) live from this boot; committed `1ae9d98`. Validated: 97 fallbacks / 18.9h / 0 kit-caused deaths / 1 designed SERVERKILL. All knobs kept: FALLBACK_LIMIT=3, budget=8, TTL=150s, PIN=10s, WINDOW=300s.
- **Canary v3.2** (deployed 2026-09-22 13:50; the v3.1 station died silently at 13:31 — `-u` removed and an EXIT trap added, so any future death leaves an `ALERT: CANARY-EXIT(rc=N)` forensics row instead of a blind gap): persistent dir `/mnt/data/shared/models/vllm-glm-5.3-flash-nvfp4/logs/canary/` = `{canary5m.sh, hb.json, canary5m.log, canary.state, station.pid}`. md5 script `427299f5d9137c22425fb1fc85b58d8e` (= `tools/canary5m.sh` mirror), hb payload md5 `ab306c5503e927000e52f883eb13b209` (= `tools/canary_hb.json`).

## Incident ref (crash 09-22 ~09:47)
Root cause = **GPU driver-level lock** (`NVRM RC watchdog: GPU is probably locked!`, GPU3; preceded by worker shm_broadcast stalls + `sample_tokens` RPC timeout → EngineDeadError). D-state workers killed the hard way (reboot hang → power-off). No kit knob changed (advisory-validated). NOT addressed in code; detection added instead (canary v3.1: `shm=`, `gpu=`, `hb=`). Full record: `patches/README.md` 09-22 section (commit `b200109`).
Known gaps: pstore unusable (erst write error −22); driver-level lock itself is NVIDIA/diagnosis territory if it recurs.

## Relay contract (§9, per cycle)
1. Fresh fixer session (`background: true`), description `"Canary relay cycle N (5-min watchdog)"`. Never reuse task_id.
2. Sleep 301s (`C:\Windows\System32\ping.exe -n 301 127.0.0.1 >nul`), then probe verbatim: ssh **pidfile liveness** (`ks=$(cat station.pid); kill -0 $ks` → `CANARY_ALIVE_$ks`/`CANARY_DEAD`; pgrep NOT used — self-match trap), `tail -14 /mnt/data/.../logs/canary/canary5m.log`, health curl. Timeout 60s on the ssh.
3. Verdict `CYCLE_OK` iff CANARY_ALIVE + health=200 + newest h= row alert-free (incl. no `CANARY-EXIT` row) + ≥1 fb= row; else `CYCLE_ALERT: <problems>`. A `CANARY-EXIT` row or CANARY_DEAD → orchestrator immediately dispatches a canary-restart lane (redeploy from mirror if md5 differs, restart, row-1 verify) before resuming the relay; the trap rc feeds the post-mortem.
4. Decode (v3.1 row schema: `[HH:MM:SS] h= p= 500= oom= shm= gpu= hb= fb= kv= rmax= rep= apc= bytes= ev= [ALERT tags]`):
   - Cumulative counters (fb/kv/500/oom/rep/apc) static on tag-free rows = no escalation; fb +≤3/window = informational one-shot strike band.
   - `SHM-STALL(n)`, `GPU-LOCK(d)`, `SERVE-WEDGE` (hb=000 timeout), `CANARY-EXIT(rc=N)` (station silent death forensics; restart lane follows) → CYCLE_ALERT. `NEWBOOTLOG` alone = informational rebase → CYCLE_OK unless h≠200/p≠9.
   - p=9 native baseline; hb expects 200; ev= gauges; cadence ~5 min (gap >10 min → flag).
5. Reconcile each cycle before dispatching the next. Relay runs autonomously; user is pinged only on alerts.

## Git
`master`: `1ae9d98` (ESC211 files), `b200109` (ESC211 production evidence + 09-22 incident record + canary v3.0 script capture), `03ee209` (hb payload), `11381d1` (canary v3.1 script content). `prompt.md` untracked by choice. Identity mirrored per-commit via `git -c user.name=none -c user.email=none@nowhere.com` — never set config.

## Workflow rules (ssh / windows quirk)
- ssh: `C:\Windows\System32\OpenSSH\ssh.exe -o BatchMode=yes user@testcomp2`. Wrap possibly-hanging remote cmds in `timeout 10/12`. Passwordless sudo available.
- **No `$()` inside grep args** (reads stdin, hangs). `awk` broken locally → `cut`/`grep -o`+`tr`. Local md5: `certutil -hashfile <file> MD5`. Remote: `md5sum`.
- **pgrep trap**: never put another clause containing literal `canary5m.sh` in the same ssh command as a `pgrep -f` check (self-matches wrapper → false CANARY_ALIVE).
- Engine restarts are user-owned in the launch window (§7); canary handles rebase autonomously.
