# Running on CHTC (HTCondor)

Access point: `ap2002` (no GPU). GPU work runs as HTCondor jobs inside an Apptainer container.
Large files live in `/staging/s/smittal39` (100 GB, **max 1,000 files** — bundle outputs into tarballs).

## One-time setup

### 1. Copy the cluster folder from your Mac

On your **Mac**, from the repo root:

```bash
./cluster/pack_code.sh                                   # bundles src/ + TD-MPC2 into cluster/code.tar.gz
scp -r cluster smittal39@ap2002.chtc.wisc.edu:~/wm-cluster
```

### 2. Build the container (≈20–40 min, once)

On **ap2002**:

```bash
cd ~/wm-cluster
condor_submit -i build.sub           # waits for a build node, then gives you a shell there
# --- inside the build job ---
apptainer build tdmpc2.sif tdmpc2.def
mv tdmpc2.sif /staging/s/smittal39/
exit
```

### 3. GPU smoke test

On **ap2002**:

```bash
cd ~/wm-cluster
condor_submit smoke.sub
condor_q                              # watch status (I = idle/waiting, R = running)
# when it finishes:
cat smoke.out smoke.err
```

`smoke.out` reports: GPU model, CUDA, TD-MPC2 imports, and for each ManiSkill3 task the observation/action
sizes, **episode length** (which sets TD-MPC2's discount), simulator speed, and the info keys.
`pip-freeze.txt` records exact package versions in the container.

`PickSingleYCB-v1` is expected to fail in the first smoke test until the YCB object assets are downloaded
(a later step stores them in staging).

## Useful commands

| Command | Purpose |
| :--- | :--- |
| `condor_q` | Your jobs and their status |
| `condor_q -better-analyze <job id>` | Why a job is still idle |
| `condor_rm <job id>` | Cancel a job |
| `get_quotas $USER` | Storage usage |
