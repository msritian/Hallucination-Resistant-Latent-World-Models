#!/bin/bash
# Re-analyse saved planning-diagnostic records (no simulation). Args: <result_name> <run> <task>
NAME=$1; RUN=$2; TASK=$3
echo "host $(hostname)  start $(date)"
tar -xzf code.tar.gz
export PYTHONPATH=$PWD
mkdir -p m d
tar -xzf "$NAME.tar.gz" -C m
tar -xzf "plandiag_$NAME.tar.gz" -C d
python -m src.tools.plan_diag --model m/run/final_model.pt --run "$RUN" --task "$TASK" --out diag \
    --records d/diag/records.pt --taus_file d/diag/taus_onpolicy.json
status=$?
mkdir -p diag
tar -czf diag.tar.gz diag/report.md diag/diag.json diag/taus_candidates.json
echo "done $(date)  exit status $status"
exit $status
