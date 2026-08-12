#!/bin/bash
# Memory sweep: find the real (accum, max-seq) ceiling before any long
# run. Starts from the only configuration that was ever memory-
# validated (-accum 2 -max-seq 512) and steps up, recording peak RSS.
#
# Guarded: an external sampler kills the trainer at HARD_KILL_MB and
# aborts if system free memory drops below FREE_FLOOR_MB, so an
# over-budget config fails as a data point instead of a reboot.
set -uo pipefail

RUN_DIR="${RUN_DIR:-$HOME/speech-corpora/memsweep}"
BIN="${BIN:-$RUN_DIR/qwenvoice-train}"
DATA="${DATA:-$HOME/speech-corpora/shards/stageA}"
STEPS="${STEPS:-3}"
HARD_KILL_MB="${HARD_KILL_MB:-12500}"
FREE_PCT_FLOOR="${FREE_PCT_FLOOR:-12}"   # unused; kept for compat. Real guard: kern.memorystatus_vm_pressure_level (1=normal 2=warn 4=critical) — the signal jetsam itself acts on
OUT="$RUN_DIR/sweep_results.tsv"

mkdir -p "$RUN_DIR"
[ -f "$OUT" ] || printf 'accel\tfence\taccum\tmax_seq\tsteps\tpeak_rss_mb\tsec_per_step\toutcome\n' > "$OUT"

if pgrep -f "$(basename "$BIN")" >/dev/null 2>&1; then
    echo "a trainer is already running — refusing to start" >&2; exit 2
fi

run_one() {  # accel fence accum maxseq
    local accel=$1 fence=$2 accum=$3 seq=$4
    local tag="${accel}_fence${fence}_a${accum}_s${seq}"
    local log="$RUN_DIR/$tag.log"
    local ckpt="$RUN_DIR/work_$tag"
    rm -rf "$ckpt"; mkdir -p "$ckpt"

    local extra=()
    [ "$fence" = "off" ] && extra+=(-unsafe-no-fence)

    echo "=== $tag"
    local t0=$(date +%s)
    "$BIN" -data "$DATA" -out "$ckpt" -steps "$STEPS" -resume none \
        -accum "$accum" -max-seq "$seq" -lora-r 16 -lora-alpha 32 \
        -task-ratios listen=0.45,speak=0.45,text=0.10 -seed 42 \
        -save-every 0 -warmup 100 -accel "$accel" \
        -rss-limit-mb 0 ${extra[@]+"${extra[@]}"} > "$log" 2>&1 &
    local pid=$!

    local peak=0 outcome=completed
    while kill -0 "$pid" 2>/dev/null; do
        local rss free
        rss=$(( $(ps -o rss= -p "$pid" 2>/dev/null | tr -d ' ' || echo 0) / 1024 ))
        [ "$rss" -gt "$peak" ] && peak=$rss
        plevel=$(sysctl -n kern.memorystatus_vm_pressure_level 2>/dev/null)
        if [ "$rss" -gt "$HARD_KILL_MB" ]; then
            outcome="KILLED_rss_${rss}"; kill -9 "$pid" 2>/dev/null; break
        fi
        if [ -n "$plevel" ] && [ "$plevel" -ge 4 ]; then
            outcome="KILLED_pressure_${plevel}"; kill -9 "$pid" 2>/dev/null; break
        fi
        sleep 1
    done
    wait "$pid" 2>/dev/null; local rc=$?
    local t1=$(date +%s)

    local done_steps sec_per_step
    done_steps=$(grep -cE '^step +[0-9]+' "$log" 2>/dev/null || echo 0)
    if [ "$outcome" = "completed" ] && [ "$rc" -ne 0 ]; then outcome="exit_rc_${rc}"; fi
    if [ "${done_steps:-0}" -gt 0 ]; then
        sec_per_step=$(( (t1 - t0) / done_steps ))
    else
        sec_per_step=0
    fi
    printf '%s\t%s\t%d\t%d\t%s\t%d\t%d\t%s\n' \
        "$accel" "$fence" "$accum" "$seq" "${done_steps:-0}" "$peak" "$sec_per_step" "$outcome" >> "$OUT"
    echo "    peak ${peak} MB, ${done_steps:-0} steps, ${sec_per_step}s/step, $outcome"
    rm -rf "$ckpt"
    sleep 15   # let the machine settle between configs
}

for spec in "$@"; do
    IFS=: read -r accel fence accum seq <<< "$spec"
    run_one "$accel" "$fence" "$accum" "$seq"
done
echo; column -t -s $'\t' "$OUT"
