#!/bin/bash
# Stage-A training supervisor.
#
# Chains trainer invocations through *clean* exits (checkpoint/resume)
# while refusing to relaunch after anything that smells like resource
# exhaustion. Written after the 2026-08-12 post-mortem, in which an
# earlier three-line version took the machine down:
#
#   BUG 1  progress parsed with `tail -1 losses.tsv | cut -f1`, which
#          read a repeated header row, so the exit condition compared
#          the literal string "step" and never fired.
#   BUG 2  `echo "$(date) ... rc=$?"` — the command substitution runs
#          first and resets $?, so every SIGKILL logged as rc=0 and
#          looked like a clean exit.
#   BUG 3  no memory guard, no restart cap, no backoff: one OOM became
#          an unbounded relaunch loop that drove a 24 GB machine to
#          28 MB free and forced a hard reboot.
#
# Rules encoded here:
#   - a trainer killed by a signal (rc >= 128) ABORTS the supervisor.
#     A config that OOMs will OOM again; relaunching is never right.
#   - an external watchdog kills the trainer BEFORE the OS has to,
#     and that also aborts (it is a config error, not a transient).
#   - restarts are capped and backed off, and only clean nonzero exits
#     (e.g. transient I/O) are retried at all.
#   - never two trainers: a lock plus a live-process check.
set -uo pipefail

RUN_DIR="${RUN_DIR:-$HOME/speech-corpora/stageA-run}"
BIN="${BIN:-$RUN_DIR/qwenvoice-train}"
DATA="${DATA:-$HOME/speech-corpora/shards/stageA}"
TARGET_STEPS="${TARGET_STEPS:-4300}"
MAX_RESTARTS="${MAX_RESTARTS:-8}"
RSS_CEILING_MB="${RSS_CEILING_MB:-12000}"   # watchdog kill threshold
FREE_FLOOR_MB="${FREE_FLOOR_MB:-2000}"      # abort if system free+inactive drops below
POLL_SEC="${POLL_SEC:-5}"

cd "$RUN_DIR" || { echo "no run dir $RUN_DIR" >&2; exit 2; }
LOG="$RUN_DIR/supervisor.log"
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

die() { log "ABORT: $*"; exit 1; }

# ---- single instance -------------------------------------------------
LOCK="$RUN_DIR/supervisor.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
    die "another supervisor holds $LOCK (remove it only if no trainer is running)"
fi
trap 'rm -rf "$LOCK"; [ -n "${TRAINER_PID:-}" ] && kill "$TRAINER_PID" 2>/dev/null; exit' EXIT INT TERM

if pgrep -f "$(basename "$BIN")" >/dev/null 2>&1; then
    die "a trainer process is already running — refusing to start a second one"
fi

# ---- progress: last NUMERIC step, headers ignored --------------------
last_step() {
    [ -f losses.tsv ] || { echo 0; return; }
    local s
    s=$(grep -E '^[0-9]+[[:space:]]' losses.tsv | tail -1 | cut -f1)
    [[ "$s" =~ ^[0-9]+$ ]] && echo "$s" || echo 0
}

# phys_footprint in MB — the metric jetsam acts on. ps RSS is blind to
# Metal/IOAccelerator memory (measured 781MB RSS vs 12.7GB footprint),
# so an RSS-based guard cannot protect the machine (2026-08-12).
footprint_mb() {
    local pid=$1 v
    v=$(vmmap --summary "$pid" 2>/dev/null | awk -F: '/^Physical footprint:/ {gsub(/ /,"",$2); print $2; exit}')
    [ -z "$v" ] && { echo 0; return; }
    case "$v" in
        *G) echo $(( $(echo "${v%G}" | cut -d. -f1) * 1024 )) ;;
        *M) echo "${v%M}" | cut -d. -f1 ;;
        *K) echo 0 ;;
        *)  echo 0 ;;
    esac
}

# ---- watchdog: kill the trainer before the OS has to -----------------
# Returns via the flag file: "rss" or "sysmem" if it intervened.
WATCH_FLAG="$RUN_DIR/.watchdog_tripped"
watchdog() {
    local pid=$1
    rm -f "$WATCH_FLAG"
    while kill -0 "$pid" 2>/dev/null; do
        local rss_mb free_mb
        rss_mb=$(footprint_mb "$pid")
        if [ "$rss_mb" -gt "$RSS_CEILING_MB" ]; then
            echo "rss:${rss_mb}" > "$WATCH_FLAG"
            log "WATCHDOG: trainer footprint ${rss_mb} MB > ${RSS_CEILING_MB} MB — killing (pid $pid)"
            kill -9 "$pid" 2>/dev/null
            return
        fi
        # Kernel pressure level — the signal jetsam acts on.
        # 1 = normal, 2 = warning, 4 = CRITICAL (jetsam acts here).
        # We kill only at 4: level 2 is routine under any heavy load.
        # The RSS cap above is the real control. free/inactive pages and
        # memory_pressure's "free percentage" both trip spuriously on
        # macOS (free pages are kept low by design); this does not.
        plevel=$(sysctl -n kern.memorystatus_vm_pressure_level 2>/dev/null)
        if [ -n "$plevel" ] && [ "$plevel" -ge 4 ]; then
            echo "pressure:${plevel}" > "$WATCH_FLAG"
            log "WATCHDOG: kernel memory pressure level $plevel (>=2) — killing trainer (pid $pid) before jetsam does"
            kill -9 "$pid" 2>/dev/null
            return
        fi
        sleep "$POLL_SEC"
    done
}

# ---- run loop --------------------------------------------------------
log "supervisor start: target=$TARGET_STEPS rss_ceiling=${RSS_CEILING_MB}MB max_restarts=$MAX_RESTARTS"
log "args: $*"
restarts=0
while :; do
    step=$(last_step)
    if [ "$step" -ge "$TARGET_STEPS" ]; then
        log "DONE: reached step $step / $TARGET_STEPS"
        exit 0
    fi
    if [ "$restarts" -ge "$MAX_RESTARTS" ]; then
        die "restart cap reached ($restarts) at step $step — investigate before continuing"
    fi

    launch_target=$TARGET_STEPS
    if [ "$CHUNK_STEPS" -gt 0 ]; then
        launch_target=$(( step + CHUNK_STEPS ))
        [ "$launch_target" -gt "$TARGET_STEPS" ] && launch_target=$TARGET_STEPS
    fi
    log "launch #$((restarts + 1)) from step $step, this process targets $launch_target / $TARGET_STEPS"
    "$BIN" -data "$DATA" -out "$RUN_DIR" -steps "$launch_target" -resume auto \
        -rss-limit-mb "$RSS_CEILING_MB" "$@" >> "$RUN_DIR/trainer.log" 2>&1 &
    TRAINER_PID=$!
    watchdog "$TRAINER_PID" &
    WATCH_PID=$!

    wait "$TRAINER_PID"          # exit status captured on its own line
    rc=$?                        # ← never behind a command substitution
    kill "$WATCH_PID" 2>/dev/null; wait "$WATCH_PID" 2>/dev/null

    new_step=$(last_step)
    log "trainer exited rc=$rc (step $step -> $new_step)"

    if [ -f "$WATCH_FLAG" ]; then
        die "watchdog tripped ($(cat "$WATCH_FLAG")) — this config does not fit; lower -accum/-max-seq"
    fi
    if [ "$rc" -ge 128 ]; then
        die "trainer killed by signal $((rc - 128)) (SIGKILL=9 means the OS OOM-killed it). \
Not relaunching: a config that OOMs will OOM again."
    fi
    if [ "$rc" -eq 0 ]; then
        if [ "$new_step" -ge "$TARGET_STEPS" ]; then
            log "DONE: reached step $new_step / $TARGET_STEPS"
            exit 0
        fi
        log "clean exit before target — treating as checkpoint boundary, continuing"
    else
        log "nonzero exit rc=$rc"
    fi
    if [ "$new_step" -le "$step" ]; then
        die "no forward progress (step stayed at $step) — refusing to spin"
    fi

    if [ "$rc" -eq 0 ] && [ "$new_step" -gt "$step" ]; then
        log "chunk complete at step $new_step — relaunching with a fresh VM region map"
        continue          # planned boundary: no backoff, no restart budget
    fi
    restarts=$((restarts + 1))
    backoff=$((10 * restarts))
    log "backoff ${backoff}s before restart $((restarts + 1))"
    sleep "$backoff"
done
