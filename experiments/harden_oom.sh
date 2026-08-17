#!/bin/bash
# OOM-proofing for the Spark (run as root). Goal: the box must never again
# thrash itself off the network — reachability daemons become untouchable
# to the kernel OOM killer, experiments become its PREFERRED victims, and a
# boot-persistent guard kills the largest experiment process before the
# kernel has to act.
set -u
LOGF=/var/log/dm-harden.log
echo "== harden $(date -Is)" >> "$LOGF"

# --- 1. live oom_score_adj for critical daemons (no restarts needed) ---
protect_now() {  # name, score
  for p in $(pgrep -x "$1" 2>/dev/null); do
    echo "$2" > "/proc/$p/oom_score_adj" 2>/dev/null && \
      echo "  live: $1 pid=$p -> $2" >> "$LOGF"
  done
}
protect_now sshd -1000
protect_now tailscaled -1000
protect_now dockerd -900
protect_now containerd -900
protect_now nginx -800
protect_now cron -800
protect_now systemd-logind -900

# --- 2. persistent: systemd drop-ins so protection survives restarts/boots ---
declare -A SCORES=([ssh]=-1000 [sshd]=-1000 [tailscaled]=-1000 [docker]=-900 \
                   [containerd]=-900 [nginx]=-800 [cron]=-800)
for svc in "${!SCORES[@]}"; do
  if systemctl list-unit-files "$svc.service" --no-legend 2>/dev/null | grep -q "$svc.service"; then
    mkdir -p "/etc/systemd/system/$svc.service.d"
    printf '[Service]\nOOMScoreAdjust=%s\n' "${SCORES[$svc]}" \
      > "/etc/systemd/system/$svc.service.d/oom-protect.conf"
    echo "  dropin: $svc -> ${SCORES[$svc]}" >> "$LOGF"
  fi
done

# --- 3. boot-persistent memory guard (host-side, protected itself) ---
cat > /usr/local/bin/dm-memguard.sh << 'EOG'
#!/bin/bash
# Kills the largest python inside dm-exp when host MemAvailable < 8GB
# (experiments are checkpointed/resumable; host reachability is not).
while true; do
  avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  if [ "$avail" -lt 8 ]; then
    line=$(docker exec dm-exp bash -c \
      "ps -eo pid,rss,comm --sort=-rss | awk '\$3~/python/{print \$1, \$2; exit}'" \
      2>/dev/null)
    pid=${line%% *}
    if [ -n "$pid" ]; then
      echo "$(date -Is) MEMGUARD_KILL avail=${avail}GB pid_rss='$line'" \
        >> /var/log/dm-memguard.log
      docker exec dm-exp kill -9 "$pid" 2>/dev/null
      sleep 10
    fi
  fi
  sleep 5
done
EOG
chmod +x /usr/local/bin/dm-memguard.sh
cat > /etc/systemd/system/dm-memguard.service << 'EOG'
[Unit]
Description=digital-minds memory guard (kills experiment procs before kernel OOM)
After=docker.service
Requires=docker.service
[Service]
ExecStart=/usr/local/bin/dm-memguard.sh
Restart=always
RestartSec=5
OOMScoreAdjust=-1000
Nice=-10
[Install]
WantedBy=multi-user.target
EOG
systemctl daemon-reload
systemctl enable --now dm-memguard.service >> "$LOGF" 2>&1

# --- 4. modest safety margin: keep kernel reclaim headroom ---
sysctl -w vm.min_free_kbytes=1048576 >> "$LOGF" 2>&1   # 1GB floor
grep -q dm-harden /etc/sysctl.d/99-dm-harden.conf 2>/dev/null || \
  printf '# dm-harden\nvm.min_free_kbytes=1048576\n' > /etc/sysctl.d/99-dm-harden.conf

# --- 5. report ---
echo "== verification =="
for n in sshd tailscaled dockerd containerd nginx cron; do
  for p in $(pgrep -x "$n" 2>/dev/null | head -2); do
    echo "$n pid=$p oom_score_adj=$(cat /proc/$p/oom_score_adj)"
  done
done
systemctl is-active dm-memguard.service | sed 's/^/dm-memguard: /'
echo HARDEN_DONE
