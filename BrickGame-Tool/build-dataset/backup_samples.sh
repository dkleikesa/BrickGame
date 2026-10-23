#!/usr/bin/env bash
# samples 不入库（原图留档，几百 M），冷备到仓库外。
# 用法：./build-dataset/backup_samples.sh
set -euo pipefail
cd "$(dirname "$0")"
stamp=$(date +%Y%m%d-%H%M)
out="$HOME/BrickGame-backups/samples-$stamp.tar.gz"
mkdir -p "$(dirname "$out")"
tar czf "$out" samples
echo "备份 → $out ($(du -h "$out" | cut -f1)，$(find samples -maxdepth 1 -type f | wc -l | tr -d ' ') 个文件)"
