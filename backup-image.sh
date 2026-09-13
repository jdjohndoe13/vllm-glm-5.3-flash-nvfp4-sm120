#!/usr/bin/env bash
# ============================================================================
# OPTIONAL — recommended BEFORE wiping the drive.
# Save the vllm docker image to a tar file so it can be docker-loaded on the
# fresh Ubuntu install even if the registry is unreachable at that point.
# The image is ~29 GB; point OUT at an external/backup disk.
#
#   bash backup-image.sh /path/to/external/glm-image.tar
#
# Restore later with:  docker load -i glm-image.tar
# ============================================================================
set -euo pipefail
IMAGE="cstechdev/vllm:glm53-flash-nope-sm120-cu130-20260826-r1"
OUT="${1:?usage: backup-image.sh /path/to/output/glm-image.tar}"

mkdir -p "$(dirname "$OUT")"
echo "saving $IMAGE -> $OUT (this can take several minutes)"
docker save -o "$OUT" "$IMAGE"
ls -lh "$OUT"
echo "done. Restore on the new machine with: docker load -i $(basename "$OUT")"
