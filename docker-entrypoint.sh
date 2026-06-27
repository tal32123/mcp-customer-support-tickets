#!/bin/sh
# Seed /data from the prebuilt store on first boot. The image carries a
# fully-ingested LanceDB store at /opt/store-seed; once /data has content
# (subsequent boots, or a user-supplied volume) this is a no-op.
set -e
if [ -d /opt/store-seed ] && [ -z "$(ls -A /data 2>/dev/null)" ]; then
  echo "[entrypoint] seeding /data from /opt/store-seed (first boot)..."
  cp -a /opt/store-seed/. /data/
  echo "[entrypoint] seed complete."
fi
exec "$@"
