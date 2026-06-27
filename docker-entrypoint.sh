#!/bin/sh
# Wait for Postgres to accept connections, then exec the server.
# First boot ingests the HF dataset into the configured schema; subsequent
# boots reuse the existing rows (the store's revision sidecar in the
# store_meta table is what gates rebuild vs. open).
set -e

if [ -n "$DATABASE_URL" ]; then
  echo "[entrypoint] waiting for Postgres at $(echo "$DATABASE_URL" | sed 's|.*@||')..."
  python - <<'PY'
import os, sys, time
import psycopg
dsn = os.environ["DATABASE_URL"]
deadline = time.time() + 60
while time.time() < deadline:
    try:
        with psycopg.connect(dsn, connect_timeout=3):
            print("[entrypoint] Postgres is up.")
            sys.exit(0)
    except Exception as e:
        print(f"[entrypoint] not ready ({e.__class__.__name__}); retrying...")
        time.sleep(2)
print("[entrypoint] timed out waiting for Postgres", file=sys.stderr)
sys.exit(1)
PY
fi

exec "$@"
