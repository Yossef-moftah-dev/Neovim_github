#!/usr/bin/env bash
# ==============================================================================
# Helper script to tunnel all remote ProdML services to your local machine
# Usage: ./scripts/tunnel_services.sh [KEY_PATH] [EC2_HOST] [--alt] [--stop-local]
# ==============================================================================

set -euo pipefail

DEFAULT_HOST="${EC2_HOST:-ec2-54-167-218-116.compute-1.amazonaws.com}"
DEFAULT_USER="ubuntu"
ALT_PORTS=0
STOP_LOCAL=0

# Parse arguments
ARGS=()
for arg in "$@"; do
    case "$arg" in
        --alt|--alternate)
            ALT_PORTS=1
            ;;
        --stop-local|-s)
            STOP_LOCAL=1
            ;;
        *)
            ARGS+=("$arg")
            ;;
    esac
done

ARG1="${ARGS[0]:-}"
ARG2="${ARGS[1]:-}"

# Resolve host and key
EC2_HOST="$DEFAULT_HOST"
KEY_PATH=""

if [ -n "$ARG1" ]; then
    if [[ "$ARG1" == *"amazonaws.com"* ]] || [[ "$ARG1" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] || [[ "$ARG1" =~ ^ec2- ]]; then
        EC2_HOST="$ARG1"
    else
        KEY_PATH="$ARG1"
        if [ -n "$ARG2" ]; then
            EC2_HOST="$ARG2"
        fi
    fi
fi

# If key was not explicitly provided or resolved, auto-detect
if [ -z "$KEY_PATH" ]; then
    for candidate in \
        "${HOME}/Downloads/kk.pem" \
        "${HOME}/Downloads/keypair.pem" \
        "./kk.pem" \
        "./keypair.pem" \
        "${HOME}/.ssh/kk.pem"; do
        if [ -f "$candidate" ]; then
            KEY_PATH="$candidate"
            break
        fi
    done
fi

# Fallback: check if relative to ~/Downloads
if [ -n "$KEY_PATH" ] && [ ! -f "$KEY_PATH" ] && [ -f "${HOME}/Downloads/${KEY_PATH}" ]; then
    KEY_PATH="${HOME}/Downloads/${KEY_PATH}"
fi

if [ -z "$KEY_PATH" ] || [ ! -f "$KEY_PATH" ]; then
    echo "❌ Error: Private key not found." >&2
    echo "Looked for kk.pem in ~/Downloads and current directory." >&2
    echo "Usage: $0 [path/to/kk.pem] [ec2-instance-hostname-or-ip]" >&2
    exit 1
fi

# 1. Enforce strict permissions on the key
chmod 400 "$KEY_PATH" 2>/dev/null || true
echo "✓ Key secured (chmod 400): $KEY_PATH"

# 2. Check for port collisions on local machine
COLLIDING_PORTS=()
for port in 5000 8000 9000 9001; do
    if ss -H -tl "sport = :$port" 2>/dev/null | grep -q "$port" || nc -z 127.0.0.1 "$port" 2>/dev/null; then
        COLLIDING_PORTS+=("$port")
    fi
done

if [ ${#COLLIDING_PORTS[@]} -gt 0 ]; then
    if [ "$STOP_LOCAL" -eq 1 ]; then
        echo "⚠️ Stopping local Docker containers to free standard ports..."
        docker compose stop || true
    elif [ "$ALT_PORTS" -eq 0 ]; then
        echo "⚠️ Notice: Local port(s) [${COLLIDING_PORTS[*]}] are currently in use."
        echo "   If local Docker containers are running, you can stop them with:"
        echo "     docker compose stop"
        echo "   Or run with alternate local ports (15000, 18000, 19000, 19001):"
        echo "     $0 --alt"
        echo "   Or auto-stop local containers:"
        echo "     $0 --stop-local"
        echo ""
        read -r -p "Stop local containers now to free ports? [y/N]: " answer || answer="n"
        if [[ "$answer" =~ ^[Yy]$ ]]; then
            docker compose stop || true
        else
            echo "Switching to alternate local ports..."
            ALT_PORTS=1
        fi
    fi
fi

# Assign local forwarding ports
if [ "$ALT_PORTS" -eq 1 ]; then
    L_MLFLOW=15000
    L_API=18000
    L_MINIO_API=19000
    L_MINIO_CONSOLE=19001
else
    L_MLFLOW=5000
    L_API=8000
    L_MINIO_API=9000
    L_MINIO_CONSOLE=9001
fi

# 3. Test SSH connectivity
echo "Connecting to: ${DEFAULT_USER}@${EC2_HOST}..."
if ! ssh -q -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=no -i "$KEY_PATH" "${DEFAULT_USER}@${EC2_HOST}" "exit" 2>/dev/null; then
    echo "⚠️ Warning: Could not verify SSH connection to ${DEFAULT_USER}@${EC2_HOST}."
    echo "   Ensure the EC2 instance is running and Security Group allows inbound SSH (Port 22)."
fi

echo "========================================================================"
echo " Establishing secure SSH tunnel to: $EC2_HOST"
echo "========================================================================"
echo " Services forwarded to your local machine:"
echo "   • MLflow Tracking UI: http://localhost:${L_MLFLOW}"
echo "   • MinIO Web Console:  http://localhost:${L_MINIO_CONSOLE} (User: minioadmin / minioadmin)"
echo "   • MinIO S3 API:       http://localhost:${L_MINIO_API}"
echo "   • FastAPI Docs:       http://localhost:${L_API}/docs"
echo "   • FastAPI Health:     http://localhost:${L_API}/health"
echo "========================================================================"
echo "Press Ctrl+C to disconnect the tunnel."

exec ssh -N \
    -o StrictHostKeyChecking=no \
    -o ServerAliveInterval=60 \
    -o ServerAliveCountMax=3 \
    -o ExitOnForwardFailure=yes \
    -i "$KEY_PATH" \
    -L "${L_MLFLOW}:localhost:5000" \
    -L "${L_MINIO_CONSOLE}:localhost:9001" \
    -L "${L_MINIO_API}:localhost:9000" \
    -L "${L_API}:localhost:8000" \
    "${DEFAULT_USER}@${EC2_HOST}"
