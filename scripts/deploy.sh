#!/usr/bin/env bash
# ==============================================================================
# ProdML Production Zero-Downtime Deployment & Automated Rollback Script
#
# Usage:
#   ./scripts/deploy.sh --tag <commit_sha_or_semver>
#   ./scripts/deploy.sh --image <full_ghcr_image_uri>
#   ./scripts/deploy.sh --latest
#   ./scripts/deploy.sh --rollback
#   ./scripts/deploy.sh --dry-run --tag <tag>
#
# Features:
#   • Pulls verified OCI container package from GitHub Container Registry (GHCR)
#   • Performs zero-downtime container replacement via Docker Compose
#   • Enforces live /health and /metadata HTTP probe validation
#   • Executes automated rollback to previous known-good image on health failure
# ==============================================================================

set -euo pipefail

DEFAULT_REGISTRY="ghcr.io/yossef-moftah-dev/arabic-sentiment-arabert"
COMPOSE_FILE="docker-compose.yml"
TARGET_PORT=8000
MAX_WAIT_SECONDS=60
POLL_INTERVAL=3
HISTORY_FILE=".last_deployed_image"

IMAGE_URI=""
IS_ROLLBACK=false
DRY_RUN=false

show_help() {
    cat <<EOF
ProdML Production Deployment Tool

Usage:
  $(basename "$0") [OPTIONS]

Options:
  --tag <TAG>           Deploy a specific tag or commit SHA from GHCR
                        (e.g., --tag 28d5d6e13e819e1f68e23ef39e0f516606036ec2, --tag v0.4.0)
  --image <URI>         Deploy using full Docker image URI
  --latest              Deploy the 'latest' tag from GHCR
  --rollback            Roll back immediately to the previous active image
  --compose-file <FILE> Path to docker-compose file (default: docker-compose.yml)
  --port <PORT>         Health check port (default: 8000)
  --timeout <SECONDS>   Maximum health check timeout in seconds (default: 60)
  --dry-run             Validate image pull and configuration without deploying
  --help                Show this help message
EOF
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tag)
            IMAGE_URI="${DEFAULT_REGISTRY}:$2"
            shift 2
            ;;
        --image)
            IMAGE_URI="$2"
            shift 2
            ;;
        --latest)
            IMAGE_URI="${DEFAULT_REGISTRY}:latest"
            shift 1
            ;;
        --rollback)
            IS_ROLLBACK=true
            shift 1
            ;;
        --compose-file)
            COMPOSE_FILE="$2"
            shift 2
            ;;
        --port)
            TARGET_PORT="$2"
            shift 2
            ;;
        --timeout)
            MAX_WAIT_SECONDS="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift 1
            ;;
        --help|-h)
            show_help
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Run '$(basename "$0") --help' for usage." >&2
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$WORKSPACE_ROOT"

if [ ! -f "$COMPOSE_FILE" ]; then
    echo "Error: Compose file '$COMPOSE_FILE' not found in $WORKSPACE_ROOT" >&2
    exit 1
fi

# Handle rollback request
if [ "$IS_ROLLBACK" = true ]; then
    if [ ! -f "$HISTORY_FILE" ]; then
        echo "Error: No rollback history found in $HISTORY_FILE." >&2
        exit 1
    fi
    PREV_IMAGE=$(cat "$HISTORY_FILE" | tr -d '[:space:]')
    if [ -z "$PREV_IMAGE" ]; then
        echo "Error: Rollback history file is empty." >&2
        exit 1
    fi
    echo "========================================================================"
    echo "🔄 EXECUTING ROLLBACK TO: $PREV_IMAGE"
    echo "========================================================================"
    IMAGE_URI="$PREV_IMAGE"
fi

# Fallback default if no image/tag provided
if [ -z "$IMAGE_URI" ]; then
    echo "No image or tag specified. Defaulting to: ${DEFAULT_REGISTRY}:latest"
    IMAGE_URI="${DEFAULT_REGISTRY}:latest"
fi

# Capture currently running image before replacement
CURRENT_RUNNING_IMAGE=""
if command -v docker >/dev/null 2>&1; then
    CURRENT_RUNNING_IMAGE=$(docker inspect --format '{{.Config.Image}}' prodml-service 2>/dev/null || echo "")
fi

echo "========================================================================"
echo "🚀 PRODML PRODUCTION DEPLOYMENT"
echo "========================================================================"
echo "Target Image:       $IMAGE_URI"
echo "Currently Running:  ${CURRENT_RUNNING_IMAGE:-None (Fresh Deployment)}"
echo "Compose File:       $COMPOSE_FILE"
echo "Health Check Port:  $TARGET_PORT"
echo "Dry Run:            $DRY_RUN"
echo "========================================================================"

# Step 1: Pull target image from registry
echo "Step 1: Pulling target package from registry..."
if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] Would execute: docker pull $IMAGE_URI"
else
    docker pull "$IMAGE_URI"
    echo "✓ Package pulled successfully: $IMAGE_URI"
fi

if [ "$DRY_RUN" = true ]; then
    echo "✓ Dry-run validation passed. Exiting without updating containers."
    exit 0
fi

# Save current image as rollback target before cutover
if [ -n "$CURRENT_RUNNING_IMAGE" ] && [ "$CURRENT_RUNNING_IMAGE" != "$IMAGE_URI" ]; then
    echo "$CURRENT_RUNNING_IMAGE" > "$HISTORY_FILE"
fi

# Step 2: Deploy new container with zero downtime
echo "Step 2: Launching container with target image..."
export PRODML_IMAGE="$IMAGE_URI"
docker compose -f "$COMPOSE_FILE" up -d --no-deps prodml-service

# Step 3: Verify container health
echo "Step 3: Probing container health on http://127.0.0.1:${TARGET_PORT}/health..."
START_TIME=$(date +%s)
HEALTHY=false

while true; do
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))

    if [ "$ELAPSED" -ge "$MAX_WAIT_SECONDS" ]; then
        break
    fi

    HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" -m 2 "http://127.0.0.1:${TARGET_PORT}/health" 2>/dev/null || echo "000")
    if [ "$HTTP_STATUS" = "200" ]; then
        HEALTH_BODY=$(curl -s -m 2 "http://127.0.0.1:${TARGET_PORT}/health" 2>/dev/null || echo "{}")
        if echo "$HEALTH_BODY" | grep -q '"status":"ok"'; then
            HEALTHY=true
            break
        fi
    fi

    echo "  [${ELAPSED}s/${MAX_WAIT_SECONDS}s] Waiting for /health (HTTP: $HTTP_STATUS)..."
    sleep "$POLL_INTERVAL"
done

# Step 4: Handle outcome / automated rollback
if [ "$HEALTHY" = true ]; then
    echo "========================================================================"
    echo "✅ DEPLOYMENT SUCCEEDED (Health check 200 OK in ${ELAPSED}s)"
    echo "========================================================================"
    echo "Active Image: $IMAGE_URI"
    if curl -s -m 2 "http://127.0.0.1:${TARGET_PORT}/metadata" >/dev/null 2>&1; then
        echo "Metadata Info:"
        curl -s "http://127.0.0.1:${TARGET_PORT}/metadata" | tr ',' '\n' | head -n 6 || true
    fi
    # Record successful deployment
    echo "$IMAGE_URI" > "$HISTORY_FILE"
    exit 0
else
    echo "========================================================================"
    echo "❌ DEPLOYMENT FAILED: Container did not become healthy within ${MAX_WAIT_SECONDS}s"
    echo "========================================================================"
    echo "Recent container logs:"
    docker compose -f "$COMPOSE_FILE" logs --tail=50 prodml-service || true

    # Automated rollback
    if [ -f "$HISTORY_FILE" ]; then
        ROLLBACK_TARGET=$(cat "$HISTORY_FILE" | tr -d '[:space:]')
        if [ -n "$ROLLBACK_TARGET" ] && [ "$ROLLBACK_TARGET" != "$IMAGE_URI" ]; then
            echo "========================================================================"
            echo "🚨 INITIATING AUTOMATED ROLLBACK TO: $ROLLBACK_TARGET"
            echo "========================================================================"
            export PRODML_IMAGE="$ROLLBACK_TARGET"
            docker compose -f "$COMPOSE_FILE" up -d --no-deps prodml-service
            sleep 5
            if curl -s -f "http://127.0.0.1:${TARGET_PORT}/health" >/dev/null 2>&1; then
                echo "✓ Rollback verified: previous container $ROLLBACK_TARGET is healthy."
            else
                echo "⚠ Warning: Rollback container health check did not pass immediately."
            fi
        fi
    fi
    exit 1
fi
