#!/usr/bin/env bash
# ==============================================================================
# ProdML Production Docker Image Verification & Smoke Testing Suite
#
# Validates:
#   1. Static Image & Security Inspection (Non-root UID, exposed ports, healthcheck)
#   2. Container Lifecycle & Healthcheck Transition (starting -> healthy)
#   3. Functional API Smoke Tests (/health, /metadata, /predict, /predict/batch)
#   4. Input Validation & Fault Tolerance (422 rejection, correlation ID tracing)
#   5. Graceful SIGTERM Shutdown & Degraded Mode Resilience (503 without crash)
#
# Usage:
#   ./scripts/test_docker_image.sh [IMAGE_TAG] [TEST_PORT]
#
# Examples:
#   ./scripts/test_docker_image.sh prodml-service:latest
#   ./scripts/test_docker_image.sh prodml-service:latest 8000
# ==============================================================================

set -euo pipefail

# ANSI Color Codes
GREEN="\033[0;32m"
RED="\033[0;31m"
YELLOW="\033[1;33m"
CYAN="\033[0;36m"
BOLD="\033[1m"
NC="\033[0m" # No Color

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${1:-prodml-service:latest}"
PORT="${2:-8000}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_ROOT/outputs/final_model}"
CONTAINER_NAME="prodml-image-test-$$"

PASSED_COUNT=0
FAILED_COUNT=0

log_header() {
    echo -e "\n${BOLD}${CYAN}=== $1 ===${NC}"
}

assert_pass() {
    echo -e "  [${GREEN}PASS${NC}] $1"
    PASSED_COUNT=$((PASSED_COUNT + 1))
}

assert_fail() {
    echo -e "  [${RED}FAIL${NC}] $1"
    if [ -n "${2:-}" ]; then
        echo -e "         ${YELLOW}Details: $2${NC}"
    fi
    FAILED_COUNT=$((FAILED_COUNT + 1))
}

cleanup() {
    local exit_code=$?
    echo -e "\n${CYAN}>>> Cleaning up test containers...${NC}"
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        docker rm -f "${CONTAINER_NAME}" > /dev/null 2>&1 || true
    fi
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}-degraded$"; then
        docker rm -f "${CONTAINER_NAME}-degraded" > /dev/null 2>&1 || true
    fi
    exit "$exit_code"
}
trap cleanup EXIT INT TERM

echo -e "${BOLD}========================================================================${NC}"
echo -e "${BOLD}PRODML DOCKER CONTAINER IMAGE VERIFICATION SUITE${NC}"
echo -e "${BOLD}========================================================================${NC}"
echo "Target Image:     $IMAGE_NAME"
echo "Test Port:        $PORT"
echo "Model Directory:  $MODEL_DIR"
echo "Container Name:   $CONTAINER_NAME"
echo "------------------------------------------------------------------------"

# ------------------------------------------------------------------------------
# Pre-flight: Check Prerequisites
# ------------------------------------------------------------------------------
log_header "0. Pre-Flight Checks"

if ! command -v docker &> /dev/null; then
    assert_fail "Docker CLI is installed and available in PATH"
    exit 1
fi
assert_pass "Docker CLI detected: $(docker --version)"

if ! docker image inspect "$IMAGE_NAME" > /dev/null 2>&1; then
    assert_fail "Target Docker image '$IMAGE_NAME' exists locally" "Image not found. Run 'docker build -t $IMAGE_NAME -f docker/Dockerfile .' first."
    exit 1
fi
assert_pass "Target image '$IMAGE_NAME' is present"

if [ ! -d "$MODEL_DIR" ] || [ ! -f "$MODEL_DIR/model.safetensors" ]; then
    echo -e "  [${YELLOW}WARN${NC}] Model weights not found in $MODEL_DIR. Checking ONNX fallback..."
    if [ ! -f "$MODEL_DIR/model.onnx" ]; then
        assert_fail "Model weights directory contains model.safetensors or model.onnx"
    else
        assert_pass "Found ONNX weights in $MODEL_DIR"
    fi
else
    assert_pass "Validated model directory with model.safetensors: $MODEL_DIR"
fi

# ------------------------------------------------------------------------------
# Phase 1: Static Image & Security Inspection
# ------------------------------------------------------------------------------
log_header "1. Static Image & Security Inspection"

# Check 1.1: Non-root User Configuration
CONFIG_USER=$(docker inspect --format '{{.Config.User}}' "$IMAGE_NAME")
if [ "$CONFIG_USER" = "appuser" ] || [ "$CONFIG_USER" = "10001" ] || [ "$CONFIG_USER" = "10001:10001" ]; then
    assert_pass "Non-root execution enforced (User: '$CONFIG_USER')"
else
    assert_fail "Image user is not configured as non-root appuser/10001" "Current User: '$CONFIG_USER'"
fi

# Check 1.2: Exposed Ports
EXPOSED_PORTS=$(docker inspect --format '{{json .Config.ExposedPorts}}' "$IMAGE_NAME")
if echo "$EXPOSED_PORTS" | grep -q "8000/tcp"; then
    assert_pass "Standard service port 8000/tcp is exposed"
else
    assert_fail "Port 8000/tcp not found in exposed ports" "$EXPOSED_PORTS"
fi

# Check 1.3: Baked-in Healthcheck Directive
HEALTHCHECK_CMD=$(docker inspect --format '{{json .Config.Healthcheck}}' "$IMAGE_NAME")
if [ "$HEALTHCHECK_CMD" != "null" ] && echo "$HEALTHCHECK_CMD" | grep -q "curl"; then
    assert_pass "Built-in Docker HEALTHCHECK directive present in metadata"
else
    assert_fail "Built-in Docker HEALTHCHECK directive missing or not using curl" "$HEALTHCHECK_CMD"
fi

# Check 1.4: Multi-stage Lean Build (Compilers excluded from runtime)
COMPILER_CHECK=$(docker run --rm --entrypoint /bin/sh "$IMAGE_NAME" -c "which gcc make g++ 2>/dev/null || true")
if [ -z "$COMPILER_CHECK" ]; then
    assert_pass "Runtime image is lean (build-essential / gcc / make excluded)"
else
    assert_fail "Compilers detected in runtime stage" "$COMPILER_CHECK"
fi

# ------------------------------------------------------------------------------
# Phase 2: Container Lifecycle & Healthcheck Transition
# ------------------------------------------------------------------------------
log_header "2. Container Lifecycle & Healthcheck Transitions"

echo ">>> Launching container '$CONTAINER_NAME' on port $PORT..."
docker run -d \
    --name "$CONTAINER_NAME" \
    -p "$PORT:8000" \
    -v "$MODEL_DIR:/app/outputs/final_model:ro" \
    "$IMAGE_NAME" > /dev/null

assert_pass "Container started in detached mode"

# Verify runtime UID inside active container
CONTAINER_UID=$(docker exec "$CONTAINER_NAME" id -u 2>/dev/null || echo "unknown")
if [ "$CONTAINER_UID" = "10001" ]; then
    assert_pass "Active container process runs strictly with UID 10001 (appuser)"
else
    assert_fail "Active container running with unexpected UID" "UID: $CONTAINER_UID (expected 10001)"
fi

# Monitor health status transition
echo ">>> Polling container health status transition (timeout: 45s)..."
MAX_WAIT=45
WAIT_COUNT=0
HEALTH_STATUS="starting"

while [ "$WAIT_COUNT" -lt "$MAX_WAIT" ]; do
    HEALTH_STATUS=$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "$CONTAINER_NAME")
    if [ "$HEALTH_STATUS" = "healthy" ]; then
        break
    fi
    sleep 2
    WAIT_COUNT=$((WAIT_COUNT + 2))
done

if [ "$HEALTH_STATUS" = "healthy" ]; then
    assert_pass "Docker engine confirmed container health: 'healthy' (${WAIT_COUNT}s)"
else
    assert_fail "Container did not reach 'healthy' state within ${MAX_WAIT}s" "Status: $HEALTH_STATUS"
    echo -e "${YELLOW}Container Logs:${NC}"
    docker logs --tail 30 "$CONTAINER_NAME"
fi

# ------------------------------------------------------------------------------
# Phase 3: Functional API Smoke Suite
# ------------------------------------------------------------------------------
log_header "3. Functional API Smoke Tests"

BASE_URL="http://127.0.0.1:$PORT"

# Test 3.1: GET /health
HEALTH_CODE=$(curl -s -o /tmp/health_resp.json -w "%{http_code}" "$BASE_URL/health")
if [ "$HEALTH_CODE" = "200" ] && grep -q '"status":"ok"' /tmp/health_resp.json && grep -q '"model_loaded":true' /tmp/health_resp.json; then
    assert_pass "GET /health returned HTTP 200 with status='ok' and model_loaded=true"
else
    assert_fail "GET /health failed" "HTTP $HEALTH_CODE, Body: $(cat /tmp/health_resp.json 2>/dev/null)"
fi

# Test 3.2: GET /metadata
META_CODE=$(curl -s -o /tmp/meta_resp.json -w "%{http_code}" "$BASE_URL/metadata")
if [ "$META_CODE" = "200" ] && grep -q '"Negative"' /tmp/meta_resp.json && grep -q '"Positive"' /tmp/meta_resp.json; then
    assert_pass "GET /metadata returned HTTP 200 with 3-class Arabic labels"
else
    assert_fail "GET /metadata failed" "HTTP $META_CODE, Body: $(cat /tmp/meta_resp.json 2>/dev/null)"
fi

# Test 3.3: POST /predict (Single Prediction)
PRED_PAYLOAD='{"text": "المنتج رائع جدا وتوصيل سريع وممتاز"}'
PRED_CODE=$(curl -s -o /tmp/pred_resp.json -w "%{http_code}" \
    -X POST "$BASE_URL/predict" \
    -H "Content-Type: application/json" \
    -d "$PRED_PAYLOAD")

if [ "$PRED_CODE" = "200" ] && grep -q '"label"' /tmp/pred_resp.json && grep -q '"probabilities"' /tmp/pred_resp.json; then
    LABEL=$(grep -o '"label":"[^"]*"' /tmp/pred_resp.json | cut -d'"' -f4)
    CONF=$(grep -o '"confidence":[0-9.]*' /tmp/pred_resp.json | cut -d':' -f2)
    assert_pass "POST /predict returned HTTP 200 (Class: $LABEL, Confidence: $CONF)"
else
    assert_fail "POST /predict failed" "HTTP $PRED_CODE, Body: $(cat /tmp/pred_resp.json 2>/dev/null)"
fi

# Test 3.4: POST /predict/batch (Batch Prediction)
BATCH_PAYLOAD='{"texts": ["خدمة عملاء رائعة ومحترمة", "تجربة سيئة للغاية ولن أتعامل معكم ثانية"]}'
BATCH_CODE=$(curl -s -o /tmp/batch_resp.json -w "%{http_code}" \
    -X POST "$BASE_URL/predict/batch" \
    -H "Content-Type: application/json" \
    -d "$BATCH_PAYLOAD")

if [ "$BATCH_CODE" = "200" ] && grep -q '\[' /tmp/batch_resp.json; then
    ITEM_COUNT=$(grep -o '"label"' /tmp/batch_resp.json | wc -l)
    if [ "$ITEM_COUNT" -eq 2 ]; then
        assert_pass "POST /predict/batch returned HTTP 200 with 2 predicted items"
    else
        assert_fail "POST /predict/batch item count mismatch" "Expected 2 items, got $ITEM_COUNT"
    fi
else
    assert_fail "POST /predict/batch failed" "HTTP $BATCH_CODE, Body: $(cat /tmp/batch_resp.json 2>/dev/null)"
fi

# Test 3.5: Input Validation & Schema Enforcement (Empty string -> HTTP 422)
EMPTY_PAYLOAD='{"text": "   "}'
EMPTY_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
    -X POST "$BASE_URL/predict" \
    -H "Content-Type: application/json" \
    -d "$EMPTY_PAYLOAD")

if [ "$EMPTY_CODE" = "422" ]; then
    assert_pass "POST /predict with blank whitespace rejected with HTTP 422 Unprocessable Entity"
else
    assert_fail "POST /predict empty text did not reject with 422" "HTTP $EMPTY_CODE"
fi

# Test 3.6: Distributed Request Tracing (X-Request-ID Header Passthrough)
CUSTOM_REQ_ID="smoke-trace-id-$$-999"
TRACE_HEADER=$(curl -s -I -X GET "$BASE_URL/health" -H "X-Request-ID: $CUSTOM_REQ_ID" | grep -i "X-Request-ID:" || true)

if echo "$TRACE_HEADER" | grep -q "$CUSTOM_REQ_ID"; then
    assert_pass "X-Request-ID header preserved and returned in HTTP response ($CUSTOM_REQ_ID)"
else
    assert_fail "X-Request-ID header passthrough failed" "Header received: $TRACE_HEADER"
fi

# ------------------------------------------------------------------------------
# Phase 4: Lifecycle Teardown & Degraded Mode Resilience
# ------------------------------------------------------------------------------
log_header "4. Shutdown & Resilience"

# Test 4.1: Graceful SIGTERM Container Shutdown
echo ">>> Stopping container '$CONTAINER_NAME' with 10s graceful termination window..."
START_STOP=$(date +%s)
docker stop -t 10 "$CONTAINER_NAME" > /dev/null
END_STOP=$(date +%s)
STOP_DURATION=$((END_STOP - START_STOP))

if [ "$STOP_DURATION" -le 10 ]; then
    assert_pass "Container gracefully terminated on SIGTERM in ${STOP_DURATION}s"
else
    assert_fail "Container termination timed out or took longer than 10s" "${STOP_DURATION}s"
fi

docker rm "$CONTAINER_NAME" > /dev/null

# Test 4.2: Degraded Mode Resilience (Container starts without model weights -> 503, not crash)
echo ">>> Testing degraded mode startup without mounted model weights..."
EMPTY_MNT=$(mktemp -d)
docker run -d \
    --name "${CONTAINER_NAME}-degraded" \
    -p "$PORT:8000" \
    -v "$EMPTY_MNT:/app/outputs/final_model:ro" \
    "$IMAGE_NAME" > /dev/null

DEGRADED_CODE="000"
for i in $(seq 1 25); do
    DEGRADED_CODE=$(curl -s -o /tmp/degraded.json -w "%{http_code}" "$BASE_URL/health" || echo "000")
    if [ "$DEGRADED_CODE" = "503" ] || [ "$DEGRADED_CODE" = "200" ]; then
        break
    fi
    sleep 1
done

docker rm -f "${CONTAINER_NAME}-degraded" > /dev/null 2>&1 || true
rm -rf "$EMPTY_MNT"

if [ "$DEGRADED_CODE" = "503" ] && grep -q '"status":"degraded"' /tmp/degraded.json; then
    assert_pass "Degraded mode verified: Missing weights yields HTTP 503 without process crash"
else
    assert_fail "Degraded mode test failed" "HTTP $DEGRADED_CODE, Body: $(cat /tmp/degraded.json 2>/dev/null)"
fi

# ------------------------------------------------------------------------------
# Summary Report
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}========================================================================${NC}"
echo -e "${BOLD}TEST SUMMARY RESULTS${NC}"
echo -e "${BOLD}========================================================================${NC}"
echo -e "  Total Tests Executed: $((PASSED_COUNT + FAILED_COUNT))"
echo -e "  Passed:               ${GREEN}${PASSED_COUNT}${NC}"
echo -e "  Failed:               ${RED}${FAILED_COUNT}${NC}"
echo "------------------------------------------------------------------------"

if [ "$FAILED_COUNT" -eq 0 ]; then
    echo -e "${GREEN}${BOLD}✓ ALL IMAGE TESTS PASSED SUCCESSFULLY! The image is production-ready.${NC}\n"
    exit 0
else
    echo -e "${RED}${BOLD}✗ SOME IMAGE TESTS FAILED. Please review the failure logs above.${NC}\n"
    exit 1
fi
