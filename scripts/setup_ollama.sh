#!/bin/bash
# FinSight — Ollama Setup Script
#
# Installs Ollama (if needed), pulls the default Qwen3.5-9B model,
# and verifies it's working correctly.
#
# Usage:
#   bash scripts/setup_ollama.sh              # Install + pull default model
#   bash scripts/setup_ollama.sh --model qwen3:8b  # Pull a specific model
#   bash scripts/setup_ollama.sh --check       # Just check the installation

set -euo pipefail

# --- Configuration ---
DEFAULT_MODEL="qwen3.5:9b"
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# --- Parse arguments ---
MODEL="${DEFAULT_MODEL}"
CHECK_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --check)
            CHECK_ONLY=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--model MODEL_NAME] [--check]"
            echo ""
            echo "Options:"
            echo "  --model MODEL  Model to pull (default: ${DEFAULT_MODEL})"
            echo "  --check        Only check installation, don't install"
            echo ""
            echo "Supported models for FinSight:"
            echo "  qwen3.5:9b     Qwen 3.5 9B (default, ~6.6GB, best quality)"
            echo "  qwen3:8b       Qwen 3 8B (~5GB, has RLM fine-tune available)"
            echo "  qwen3.5:4b     Qwen 3.5 4B (~3.4GB, fastest)"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}╔══════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     FinSight — Ollama Setup          ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════╝${NC}"
echo ""

# --- Step 1: Check if Ollama is installed ---
echo -e "${YELLOW}[1/4] Checking Ollama installation...${NC}"
if command -v ollama &> /dev/null; then
    OLLAMA_VERSION=$(ollama --version 2>/dev/null || echo "unknown")
    echo -e "${GREEN}  ✓ Ollama is installed: ${OLLAMA_VERSION}${NC}"
else
    if [ "$CHECK_ONLY" = true ]; then
        echo -e "${RED}  ✗ Ollama is not installed${NC}"
        echo "  Install with: curl -fsSL https://ollama.com/install.sh | sh"
        exit 1
    fi

    echo -e "${YELLOW}  Installing Ollama...${NC}"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS — check for Homebrew first
        if command -v brew &> /dev/null; then
            brew install ollama
        else
            echo "  Downloading Ollama for macOS..."
            curl -fsSL https://ollama.com/install.sh | sh
        fi
    else
        # Linux
        curl -fsSL https://ollama.com/install.sh | sh
    fi

    if command -v ollama &> /dev/null; then
        echo -e "${GREEN}  ✓ Ollama installed successfully${NC}"
    else
        echo -e "${RED}  ✗ Ollama installation failed${NC}"
        exit 1
    fi
fi

# --- Step 2: Check if Ollama server is running ---
echo -e "${YELLOW}[2/4] Checking Ollama server...${NC}"
if curl -s "${OLLAMA_URL}/api/version" > /dev/null 2>&1; then
    echo -e "${GREEN}  ✓ Ollama server is running at ${OLLAMA_URL}${NC}"
else
    echo -e "${YELLOW}  Starting Ollama server...${NC}"
    ollama serve &> /dev/null &
    sleep 3

    if curl -s "${OLLAMA_URL}/api/version" > /dev/null 2>&1; then
        echo -e "${GREEN}  ✓ Ollama server started${NC}"
    else
        echo -e "${RED}  ✗ Failed to start Ollama server${NC}"
        echo "  Try starting manually: ollama serve"
        exit 1
    fi
fi

if [ "$CHECK_ONLY" = true ]; then
    echo ""
    echo -e "${YELLOW}[3/4] Checking installed models...${NC}"
    ollama list
    echo ""
    echo -e "${GREEN}Check complete.${NC}"
    exit 0
fi

# --- Step 3: Pull the model ---
echo -e "${YELLOW}[3/4] Pulling model: ${MODEL}...${NC}"

# Check if model already exists
if ollama list 2>/dev/null | grep -q "${MODEL}"; then
    echo -e "${GREEN}  ✓ Model '${MODEL}' is already available${NC}"
else
    echo "  This may take several minutes depending on your connection..."
    echo "  Model sizes: 9B ~6.6GB, 8B ~5GB, 4B ~3.4GB"
    echo ""
    ollama pull "${MODEL}"

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}  ✓ Model '${MODEL}' pulled successfully${NC}"
    else
        echo -e "${RED}  ✗ Failed to pull model${NC}"
        exit 1
    fi
fi

# --- Step 4: Verify with a test prompt ---
echo -e "${YELLOW}[4/4] Verifying model with a test prompt...${NC}"

TEST_RESPONSE=$(ollama run "${MODEL}" "What is a 10-K filing? Reply in one sentence." 2>/dev/null | head -5)

if [ -n "${TEST_RESPONSE}" ]; then
    echo -e "${GREEN}  ✓ Model is working!${NC}"
    echo ""
    echo "  Test response:"
    echo "  ${TEST_RESPONSE}"
else
    echo -e "${RED}  ✗ Model did not respond${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}╔══════════════════════════════════════╗${NC}"
echo -e "${GREEN}║     Setup Complete!                  ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════╝${NC}"
echo ""
echo "To use FinSight:"
echo "  1. Copy .env.example to .env and configure"
echo "  2. uv sync"
echo "  3. uv run streamlit run mac/src/finsight_mac/ui/app.py"
echo ""
echo "Model: ${MODEL}"
echo "Ollama URL: ${OLLAMA_URL}"
