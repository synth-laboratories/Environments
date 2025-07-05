#!/bin/bash
# Run the unified Synth environment viewer

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default values
PORT=8999
EVAL_DIR="src/evals"

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --latest       View the latest evaluation"
    echo "  --list         List all available evaluations"
    echo "  --env ENV      Filter by environment (crafter, nethack, sokoban, etc.)"
    echo "  --run PATH     View a specific evaluation directory"
    echo "  --port PORT    Port to run viewer on (default: 8999)"
    echo "  --eval-dir DIR Base directory for evaluations (default: src/evals)"
    echo "  --help         Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                    # View landing page"
    echo "  $0 --list             # List all evaluations"
    echo "  $0 --env crafter      # View latest Crafter evaluation"
    echo "  $0 --run src/evals/crafter/run_20240115_143022"
    echo "  $0 --port 8080        # Use different port"
}

# Parse command line arguments
ARGS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            usage
            exit 0
            ;;
        --port)
            PORT="$2"
            ARGS="$ARGS --port $PORT"
            shift 2
            ;;
        --eval-dir)
            EVAL_DIR="$2"
            ARGS="$ARGS --eval-dir $EVAL_DIR"
            shift 2
            ;;
        --list)
            ARGS="$ARGS --list"
            shift
            ;;
        --run)
            ARGS="$ARGS --run $2"
            shift 2
            ;;
        --env)
            ARGS="$ARGS --env $2"
            shift 2
            ;;
        --latest)
            ARGS="$ARGS --latest"
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            exit 1
            ;;
    esac
done

# If no specific option given, we'll get the landing page
# Don't add --latest by default anymore

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}Error: Must run from the Environments repository root${NC}"
    echo "Please cd to the Environments directory first"
    exit 1
fi

# Check if evaluation directory exists
if [ ! -d "$EVAL_DIR" ]; then
    echo -e "${RED}Error: No evaluations found at $EVAL_DIR${NC}"
    echo "Run an evaluation first with one of:"
    echo "  python -m src.synth_env.examples.crafter_classic.agent_demos.run_full_enchilada"
    echo "  # (or other environment evaluation scripts)"
    exit 1
fi

# Kill any existing process on the port (only if not listing)
if [[ "$ARGS" != *"--list"* ]]; then
    if lsof -ti:$PORT > /dev/null 2>&1; then
        echo -e "${BLUE}Killing existing process on port $PORT...${NC}"
        lsof -ti:$PORT | xargs kill -9 2>/dev/null || true
        sleep 1
    fi
fi

# Run the viewer
echo -e "${BLUE}🎮 Unified Synth Environment Viewer${NC}"

if [[ -z "$ARGS" ]] || [[ "$ARGS" == *"--port"* && ! "$ARGS" == *"--latest"* && ! "$ARGS" == *"--run"* && ! "$ARGS" == *"--list"* ]]; then
    echo "🌐 Launching landing page at http://localhost:$PORT"
else
    echo "📊 Launching viewer with options: $ARGS"
fi
echo ""

python -m src.synth_env.viewer.unified_viewer $ARGS