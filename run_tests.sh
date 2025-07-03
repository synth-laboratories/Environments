#!/bin/bash

# Synth-Env Test Runner Script
# Usage: ./run_tests.sh [options]
# Options:
#   -u, --unit          Run unit tests only
#   -i, --integration   Run integration tests only
#   -s, --skip-service  Skip service check
#   -e, --env ENV_NAME  Run tests for specific environment only (tictactoe, sokoban, verilog, crafter_classic)
#   -v, --verbose       Verbose output
#   -h, --help          Show this help message

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
RUN_UNIT=true
RUN_INTEGRATION=true
CHECK_SERVICE=true
SPECIFIC_ENV=""
VERBOSE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -u|--unit)
            RUN_UNIT=true
            RUN_INTEGRATION=false
            shift
            ;;
        -i|--integration)
            RUN_UNIT=false
            RUN_INTEGRATION=true
            shift
            ;;
        -s|--skip-service)
            CHECK_SERVICE=false
            shift
            ;;
        -e|--env)
            SPECIFIC_ENV="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            echo "Synth-Env Test Runner"
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  -u, --unit          Run unit tests only"
            echo "  -i, --integration   Run integration tests only"
            echo "  -s, --skip-service  Skip service check"
            echo "  -e, --env ENV_NAME  Run tests for specific environment only"
            echo "                      (tictactoe, sokoban, verilog, crafter_classic)"
            echo "  -v, --verbose       Verbose output"
            echo "  -h, --help          Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Function to print colored messages
print_header() {
    echo -e "\n${BLUE}==== $1 ====${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Function to check if service is running
check_service() {
    if [ "$CHECK_SERVICE" = true ]; then
        print_header "Checking if service is running"
        
        # Check if service is running on default port 6532
        if curl -s http://localhost:6532/health > /dev/null 2>&1; then
            print_success "Service is running on http://localhost:6532"
        else
            print_warning "Service is not running on http://localhost:6532"
            echo "To start the service, run: cd src && python -m synth_env.service.app"
            echo "Continuing with tests that don't require the service..."
        fi
    fi
}

# Function to run tests for a specific environment
run_env_tests() {
    local env_name=$1
    local env_path="src/synth_env/examples/$env_name"
    
    if [ ! -d "$env_path" ]; then
        print_error "Environment '$env_name' not found at $env_path"
        return 1
    fi
    
    print_header "Testing $env_name environment"
    
    # Run unit tests
    if [ "$RUN_UNIT" = true ] && [ -d "$env_path/units" ]; then
        echo -e "\n${YELLOW}Running unit tests for $env_name...${NC}"
        if [ "$VERBOSE" = true ]; then
            PYTHONPATH=src python -m pytest "$env_path/units/" -v
        else
            PYTHONPATH=src python -m pytest "$env_path/units/" -q
        fi
        if [ $? -eq 0 ]; then
            print_success "Unit tests passed for $env_name"
        else
            print_error "Unit tests failed for $env_name"
            return 1
        fi
    fi
    
    # Run integration tests (agent demos)
    if [ "$RUN_INTEGRATION" = true ] && [ -d "$env_path/agent_demos" ]; then
        echo -e "\n${YELLOW}Running integration tests for $env_name...${NC}"
        
        # Find test files in agent_demos
        test_files=$(find "$env_path/agent_demos" -name "test_*.py" -type f)
        
        if [ -z "$test_files" ]; then
            print_warning "No integration tests found for $env_name"
        else
            for test_file in $test_files; do
                echo "Running $(basename $test_file)..."
                if [ "$VERBOSE" = true ]; then
                    PYTHONPATH=src python -m pytest "$test_file" -v -k "not eval_" --tb=short
                else
                    PYTHONPATH=src python -m pytest "$test_file" -q -k "not eval_" --tb=short
                fi
                if [ $? -eq 0 ]; then
                    print_success "$(basename $test_file) passed"
                else
                    print_error "$(basename $test_file) failed"
                    return 1
                fi
            done
        fi
    fi
    
    return 0
}

# Function to run type checking
run_type_check() {
    local env_name=$1
    local env_path="src/synth_env/examples/$env_name"
    
    if command -v uvx &> /dev/null; then
        echo -e "\n${YELLOW}Running type check for $env_name...${NC}"
        if uvx ty check "$env_path" 2>&1 | grep -q "error\["; then
            print_warning "Type errors found in $env_name (non-blocking)"
        else
            print_success "No type errors in $env_name"
        fi
    fi
}

# Main execution
main() {
    print_header "Synth-Env Test Runner"
    
    # Check Python version
    python_version=$(python --version 2>&1 | awk '{print $2}')
    echo "Python version: $python_version"
    
    # Check if pytest is installed
    if ! python -m pytest --version > /dev/null 2>&1; then
        print_error "pytest is not installed. Please install it with: pip install pytest"
        exit 1
    fi
    
    # Check service status
    check_service
    
    # List of all environments
    environments=("tictactoe" "sokoban" "verilog" "crafter_classic")
    
    # If specific environment is requested
    if [ -n "$SPECIFIC_ENV" ]; then
        if [[ " ${environments[@]} " =~ " ${SPECIFIC_ENV} " ]]; then
            environments=("$SPECIFIC_ENV")
        else
            print_error "Unknown environment: $SPECIFIC_ENV"
            echo "Available environments: ${environments[*]}"
            exit 1
        fi
    fi
    
    # Track overall success
    all_passed=true
    
    # Run tests for each environment
    for env in "${environments[@]}"; do
        if run_env_tests "$env"; then
            run_type_check "$env"
        else
            all_passed=false
        fi
    done
    
    # Run core tests if not testing specific environment
    if [ -z "$SPECIFIC_ENV" ]; then
        print_header "Testing core modules"
        
        # Test core task module
        if [ "$RUN_UNIT" = true ]; then
            echo -e "\n${YELLOW}Running core task tests...${NC}"
            if [ "$VERBOSE" = true ]; then
                PYTHONPATH=src python -m pytest src/synth_env/tasks/units/ -v
            else
                PYTHONPATH=src python -m pytest src/synth_env/tasks/units/ -q
            fi
            if [ $? -eq 0 ]; then
                print_success "Core task tests passed"
            else
                print_error "Core task tests failed"
                all_passed=false
            fi
        fi
    fi
    
    # Summary
    print_header "Test Summary"
    
    if [ "$all_passed" = true ]; then
        print_success "All tests passed!"
        exit 0
    else
        print_error "Some tests failed!"
        exit 1
    fi
}

# Run main function
main