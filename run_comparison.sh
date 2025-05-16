#!/bin/bash

# Default values
PHASE="test"
CONFIG="src/config.yml"
API_KEY=""
MODEL="gpt-3.5-turbo"

# Function to display usage
usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -p, --phase PHASE       Dataset phase (train, valid, test)"
    echo "  -c, --config CONFIG     Path to config file"
    echo "  -k, --api-key KEY       OpenAI API key"
    echo "  -m, --model MODEL       LLM model to use"
    echo "  -h, --help              Display this help message"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--phase)
            PHASE="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG="$2"
            shift 2
            ;;
        -k|--api-key)
            API_KEY="$2"
            shift 2
            ;;
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate phase
if [[ ! "$PHASE" =~ ^(train|valid|test)$ ]]; then
    echo "Invalid phase: $PHASE. Must be one of: train, valid, test"
    exit 1
fi

# Check if config file exists
if [ ! -f "$CONFIG" ]; then
    echo "Config file not found: $CONFIG"
    exit 1
fi

# Run the comparison script
CMD="python src/run_and_compare.py --phase $PHASE --config $CONFIG --model $MODEL"

# Add API key if provided
if [ -n "$API_KEY" ]; then
    CMD="$CMD --api-key $API_KEY"
fi

echo "Running command: $CMD"
$CMD

# Check if the script ran successfully
if [ $? -eq 0 ]; then
    echo "Comparison completed successfully!"
    echo "Results saved in: results/comparison"
else
    echo "Error running comparison script."
    exit 1
fi 