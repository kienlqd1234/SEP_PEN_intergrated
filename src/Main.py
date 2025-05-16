#!/usr/local/bin/python
import os
import sys
import argparse
from SEP_Integration import SEP_PEN_Integration

def main():
    """
    Run the SEP-PEN integration using OpenAI API
    without requiring PyTorch or transformers
    """
    parser = argparse.ArgumentParser(description='Run SEP-PEN Integration with OpenAI')
    
    # Add arguments
    parser.add_argument('--openai_api_key', type=str, required=True,
                       help='OpenAI API key')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo',
                       help='OpenAI model to use')
    parser.add_argument('--mode', type=str, default='eval',
                       choices=['train', 'eval', 'full'],
                       help='Mode to run: train, eval, or full')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of epochs for training')
    parser.add_argument('--output_dir', type=str, default='results/openai_run',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set OpenAI API key
    os.environ['OPENAI_API_KEY'] = args.openai_api_key
    
    # Configure LLM to use OpenAI
    llm_config = {
        "type": "openai",
        "model": args.model,
        "max_tokens": 500,
        "temperature": 0.7,
        "api_key": args.openai_api_key
    }
    
    # Create integration instance
    print("Initializing SEP-PEN Integration with OpenAI...")
    integration = SEP_PEN_Integration(
        config_path="src/config.yml",
        llm_config=llm_config
    )
    
    # Set results directory
    integration.results_dir = args.output_dir
    
    # Run based on mode
    if args.mode in ['train', 'full']:
        print(f"Starting training for {args.epochs} epochs")
        integration.train(n_epochs=args.epochs)
    
    if args.mode in ['eval', 'full']:
        print("Starting evaluation on test set")
        accuracy, examples = integration.evaluate(phase="test")
        print(f"Evaluation accuracy: {accuracy:.4f}")
    
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()

