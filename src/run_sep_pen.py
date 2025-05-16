#!/usr/local/bin/python
import os
import argparse
import json
from SEP_Integration import SEP_PEN_Integration

def setup_arg_parser():
    """Set up command line argument parser"""
    parser = argparse.ArgumentParser(description='Run SEP-PEN Integration')
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='train', 
                        choices=['train', 'eval', 'full'],
                        help='Mode to run: train, eval, or full (both)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs for training')
    parser.add_argument('--config', type=str, default='src/config.yml',
                        help='Path to configuration file')
    
    # LLM configuration
    parser.add_argument('--llm_type', type=str, default='mock',
                        choices=['mock', 'openai', 'transformer'],
                        help='Type of LLM to use')
    parser.add_argument('--model_name', type=str, default='gpt-3.5-turbo',
                        help='Model name for OpenAI or transformer LLM')
    parser.add_argument('--api_key', type=str, default=None,
                        help='API key for OpenAI (if not set in environment)')
    
    # Evaluation parameters
    parser.add_argument('--eval_phase', type=str, default='test',
                        choices=['dev', 'test'],
                        help='Evaluation phase: dev or test')
    
    # Output directory
    parser.add_argument('--output_dir', type=str, default='results/sep_pen_integration',
                        help='Directory to save results')
    
    return parser

def main():
    """Main function to run SEP-PEN integration"""
    parser = setup_arg_parser()
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configure LLM
    llm_config = {
        "type": args.llm_type,
        "model": args.model_name,
        "max_tokens": 500,
        "temperature": 0.7
    }
    
    if args.api_key:
        llm_config["api_key"] = args.api_key
    
    # Create integration instance
    integration = SEP_PEN_Integration(
        config_path=args.config,
        llm_config=llm_config
    )
    
    # Set results directory
    integration.results_dir = args.output_dir
    
    # Run based on selected mode
    results = {}
    
    if args.mode in ['train', 'full']:
        print(f"Starting training for {args.epochs} epochs")
        integration.train(n_epochs=args.epochs)
        
        # Optimize LLM with PPO after training if using a real LLM
        if args.llm_type != 'mock':
            print("Starting LLM optimization with PPO")
            integration.optimize_llm_with_ppo(n_iterations=3)
            
        results['training'] = {
            'epochs': args.epochs,
            'config': args.config
        }
    
    if args.mode in ['eval', 'full']:
        print(f"Starting evaluation on {args.eval_phase} set")
        integration.evaluate(phase=args.eval_phase)
        # The evaluate method doesn't return values, it saves results to files
        results['evaluation'] = {
            'phase': args.eval_phase,
            'output_dir': os.path.join(args.output_dir, f"{args.eval_phase}_evaluation")
        }
    
    # Save run configuration and results
    with open(os.path.join(args.output_dir, 'run_config.json'), 'w') as f:
        config = {
            'args': vars(args),
            'results': results
        }
        json.dump(config, f, indent=2)
    
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 