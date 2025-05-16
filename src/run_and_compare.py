#!/usr/bin/env python
import os
import sys
import json
import numpy as np
import tensorflow as tf
from datetime import datetime
import argparse
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

# Import PEN model components
from DataPipe import DataPipe
from Model import Model as PENModel

# Import SEP integration
from SEP_Integration import SEP_PEN_Integration

# Import metrics
from metrics import eval_res, n_accurate

class ModelComparison:
    """Compares original PEN model with SEP-PEN integrated model"""
    
    def __init__(self, config_path="config.yml", llm_config=None):
        self.config_path = config_path
        self.llm_config = llm_config or {
            "type": "openai",
            "model": "gpt-3.5-turbo",
            "max_tokens": 500,
            "temperature": 0.7
        }
        
        # Initialize DataPipe for consistent data feeding
        self.pipe = DataPipe()
        
        # Results storage
        self.results_dir = "../results/comparison"
        os.makedirs(self.results_dir, exist_ok=True)
        
        print("Initializing models...")
        
    def _initialize_original_pen(self):
        """Initialize the original PEN model"""
        print("Initializing original PEN model...")
        pen_model = PENModel()
        pen_model.assemble_graph()
        return pen_model
        
    def _initialize_sep_pen(self):
        """Initialize the SEP-PEN integrated model"""
        print("Initializing SEP-PEN integrated model...")
        sep_pen = SEP_PEN_Integration(
            config_path=self.config_path,
            llm_config=self.llm_config
        )
        return sep_pen
    
    def _create_feed_dict(self, model, batch_dict, is_training=False):
        """Create feed dict for model from batch dict"""
        return {
            model.is_training_phase: is_training,
            model.batch_size: batch_dict['batch_size'],
            model.stock_ph: batch_dict['stock_batch'],
            model.T_ph: batch_dict['T_batch'],
            model.n_words_ph: batch_dict['n_words_batch'],
            model.n_msgs_ph: batch_dict['n_msgs_batch'],
            model.y_ph: batch_dict['y_batch'],
            model.price_ph: batch_dict['price_batch'],
            model.mv_percent_ph: batch_dict['mv_percent_batch'],
            model.word_ph: batch_dict['word_batch'],
            model.ss_index_ph: batch_dict['ss_index_batch'],
            # Set dropout to 0 when not training
            model.dropout_mel_in: 0.0 if not is_training else None,
            model.dropout_mel: 0.0 if not is_training else None,
            model.dropout_ce: 0.0 if not is_training else None,
            model.dropout_vmd_in: 0.0 if not is_training else None,
            model.dropout_vmd: 0.0 if not is_training else None,
        }
    
    def _initialize_session(self, model):
        """Initialize TensorFlow session with model"""
        tf_config = tf.ConfigProto(allow_soft_placement=True)
        tf_config.gpu_options.allow_growth = True
        
        sess = tf.Session(config=tf_config)
        
        # Initialize word table
        word_table_init = self.pipe.init_word_table()
        feed_table_init = {model.word_table_init: word_table_init}
        sess.run(tf.global_variables_initializer(), feed_dict=feed_table_init)
        
        # Try to restore checkpoint if available
        saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state(os.path.dirname(model.tf_checkpoint_file_path))
        
        if checkpoint and checkpoint.model_checkpoint_path:
            # Restore saved vars
            reader = tf.train.NewCheckpointReader(checkpoint.model_checkpoint_path)
            restore_dict = dict()
            for v in tf.all_variables():
                tensor_name = v.name.split(':')[0]
                if reader.has_tensor(tensor_name):
                    print('has tensor: {0}'.format(tensor_name))
                    restore_dict[tensor_name] = v

            checkpoint_saver = tf.train.Saver(restore_dict)
            checkpoint_saver.restore(sess, checkpoint.model_checkpoint_path)
            print(f'Model restored from {checkpoint.model_checkpoint_path}')
        else:
            print('Starting new session')
            
        return sess, saver
    
    def _calculate_rtt(self, attention_weights):
        """
        Calculate Related Text Tightness (RTT)
        Measures how well attention is concentrated on a few texts
        """
        rtt_values = []
        for weights in attention_weights:
            # Sort weights in descending order
            sorted_weights = np.sort(weights)[::-1]
            # Calculate cumulative sum
            cumsum = np.cumsum(sorted_weights)
            # Find position where cumulative sum exceeds 0.95 (95%)
            positions = np.where(cumsum >= 0.95)[0]
            if len(positions) > 0:
                # Get first position that exceeds threshold (+1 because positions are 0-indexed)
                key_text_count = positions[0] + 1
                rtt_values.append(key_text_count)
            else:
                # If no position found, all texts are needed
                rtt_values.append(len(weights))
        
        # Calculate proportion of samples where top 2 texts account for >95% attention
        top2_proportion = np.mean([1 if val <= 2 else 0 for val in rtt_values])
        return top2_proportion
    
    def evaluate_models(self, phase="test"):
        """
        Evaluate both models on the same data and compare their performance
        """
        print(f"Evaluating models on {phase} data")
        
        # Reset the TensorFlow graph between models
        tf.reset_default_graph()
        
        # ===== Evaluate original PEN model =====
        pen_model = self._initialize_original_pen()
        pen_sess, _ = self._initialize_session(pen_model)
        
        pen_results = self._evaluate_pen(pen_model, pen_sess, phase)
        
        # Close session to free resources
        pen_sess.close()
        
        # ===== Evaluate SEP-PEN integrated model =====
        tf.reset_default_graph()
        
        # For SEP-PEN, we use its built-in evaluation method
        sep_pen = self._initialize_sep_pen()
        sep_accuracy, sep_examples = sep_pen.evaluate(phase)
        
        # Extract SEP-PEN metrics from examples
        sep_results = self._extract_sep_metrics(sep_examples)
        
        # ===== Compare results =====
        comparison_results = self._compare_results(pen_results, sep_results)
        
        # Save results to file
        self._save_results(comparison_results, phase)
        
        # Generate visualizations
        self._generate_visualizations(comparison_results, phase)
        
        return comparison_results
        
    def _evaluate_pen(self, model, sess, phase):
        """Evaluate original PEN model and collect metrics"""
        # Initialize metrics
        all_losses = []
        all_predictions = []
        all_actuals = []
        all_attention_weights = []
        
        # Test data generator
        test_gen = self.pipe.batch_gen_by_stocks(phase)
        
        for batch_dict in tqdm(test_gen):
            feed_dict = self._create_feed_dict(model, batch_dict, is_training=False)
            
            # Forward pass with PEN model
            ops = [
                model.y_T,      # True labels
                model.y_T_,     # Predictions
                model.loss,     # Loss
                model.P,        # Attention weights
            ]
            
            actuals, predictions, batch_loss, attention_weights = sess.run(ops, feed_dict)
            
            # Collect metrics
            all_losses.append(batch_loss)
            all_predictions.extend(predictions)
            all_actuals.extend(actuals)
            all_attention_weights.extend(attention_weights)
        
        # Calculate accuracy
        accuracy = np.mean([1 if (pred > 0.5 and actual > 0.5) or (pred <= 0.5 and actual <= 0.5) 
                           else 0 for pred, actual in zip(all_predictions, all_actuals)])
        
        # Calculate RTT (Related Text Tightness)
        rtt = self._calculate_rtt(all_attention_weights)
        
        # For RoR and Kappa, we'd typically need human evaluations
        # For this comparison, we'll use default/placeholder values 
        # that should be replaced with real values if available
        ror = 0.893  # Default from PEN paper (89.3%)
        kappa = 0.591  # Default from PEN paper
        
        return {
            "model": "Original PEN",
            "loss": np.mean(all_losses),
            "accuracy": accuracy,
            "rtt": rtt,
            "ror": ror,
            "kappa": kappa
        }
    
    def _extract_sep_metrics(self, examples):
        """Extract metrics from SEP-PEN evaluation examples"""
        # Calculate accuracy from examples
        correct_count = sum(1 for ex in examples if ex["is_correct"])
        accuracy = correct_count / len(examples) if examples else 0
        
        # For other metrics, we'd need more data about the attention weights
        # Here, we'll use automated metrics rather than the traditional human ones
        
        return {
            "model": "SEP-PEN Integration",
            "loss": None,  # Not directly available from examples
            "accuracy": accuracy,
            "rtt": None,  # Could calculate if weights are included in examples
            "ror": None,  # Requires human evaluation or a system to simulate it
            "kappa": None  # Requires human evaluation or a system to simulate it
        }
    
    def _compare_results(self, pen_results, sep_results):
        """Compare results between the two models"""
        comparison = {
            "timestamp": datetime.now().isoformat(),
            "pen": pen_results,
            "sep_pen": sep_results,
            "differences": {}
        }
        
        # Calculate differences where possible
        for metric in ["loss", "accuracy", "rtt", "ror", "kappa"]:
            pen_val = pen_results.get(metric)
            sep_val = sep_results.get(metric)
            
            if pen_val is not None and sep_val is not None:
                comparison["differences"][metric] = sep_val - pen_val
                comparison["differences"][f"{metric}_percent"] = (
                    (sep_val - pen_val) / abs(pen_val) * 100 if pen_val != 0 else float('inf')
                )
        
        return comparison
    
    def _save_results(self, results, phase):
        """Save comparison results to file"""
        output_file = os.path.join(self.results_dir, f"{phase}_comparison.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")
    
    def _generate_visualizations(self, results, phase):
        """Generate visualizations for the comparison results"""
        # Create a directory for visualizations
        viz_dir = os.path.join(self.results_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Create comparison table
        metrics = ["accuracy", "loss", "rtt", "ror", "kappa"]
        available_metrics = [m for m in metrics if results["pen"].get(m) is not None and results["sep_pen"].get(m) is not None]
        
        if not available_metrics:
            print("No metrics available for visualization")
            return
            
        # Create a bar chart for comparing available metrics
        labels = [results["pen"]["model"], results["sep_pen"]["model"]]
        
        for metric in available_metrics:
            plt.figure(figsize=(10, 6))
            values = [results["pen"].get(metric, 0), results["sep_pen"].get(metric, 0)]
            
            bars = plt.bar(labels, values, color=['blue', 'orange'])
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.4f}', ha='center', va='bottom')
            
            plt.title(f'Comparison of {metric.upper()} between models')
            plt.ylabel(metric.upper())
            plt.ylim(0, max(values) * 1.2)  # Set y limit with some padding
            
            # Save the figure
            plt.savefig(os.path.join(viz_dir, f"{phase}_{metric}_comparison.png"))
            plt.close()
            
        print(f"Visualizations saved to {viz_dir}")

def main():
    parser = argparse.ArgumentParser(description='Compare PEN and SEP-PEN models')
    parser.add_argument('--phase', default='test', choices=['train', 'valid', 'test'],
                        help='Dataset phase to evaluate on (default: test)')
    parser.add_argument('--config', default='config.yml',
                        help='Path to config file (default: config.yml)')
    parser.add_argument('--api-key', default=None,
                        help='OpenAI API key (if using OpenAI)')
    parser.add_argument('--model', default='gpt-3.5-turbo',
                        help='LLM model to use (default: gpt-3.5-turbo)')
    
    args = parser.parse_args()
    
    # Set up OpenAI API key if provided
    if args.api_key:
        os.environ['OPENAI_API_KEY'] = args.api_key
    
    # Configure LLM
    llm_config = {
        "type": "openai" if args.api_key else "mock",
        "model": args.model,
        "max_tokens": 500,
        "temperature": 0.7
    }
    
    # Create comparison object
    comparison = ModelComparison(
        config_path=args.config,
        llm_config=llm_config
    )
    
    # Run evaluation
    results = comparison.evaluate_models(phase=args.phase)
    
    # Print summary
    print("\n===== COMPARISON SUMMARY =====")
    print(f"Original PEN accuracy: {results['pen']['accuracy']:.4f}")
    print(f"SEP-PEN accuracy: {results['sep_pen']['accuracy']:.4f}")
    if 'accuracy' in results['differences']:
        diff = results['differences']['accuracy']
        print(f"Difference: {diff:.4f} ({diff * 100:.2f}%)")
    
    if results['pen'].get('rtt') and results['sep_pen'].get('rtt'):
        print(f"\nOriginal PEN RTT: {results['pen']['rtt']:.4f}")
        print(f"SEP-PEN RTT: {results['sep_pen']['rtt']:.4f}")
    
    if results['pen'].get('ror') and results['sep_pen'].get('ror'):
        print(f"\nOriginal PEN RoR: {results['pen']['ror']:.4f}")
        print(f"SEP-PEN RoR: {results['sep_pen']['ror']:.4f}")
    
    if results['pen'].get('kappa') and results['sep_pen'].get('kappa'):
        print(f"\nOriginal PEN Kappa: {results['pen']['kappa']:.4f}")
        print(f"SEP-PEN Kappa: {results['sep_pen']['kappa']:.4f}")

if __name__ == "__main__":
    main() 