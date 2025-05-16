#!/usr/local/bin/python
import os
import tensorflow as tf
import numpy as np
import sys
import json
from datetime import datetime
from tqdm import tqdm

# Import PEN components
from DataPipe import DataPipe
from SEP_Integrated_Model import Model
from metrics import eval_res, n_accurate

# Import new SEP integration components
from PENReflectAgent import PENReflectAgent
from RewardModel import SimpleRewardModel, TransformerRewardModel, create_explanation_pairs
from LLMInterface import create_llm, MockLLM
from PPOTrainer import PPOTrainer

class SEP_PEN_Integration:
    """
    Integration class that combines PEN's Vector of Salience (VoS) explainability
    with SEP's self-reflection and automated evaluation approach.
    """
    
    def __init__(self, config_path="src/config.yml", llm_config=None, device=None):
        self.config_path = config_path
        
        # Default LLM config if none provided
        if llm_config is None:
            self.llm_config = {
                "type": "mock",  # Can be "mock", "openai", "transformer"
                "model": "gpt-3.5-turbo",
                "max_tokens": 500,
                "temperature": 0.7
            }
        else:
            self.llm_config = llm_config
            
        # Initialize components
        self.model = Model(config_path)
        self.model.assemble_graph()  # Assemble the TensorFlow graph
        self.pipe = DataPipe()
        self.llm = create_llm(self.llm_config)
        self.reward_model = SimpleRewardModel()  # Start with simple reward model
        
        # Create TF saver and session config
        self.saver = tf.train.Saver()
        self.tf_config = tf.ConfigProto(allow_soft_placement=True)
        self.tf_config.gpu_options.allow_growth = True
        
        # Tracking for training
        self.agents = []
        self.train_stats = {
            'epochs': 0,
            'batches': 0,
            'correct_predictions': 0,
            'reflection_improvements': 0,
            'avg_rewards': []
        }
        
        # Results storage
        self.results_dir = "results/sep_pen_integration"
        os.makedirs(self.results_dir, exist_ok=True)
        
    def _initialize_session(self):
        """Initialize TensorFlow session"""
        sess = tf.Session(config=self.tf_config)
        
        # Initialize word table
        word_table_init = self.pipe.init_word_table()
        feed_table_init = {self.model.word_table_init: word_table_init}
        sess.run(tf.global_variables_initializer(), feed_dict=feed_table_init)
        
        # Try to restore checkpoint if available
        checkpoint = tf.train.get_checkpoint_state(os.path.dirname(self.model.tf_checkpoint_file_path))
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
            
        return sess
    
    def _create_feed_dict(self, batch_dict, is_training=True):
        """Create feed dict for model from batch dict"""
        return {
            self.model.is_training_phase: is_training,
            self.model.batch_size: batch_dict['batch_size'],
            self.model.stock_ph: batch_dict['stock_batch'],
            self.model.T_ph: batch_dict['T_batch'],
            self.model.n_words_ph: batch_dict['n_words_batch'],
            self.model.n_msgs_ph: batch_dict['n_msgs_batch'],
            self.model.y_ph: batch_dict['y_batch'],
            self.model.price_ph: batch_dict['price_batch'],
            self.model.mv_percent_ph: batch_dict['mv_percent_batch'],
            self.model.word_ph: batch_dict['word_batch'],
            self.model.ss_index_ph: batch_dict['ss_index_batch'],
            # Set dropout to 0 when not training
            self.model.dropout_mel_in: 0.0 if not is_training else None,
            self.model.dropout_mel: 0.0 if not is_training else None,
            self.model.dropout_ce: 0.0 if not is_training else None,
            self.model.dropout_vmd_in: 0.0 if not is_training else None,
            self.model.dropout_vmd: 0.0 if not is_training else None,
        }
    
    def _extract_texts_from_batch(self, batch_dict):
        """Extract all texts from a batch for explanation"""
        # This is a simplified placeholder - the actual implementation
        # would depend on how texts are stored in your batches
        texts = []
        for i in range(batch_dict['batch_size']):
            sample_texts = []
            for day in range(min(5, self.model.max_n_days)):  # Limit to 5 days for simplicity
                for msg in range(min(10, self.model.max_n_msgs)):  # Limit to 10 messages
                    if batch_dict['n_words_batch'][i][day][msg] > 0:
                        # Create dummy text for now - in real implementation, 
                        # you'd extract actual text content
                        sample_texts.append(f"Text from day {day}, message {msg}")
            texts.append(sample_texts)
        return texts
    
    def _create_agents_from_batch(self, batch_dict, predictions, actuals, attention_weights):
        """Create PENReflectAgents from batch data"""
        agents = []
        texts_list = self._extract_texts_from_batch(batch_dict)
        
        for i in range(batch_dict['batch_size']):
            ticker = f"Stock_{batch_dict['stock_batch'][i]}"
            
            # Safely get target value from actuals
            try:
                target_val = actuals[i][0]
                if isinstance(target_val, np.ndarray) or isinstance(target_val, tf.Tensor):
                    # Try to convert tensor to scalar
                    try:
                        if hasattr(target_val, 'numpy'):
                            target_val = float(target_val.numpy())
                        else:
                            target_val = float(target_val)
                    except:
                        target_val = 0.5  # Default value
                target = 1 if target_val > 0.5 else 0
            except:
                target = 0  # Default to negative class
            
            # Safely get price data
            try:
                prices = batch_dict['price_batch'][i]
                if isinstance(prices, tf.Tensor) and hasattr(prices, 'numpy'):
                    prices = prices.numpy()
            except:
                prices = np.array([0.5, 0.3, 0.4])  # Default prices
            
            agent = PENReflectAgent(
                ticker=ticker,
                texts=texts_list[i],
                prices=prices,
                target=target
            )
            
            # Safely set prediction
            try:
                pred_val = predictions[i][0]
                if isinstance(pred_val, np.ndarray) or isinstance(pred_val, tf.Tensor):
                    try:
                        if hasattr(pred_val, 'numpy'):
                            pred_val = float(pred_val.numpy())
                        else:
                            pred_val = float(pred_val)
                    except:
                        pred_val = 0.5  # Default value
                agent.prediction = pred_val
            except:
                agent.prediction = 0.5  # Default to uncertain prediction
            
            # Safely set attention weights
            try:
                weights = attention_weights[i]
                if isinstance(weights, tf.Tensor) and hasattr(weights, 'numpy'):
                    weights = weights.numpy()
                agent.vos_weights = weights
            except:
                # Create dummy weights if needed
                agent.vos_weights = np.ones(len(texts_list[i])) / len(texts_list[i])
            
            # Extract top texts based on attention weights
            try:
                agent.extract_top_texts(n=2)
            except:
                # If extraction fails, manually set top texts
                if texts_list[i]:
                    agent.top_texts = texts_list[i][:min(2, len(texts_list[i]))]
                else:
                    agent.top_texts = ["No relevant texts found"]
            
            agents.append(agent)
            
        return agents
    
    def train(self, n_epochs=10):
        """Train model using SEP's self-reflection approach"""
        # Create TensorFlow session without using 'with'
        sess = self._initialize_session()
        
        try:
            # Start from current global step
            # Properly pass session to eval
            start_epoch = sess.run(self.model.global_step) // 100  # Assuming 100 steps per epoch
            
            for epoch in range(start_epoch, start_epoch + n_epochs):
                print(f"Epoch {epoch+1}/{start_epoch + n_epochs}")
                
                # Training batch generator
                train_batch_gen = self.pipe.batch_gen(phase='train')
                
                epoch_agents = []
                epoch_correct = 0
                epoch_total = 0
                
                # Process each batch
                for batch_idx, train_batch_dict in enumerate(train_batch_gen):
                    feed_dict = self._create_feed_dict(train_batch_dict, is_training=True)
                    
                    # Forward pass with PEN model
                    ops = [
                        self.model.y_T,       # True labels
                        self.model.y_T_,      # Predictions
                        self.model.loss,      # Loss
                        self.model.P,         # Attention weights from MSIN
                    ]
                    
                    actuals, predictions, batch_loss, vos_weights = sess.run(ops, feed_dict)
                    
                    # Create agents for this batch
                    batch_agents = self._create_agents_from_batch(
                        train_batch_dict, predictions, actuals, vos_weights
                    )
                    
                    # Generate initial explanations
                    for agent in batch_agents:
                        agent.generate_explanation(self.llm)
                    
                    # Run self-reflection loop for incorrectly predicted samples
                    for agent in batch_agents:
                        if not agent.is_correct():
                            agent.run_reflection_loop(
                                llm_model=self.llm,
                                reward_model=self.reward_model,
                                max_iterations=3
                            )
                    
                    # Collect statistics
                    correct_agents = [a for a in batch_agents if a.is_correct()]
                    epoch_correct += len(correct_agents)
                    epoch_total += len(batch_agents)
                    epoch_agents.extend(batch_agents)
                    
                    # Every 10 batches, print stats
                    if batch_idx % 10 == 0:
                        print(f"  Batch {batch_idx}, Accuracy: {len(correct_agents)}/{len(batch_agents)}")
                        
                    # Every 50 batches, save model
                    if batch_idx % 50 == 0 and batch_idx > 0:
                        save_path = self.saver.save(
                            sess, 
                            self.model.tf_saver_path, 
                            global_step=self.model.global_step
                        )
                        print(f"  Model saved: {save_path}")
                
                # After each epoch, collect explanation pairs for reward model training
                print("Collecting explanation pairs for reward model training...")
                explanation_pairs = create_explanation_pairs(epoch_agents, reflection_iterations=1)
                
                if explanation_pairs:
                    # Train reward model if we have enough data
                    if len(explanation_pairs) >= 20:  # Arbitrary threshold
                        print(f"Training reward model on {len(explanation_pairs)} explanation pairs")
                        self._train_reward_model(explanation_pairs)
                
                # Save epoch results
                self._save_epoch_results(epoch, epoch_agents)
                
                # Print epoch stats
                print(f"Epoch {epoch+1} complete. Accuracy: {epoch_correct}/{epoch_total} = {epoch_correct/max(1, epoch_total):.4f}")
                self.train_stats['epochs'] += 1
                
                # Save model after each epoch
                save_path = self.saver.save(
                    sess, 
                    self.model.tf_saver_path, 
                    global_step=self.model.global_step
                )
                print(f"Model saved: {save_path}")
        finally:
            # Close session when done
            sess.close()
    
    def _train_reward_model(self, explanation_pairs):
        """Train reward model on explanation pairs"""
        if not explanation_pairs:
            return
            
        # Extract good and bad explanations
        good_explanations = [pair[1] for pair in explanation_pairs]  # improved versions
        bad_explanations = [pair[0] for pair in explanation_pairs]   # original versions
        
        # Initialize transformer reward model if we have enough data
        if len(explanation_pairs) >= 50 and isinstance(self.reward_model, SimpleRewardModel):
            print("Upgrading to transformer reward model")
            self.reward_model = TransformerRewardModel()
            
        # Train the model
        if isinstance(self.reward_model, TransformerRewardModel):
            self.reward_model.train(
                good_explanations=good_explanations,
                bad_explanations=bad_explanations,
                epochs=3
            )
            
            # Save the trained model
            self.reward_model.save(os.path.join(self.results_dir, "reward_model"))
    
    def _save_epoch_results(self, epoch, agents):
        """Save results from an epoch"""
        epoch_dir = os.path.join(self.results_dir, f"epoch_{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)
        
        # Save agent examples
        examples = []
        for i, agent in enumerate(agents[:10]):  # Save first 10 for brevity
            example = {
                "ticker": agent.ticker,
                "prediction": float(agent.prediction),
                "target": int(agent.target),
                "top_texts": agent.top_texts,
                "explanation": agent.explanation,
                "reflections": agent.reflections,
                "is_correct": agent.is_correct()
            }
            examples.append(example)
        
        with open(os.path.join(epoch_dir, "examples.json"), 'w') as f:
            json.dump(examples, f, indent=2)
        
        # Save statistics
        correct_agents = [a for a in agents if a.is_correct()]
        stats = {
            "epoch": epoch,
            "total_samples": len(agents),
            "correct_predictions": len(correct_agents),
            "accuracy": len(correct_agents) / max(1, len(agents)),
            "timestamp": datetime.now().isoformat()
        }
        
        # Save automated metrics (replacing human evaluation metrics)
        if hasattr(self.model, 'automated_metrics'):
            metrics = {}
            for key, tensor in self.model.automated_metrics.items():
                if isinstance(tensor, tf.Tensor):
                    # Create a new session instead of using with
                    sess = self._initialize_session()
                    try:
                        metrics[key] = float(sess.run(tensor))
                    finally:
                        sess.close()
                else:
                    metrics[key] = tensor
            stats["automated_metrics"] = metrics
        
        with open(os.path.join(epoch_dir, "stats.json"), 'w') as f:
            json.dump(stats, f, indent=2)
            
    def optimize_llm_with_ppo(self, n_iterations=5):
        """
        Optimize the LLM for generating high-quality explanations using PPO
        based on the reward model trained during the regular training process.
        
        Args:
            n_iterations: Number of PPO optimization iterations
        """
        print("Optimizing LLM with PPO for explanation generation...")
        
        # Skip if LLM is not optimizable or reward model is not available
        if self.llm_config["type"] == "mock" or not hasattr(self, "reward_model"):
            print("Skipping PPO optimization: requires a real LLM and reward model")
            return
            
        # Create PPO trainer
        ppo_trainer = PPOTrainer(
            model=self.llm.model if hasattr(self.llm, "model") else None,
            tokenizer=self.llm.tokenizer if hasattr(self.llm, "tokenizer") else None,
            reward_model=self.reward_model
        )
        
        # Collect prompts from previous examples
        prompts = []
        for root, dirs, files in os.walk(self.results_dir):
            for file in files:
                if file == "examples.json":
                    with open(os.path.join(root, file), 'r') as f:
                        examples = json.load(f)
                        for example in examples:
                            ticker = example.get("ticker", "Stock")
                            texts = example.get("top_texts", [])
                            
                            if texts:
                                prompt = f"Stock: {ticker}\nRelevant information:\n"
                                for i, text in enumerate(texts):
                                    prompt += f"- {text}\n"
                                prompt += "Based only on this information, explain the stock price movement."
                                prompts.append(prompt)
        
        if not prompts:
            print("No prompts found for PPO training. Skipping.")
            return
            
        # Train with PPO
        unique_prompts = list(set(prompts))  # Remove duplicates
        print(f"Training with {len(unique_prompts)} unique prompts")
        
        ppo_trainer.train(unique_prompts, n_epochs=n_iterations)
        
        # Save optimized model
        ppo_save_path = os.path.join(self.results_dir, "ppo_optimized_model")
        ppo_trainer.save(ppo_save_path)
        print(f"PPO-optimized model saved to {ppo_save_path}")
        
        # Update the LLM to use the optimized model if possible
        if hasattr(self.llm, "update_model") and callable(self.llm.update_model):
            self.llm.update_model(ppo_save_path)
            print("LLM updated to use PPO-optimized model")
    
    def evaluate(self, phase="test"):
        """
        Evaluate model on test set using the SEP approach of self-reflection
        and multi-shot sampling for best explanation generation
        """
        # Create TensorFlow session without using 'with'
        sess = self._initialize_session()
        
        try:
            # Test data generator
            test_gen = self.pipe.batch_gen_by_stocks(phase)
            
            all_agents = []
            total_correct = 0
            total_samples = 0
            
            print(f"Evaluating on {phase} set")
            
            for batch_idx, test_batch_dict in enumerate(tqdm(test_gen)):
                feed_dict = self._create_feed_dict(test_batch_dict, is_training=False)
                
                # Forward pass with PEN model
                ops = [
                    self.model.y_T,       # True labels
                    self.model.y_T_,      # Predictions
                    self.model.loss,      # Loss
                    self.model.P,         # Use P instead of vos_weights for consistency
                ]
                
                actuals, predictions, batch_loss, vos_weights = sess.run(ops, feed_dict)
                
                # Create agents for this batch
                batch_agents = self._create_agents_from_batch(
                    test_batch_dict, predictions, actuals, vos_weights
                )
                
                # For each agent, generate multiple explanations and pick the best one
                for agent in batch_agents:
                    # Generate multiple explanations and select the best one
                    # using the reward model for scoring
                    
                    best_explanation = ""
                    best_score = -float("inf")
                    
                    for _ in range(5):  # Generate 5 explanations
                        explanation = agent.generate_explanation(self.llm)
                        score = self.reward_model(explanation)
                        
                        if score > best_score:
                            best_score = score
                            best_explanation = explanation
                    
                    agent.explanation = best_explanation
                
                # Collect statistics
                total_correct += sum(1 for a in batch_agents if a.is_correct())
                total_samples += len(batch_agents)
                all_agents.extend(batch_agents)
            
            # Calculate final metrics
            accuracy = total_correct / max(1, total_samples)
            
            # Save evaluation results
            eval_dir = os.path.join(self.results_dir, f"{phase}_evaluation")
            os.makedirs(eval_dir, exist_ok=True)
            
            # Save examples
            examples = []
            for i, agent in enumerate(all_agents[:50]):  # Save first 50 for analysis
                example = {
                    "ticker": agent.ticker,
                    "prediction": float(agent.prediction),
                    "target": int(agent.target),
                    "top_texts": agent.top_texts,
                    "explanation": agent.explanation,
                    "is_correct": agent.is_correct()
                }
                examples.append(example)
            
            with open(os.path.join(eval_dir, "examples.json"), 'w') as f:
                json.dump(examples, f, indent=2)
            
            # Save statistics
            stats = {
                "phase": phase,
                "total_samples": total_samples,
                "correct_predictions": total_correct,
                "accuracy": accuracy,
                "timestamp": datetime.now().isoformat()
            }
            
            with open(os.path.join(eval_dir, "stats.json"), 'w') as f:
                json.dump(stats, f, indent=2)
            
            print(f"Evaluation on {phase} complete. Accuracy: {total_correct}/{total_samples} = {accuracy:.4f}")
            return accuracy, examples
        finally:
            # Close session when done
            sess.close()


if __name__ == "__main__":
    # Simple test script
    integration = SEP_PEN_Integration()
    
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        integration.train(n_epochs=5)
    else:
        integration.evaluate(phase="test") 