#!/usr/local/bin/python
import os
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any, Callable, Optional, Tuple
import random

# Try importing transformer libraries, but make them optional
TRANSFORMERS_AVAILABLE = False
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass

if TRANSFORMERS_AVAILABLE:
    class PPOTrainer:
        """
        PPO Trainer to optimize LLM for generating high-quality explanations
        without human feedback.
        """
        def __init__(
            self,
            model,
            tokenizer,
            reward_model,
            learning_rate=1e-5,
            clip_param=0.2,
            value_loss_coef=0.1,
            entropy_coef=0.01,
            max_grad_norm=0.5,
            batch_size=4,
            epochs=4,
            device=None
        ):
            self.model = model
            self.tokenizer = tokenizer
            self.reward_model = reward_model
            
            # PPO parameters
            self.learning_rate = learning_rate
            self.clip_param = clip_param
            self.value_loss_coef = value_loss_coef
            self.entropy_coef = entropy_coef
            self.max_grad_norm = max_grad_norm
            self.batch_size = batch_size
            self.epochs = epochs
            
            # Set device
            self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            
            # Setup optimizer
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
            
        def _get_explanation(self, prompt: str, temperature=0.7, max_new_tokens=200) -> str:
            """Generate an explanation using the model"""
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generate with sampling
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Decode and remove the prompt
            full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            explanation = full_text[len(self.tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)):].strip()
            
            return explanation
        
        def _get_log_probs(self, prompt: str, response: str) -> torch.Tensor:
            """
            Calculate token-wise log probabilities for a response given a prompt
            """
            # Tokenize prompt and full sequence
            prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            full_ids = self.tokenizer.encode(prompt + response, return_tensors="pt").to(self.device)
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(full_ids)
                logits = outputs.logits
            
            # Get log probs for response tokens only
            response_logits = logits[0, prompt_ids.shape[1]-1:-1, :]
            response_ids = full_ids[0, prompt_ids.shape[1]:]
            
            # Calculate token-wise log probabilities
            log_probs = torch.log_softmax(response_logits, dim=-1)
            token_log_probs = torch.gather(log_probs, 1, response_ids.unsqueeze(1)).squeeze(1)
            
            return token_log_probs
        
        def _calculate_advantages(self, rewards, values, gamma=0.99, lambda_=0.95):
            """Calculate advantages using Generalized Advantage Estimation (GAE)"""
            advantages = []
            last_advantage = 0
            
            for i in reversed(range(len(rewards))):
                if i == len(rewards) - 1:
                    # For the last step, use reward only
                    delta = rewards[i] - values[i]
                else:
                    # For all other steps, include next state value
                    delta = rewards[i] + gamma * values[i+1] - values[i]
                
                advantage = delta + gamma * lambda_ * last_advantage
                advantages.insert(0, advantage)
                last_advantage = advantage
                
            return torch.tensor(advantages, device=self.device)
        
        def train_step(self, prompts: List[str], max_explanations_per_prompt=3):
            """Run a single training step"""
            all_prompts = []
            all_explanations = []
            all_log_probs = []
            all_rewards = []
            all_values = []
            
            # Generate multiple explanations per prompt
            for prompt in tqdm(prompts, desc="Generating explanations"):
                for _ in range(max_explanations_per_prompt):
                    # Generate explanation
                    explanation = self._get_explanation(prompt)
                    
                    # Calculate initial log probs
                    log_probs = self._get_log_probs(prompt, explanation)
                    
                    # Get reward from reward model
                    reward = self.reward_model(explanation)
                    
                    # Calculate value (this is simplified; normally would come from a value head)
                    # Here just using reward as value for simplicity
                    value = reward
                    
                    # Store all data
                    all_prompts.append(prompt)
                    all_explanations.append(explanation)
                    all_log_probs.append(log_probs)
                    all_rewards.append(reward)
                    all_values.append(value)
            
            # Calculate advantages
            advantages = self._calculate_advantages(all_rewards, all_values)
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # PPO epochs
            for _ in range(self.epochs):
                # Process in batches
                for i in range(0, len(all_prompts), self.batch_size):
                    batch_prompts = all_prompts[i:i+self.batch_size]
                    batch_explanations = all_explanations[i:i+self.batch_size]
                    batch_old_log_probs = all_log_probs[i:i+self.batch_size]
                    batch_advantages = advantages[i:i+self.batch_size]
                    
                    # Calculate new log probs
                    batch_new_log_probs = []
                    for prompt, explanation in zip(batch_prompts, batch_explanations):
                        new_log_probs = self._get_log_probs(prompt, explanation)
                        batch_new_log_probs.append(new_log_probs)
                    
                    # Calculate ratio and clipped ratio
                    ratios = []
                    for old_log_probs, new_log_probs in zip(batch_old_log_probs, batch_new_log_probs):
                        ratio = torch.exp(torch.sum(new_log_probs) - torch.sum(old_log_probs))
                        ratios.append(ratio)
                    
                    ratios = torch.stack(ratios)
                    
                    # Clipped PPO objective
                    surr1 = ratios * batch_advantages
                    surr2 = torch.clamp(ratios, 1.0 - self.clip_param, 1.0 + self.clip_param) * batch_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # Value loss (using rewards as target)
                    # In a real implementation, you'd compare value network outputs to actual returns
                    value_loss = self.value_loss_coef * torch.tensor(0.0, device=self.device)
                    
                    # Entropy bonus (encourage exploration)
                    # Simplified; normally would calculate from action distribution
                    entropy = self.entropy_coef * torch.tensor(0.0, device=self.device)
                    
                    # Total loss
                    loss = policy_loss + value_loss - entropy
                    
                    # Optimize
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
            
            # Return average reward for monitoring
            return sum(all_rewards) / len(all_rewards)
        
        def train(self, prompts: List[str], n_epochs=10, prompts_per_epoch=None):
            """Train the model for multiple epochs"""
            prompts_per_epoch = prompts_per_epoch or len(prompts)
            
            rewards = []
            
            for epoch in range(n_epochs):
                # Sample prompts for this epoch
                if len(prompts) > prompts_per_epoch:
                    epoch_prompts = random.sample(prompts, prompts_per_epoch)
                else:
                    epoch_prompts = prompts
                
                # Run training step
                avg_reward = self.train_step(epoch_prompts)
                rewards.append(avg_reward)
                
                print(f"Epoch {epoch+1}/{n_epochs}, Average Reward: {avg_reward:.4f}")
            
            return rewards
        
        def save(self, path):
            """Save the model"""
            os.makedirs(path, exist_ok=True)
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            
        def generate_explanation(self, prompt, n_samples=5):
            """
            Generate multiple explanations and return the one with highest reward
            """
            best_explanation = ""
            best_reward = -float("inf")
            
            for _ in range(n_samples):
                explanation = self._get_explanation(prompt)
                reward = self.reward_model(explanation)
                
                if reward > best_reward:
                    best_reward = reward
                    best_explanation = explanation
            
            return best_explanation, best_reward
else:
    # Fallback implementation when PyTorch/transformers aren't available
    class PPOTrainer:
        """
        Mock PPO Trainer that provides basic functionality without PyTorch/transformers.
        This allows the rest of the code to run without these dependencies.
        """
        def __init__(
            self,
            model=None,
            tokenizer=None,
            reward_model=None,
            learning_rate=1e-5,
            clip_param=0.2,
            value_loss_coef=0.1,
            entropy_coef=0.01,
            max_grad_norm=0.5,
            batch_size=4,
            epochs=4,
            device=None
        ):
            print("WARNING: Using mock PPOTrainer because transformers/PyTorch are not available")
            self.reward_model = reward_model
            self.mock_explanations = [
                "The stock price is likely to increase due to better than expected quarterly earnings.",
                "Based on recent product announcements, the company's stock is expected to rise.",
                "Due to market uncertainty and declining revenues, the stock price may decrease.",
                "The stock performance is likely to improve following the new partnership announcement.",
                "Considering the negative industry trends, the stock price may drop in the near term."
            ]
            
        def train_step(self, prompts: List[str], max_explanations_per_prompt=3):
            """Mock training step that simply prints progress"""
            print(f"Mock PPO training on {len(prompts)} prompts")
            return 0.75  # Return fake average reward
            
        def train(self, prompts: List[str], n_epochs=10, prompts_per_epoch=None):
            """Mock training function"""
            rewards = []
            for epoch in range(n_epochs):
                reward = self.train_step(prompts[:10])  # Only use first 10 prompts for speed
                rewards.append(reward)
                print(f"Epoch {epoch+1}/{n_epochs}, Mock Average Reward: {reward:.4f}")
            return rewards
            
        def save(self, path):
            """Mock save function"""
            os.makedirs(path, exist_ok=True)
            print(f"Mock PPO model would be saved to {path}")
            
        def generate_explanation(self, prompt, n_samples=5):
            """
            Generate a mock explanation
            """
            # Choose a random explanation from our templates
            explanations = []
            for _ in range(n_samples):
                explanation = random.choice(self.mock_explanations)
                # Customize based on prompt content
                if "increase" in prompt.lower():
                    explanation = explanation.replace("decrease", "increase")
                elif "decrease" in prompt.lower():
                    explanation = explanation.replace("increase", "decrease")
                explanations.append(explanation)
                
            # If we have a reward model, use it to pick the best one
            if self.reward_model:
                best_explanation = ""
                best_reward = -float("inf")
                
                for explanation in explanations:
                    reward = self.reward_model(explanation)
                    if reward > best_reward:
                        best_reward = reward
                        best_explanation = explanation
                
                return best_explanation, best_reward
            else:
                # Just return the first one with a mock score
                return explanations[0], 0.8 