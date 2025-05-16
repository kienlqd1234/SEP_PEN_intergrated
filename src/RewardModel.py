#!/usr/local/bin/python
import numpy as np
import tensorflow as tf
import random

# Try importing transformer libraries, but make them optional
TRANSFORMERS_AVAILABLE = False
try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass

class SimpleRewardModel:
    """
    A simple rule-based reward model for explanations
    that doesn't require ML training.
    """
    def __init__(self):
        self.quality_terms = [
            "because", "due to", "result of", "leads to", "caused by",
            "consequently", "therefore", "thus", "indicates", "suggests",
            "implies", "correlates", "analysts", "investors", "market",
            "earnings", "revenue", "growth", "decline", "increase", "decrease"
        ]
        
    def __call__(self, explanation):
        """Score an explanation based on simple heuristics"""
        if not explanation:
            return 0.0
        
        score = 0.0
        
        # Length factor (not too short, not too long)
        words = explanation.split()
        length = len(words)
        if 20 <= length <= 150:
            score += 0.2
        elif 10 <= length < 20 or 150 < length <= 200:
            score += 0.1
            
        # Quality terms usage
        quality_count = sum(1 for term in self.quality_terms if term.lower() in explanation.lower())
        score += min(0.3, quality_count * 0.03)
        
        # Coherence (simple measure: no repetition of 3+ word phrases)
        three_grams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
        unique_three_grams = set(three_grams)
        if len(three_grams) > 0:
            coherence = len(unique_three_grams) / len(three_grams)
            score += coherence * 0.2
            
        # Specificity (mentions numbers/percentages/dates)
        if any(c.isdigit() for c in explanation):
            score += 0.15
            
        # Clarity (average word length not too high)
        avg_word_length = sum(len(word) for word in words) / max(1, len(words))
        if 3.5 <= avg_word_length <= 6.0:
            score += 0.15
            
        return min(1.0, score)


if TRANSFORMERS_AVAILABLE:
    class TransformerRewardModel:
        """
        A transformer-based reward model that can be trained
        to evaluate explanation quality.
        """
        def __init__(self, model_name="distilbert-base-uncased"):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=1
            ).to(self.device)
            self.model.eval()  # Set to evaluation mode
            
        def train(self, good_explanations, bad_explanations, epochs=3, batch_size=8):
            """Train the reward model using good and bad explanation examples"""
            self.model.train()
            
            # Prepare dataset
            texts = good_explanations + bad_explanations
            labels = ([1.0] * len(good_explanations)) + ([0.0] * len(bad_explanations))
            
            # Simple training loop
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
            
            # Split into batches
            n_samples = len(texts)
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            
            for epoch in range(epochs):
                np.random.shuffle(indices)
                total_loss = 0
                
                for i in range(0, n_samples, batch_size):
                    batch_indices = indices[i:i + batch_size]
                    batch_texts = [texts[idx] for idx in batch_indices]
                    batch_labels = torch.tensor([labels[idx] for idx in batch_indices], 
                                               device=self.device).float().view(-1, 1)
                    
                    inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, 
                                           truncation=True, max_length=512).to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = self.model(**inputs)
                    loss = torch.nn.MSELoss()(outputs.logits, batch_labels)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / (n_samples // batch_size)}")
                
            self.model.eval()
            print("Training completed!")
            
        def save(self, path):
            """Save the model and tokenizer"""
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            
        def load(self, path):
            """Load the model and tokenizer"""
            self.model = AutoModelForSequenceClassification.from_pretrained(path).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.model.eval()
            
        def __call__(self, explanation):
            """Score an explanation using the trained model"""
            with torch.no_grad():
                inputs = self.tokenizer(explanation, return_tensors="pt", padding=True, 
                                      truncation=True, max_length=512).to(self.device)
                outputs = self.model(**inputs)
                score = outputs.logits.item()
                
                # Normalize score to [0, 1]
                score = max(0.0, min(1.0, score))
                return score
else:
    # Fallback implementation when transformers not available
    class TransformerRewardModel:
        """Fallback implementation when transformers package is not available"""
        def __init__(self, model_name=None):
            print("WARNING: Using fallback TransformerRewardModel because transformers package is not available")
            self.simple_model = SimpleRewardModel()
            
        def train(self, good_explanations, bad_explanations, epochs=3, batch_size=8):
            print("Cannot train without transformers package. Using rule-based scoring instead.")
            pass
            
        def save(self, path):
            print(f"Cannot save model without transformers package. Path: {path} ignored.")
            pass
            
        def load(self, path):
            print(f"Cannot load model without transformers package. Path: {path} ignored.")
            pass
            
        def __call__(self, explanation):
            # Fall back to the simple reward model
            return self.simple_model(explanation) 


def create_explanation_pairs(agents, reflection_iterations=1):
    """
    Create pairs of explanations for training the reward model.
    Each pair consists of an original explanation and an improved one after reflection.
    
    Args:
        agents: List of PENReflectAgent objects
        reflection_iterations: Number of reflection iterations to perform
        
    Returns:
        List of (original_explanation, improved_explanation) tuples
    """
    explanation_pairs = []
    
    for agent in agents:
        if agent.explanation:
            original = agent.explanation
            
            # Store reflections and explanations at each iteration
            explanations = [original]
            
            for i in range(reflection_iterations):
                # Simulate reflection and improvement
                if hasattr(agent, 'reflect') and hasattr(agent, 'improve_explanation'):
                    agent.reflect(lambda x: "This explanation could be improved with more specific details.")
                    improved = agent.improve_explanation(lambda x: f"Improved explanation #{i+1}: {original} with additional details.")
                    explanations.append(improved)
            
            # Create pairs between original and each improved version
            for i in range(1, len(explanations)):
                explanation_pairs.append((explanations[0], explanations[i]))
    
    return explanation_pairs 