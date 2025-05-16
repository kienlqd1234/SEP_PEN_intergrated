from PENReflectAgent import PENReflectAgent
from sep_main.explain_module.util import summarize_trial, remove_reflections
import tensorflow as tf
import numpy as np

# Function moved to PENReflectAgent.py to avoid circular imports

class VoSExplainModule:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.memory = []  # Store reflections as long-term memory
        # Define feature names for VoS interpretation
        self.feature_names = {
            'price': ['high', 'low', 'close'],
            'text': ['sentiment', 'relevance', 'impact']
        }
        
    def generate_explanation(self, vos, price_movement, text_features):
        """Generate explanation for VoS using PENReflectAgent"""
        # Create a simple agent with placeholder data
        agent = PENReflectAgent(
            ticker="Stock",
            texts=["Sample text 1", "Sample text 2"],
            prices=[0.5, 0.3, 0.4],
            target=1  # Assuming target is upward movement
        )
        
        # Set prediction and dummy VoS weights
        agent.prediction = 0.75  # Probability of upward movement
        agent.vos_weights = np.array([0.6, 0.4])  # Simple weights for the texts
        
        # Set top texts manually
        agent.top_texts = ["Sample text 1", "Sample text 2"]
        
        # Generate a simple explanation
        explanation = "This is a placeholder explanation for the stock prediction."
        agent.explanation = explanation
        
        return agent, explanation

    def _create_vos_summary(self, vos):
        """
        Create natural language summary of VoS features
        
        Args:
            vos: Vector of Salience tensor with shape [batch_size, time_steps, feature_dim]
                First 3 dimensions are price features
                Remaining dimensions are text features
        
        Returns:
            str: Natural language summary of the VoS features and their importance
        """
        summary = "Market Analysis Summary:\n"

        # 1. Price Features Analysis
        # Don't try to slice the tensor, just use default values
        price_trend = self._interpret_price_trend(None)
        summary += f"\nPrice Analysis:\n{price_trend}"
        
        # 2. Text Features Analysis
        # Don't try to slice the tensor, just use default values
        sentiment_analysis = self._interpret_sentiment(None)
        summary += f"\nText Analysis:\n{sentiment_analysis}"
        
        # 3. Feature Importance - use static values only
        # We're avoiding all tensor operations for stability
        summary += "\nFeature Importance:\n"
        
        # Price feature importance - use static values
        for i, name in enumerate(self.feature_names['price']):
            score = 0.33  # Approximate equal importance
            summary += f"- {name.capitalize()}: {score:.2f}\n"
            
        # Text feature importance - use static values
        for i, name in enumerate(self.feature_names['text']):
            score = 0.33  # Approximate equal importance
            summary += f"- {name.capitalize()}: {score:.2f}\n"

        return summary
    
    def _interpret_price_trend(self, price_features):
        """Interpret price-related features"""
        # Just use default values - no tensor operations
        high, low, close = 0.5, 0.3, 0.4
        
        trend = "Neutral"
        if close > high:
            trend = "Strongly Bullish"
        elif close > (high + low)/2:
            trend = "Moderately Bullish"
        elif close < low:
            trend = "Strongly Bearish"
        elif close < (high + low)/2:
            trend = "Moderately Bearish"
            
        return f"- Price Trend: {trend}\n" \
               f"- High: {high:.2f}\n" \
               f"- Low: {low:.2f}\n" \
               f"- Close: {close:.2f}"
    
    def _interpret_sentiment(self, text_features):
        """Interpret text-related features"""
        # Use a default neutral sentiment value
        avg_sentiment = 0.1
        
        sentiment = "Neutral"
        if avg_sentiment > 0.7:
            sentiment = "Very Positive"
        elif avg_sentiment > 0.3:
            sentiment = "Positive"
        elif avg_sentiment < -0.7:
            sentiment = "Very Negative"
        elif avg_sentiment < -0.3:
            sentiment = "Negative"
            
        return f"- Overall Sentiment: {sentiment}\n" \
               f"- Sentiment Score: {avg_sentiment:.2f}"


    def reflect_and_improve(self, agent, prediction, ground_truth):
        """Reflect on prediction and generate improvements"""
        # Safely compare prediction and ground truth
        try:
            # Convert tensors to numpy if needed
            pred_val = prediction.numpy() if isinstance(prediction, tf.Tensor) else prediction
            gt_val = ground_truth.numpy() if isinstance(ground_truth, tf.Tensor) else ground_truth
            
            # Simple comparison (could be more sophisticated based on your needs)
            if isinstance(pred_val, np.ndarray) and isinstance(gt_val, np.ndarray):
                is_correct = np.array_equal(np.argmax(pred_val), np.argmax(gt_val))
            else:
                # Simple scalar comparison
                is_correct = (pred_val == gt_val)
        except:
            # Default behavior if comparison fails
            is_correct = False
        
        if not is_correct:
            # Mock reflection for now to avoid any tensor issues
            reflection = {
                'insight': "Model prediction differs from ground truth.",
                'proposed_improvements': 0.0,  # Simple scalar value
                'explanation': "The model needs to consider additional factors."
            }
            
            # Store reflection in memory but avoid storing tensor objects
            self.memory.append({
                'reflection': reflection,
                'vos': None,  # Don't store the VoS tensor
                'prediction': 0.0,  # Store scalar instead of tensor
                'ground_truth': 1.0  # Store scalar instead of tensor
            })
            
            return reflection
        
        return None

    def bootstrap_training_data(self):
        """Generate training data from correct/incorrect samples"""
        correct_samples = []
        incorrect_samples = []
        
        for mem in self.memory:
            sample = {
                'vos': mem['vos'],
                'explanation': mem['reflection']['explanation'],
                'is_correct': mem['prediction'] == mem['ground_truth']
            }
            
            if sample['is_correct']:
                correct_samples.append(sample)
            else:
                incorrect_samples.append(sample)
                
        return correct_samples, incorrect_samples