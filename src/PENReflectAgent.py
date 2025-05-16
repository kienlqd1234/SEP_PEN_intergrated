#!/usr/local/bin/python
import tensorflow as tf
import numpy as np

# Helper function moved here to avoid circular import
def get_vos_weights(model, sess, feed_dict):
    """
    Helper function to extract VoS weights from the PEN model
    
    Args:
        model: The PEN model instance
        sess: TensorFlow session
        feed_dict: Feed dictionary for the model
        
    Returns:
        numpy.ndarray: Vector of Salience weights
    """
    vos_weights = sess.run(model.vos_weights, feed_dict=feed_dict)
    return vos_weights

class PENReflectAgent:
    """
    Self-reflection agent for PEN that integrates SEP's reflection capability
    to improve explanations without human evaluation.
    """
    def __init__(self, ticker, texts, prices, target, reflection_threshold=0.7):
        self.ticker = ticker
        self.texts = texts
        self.prices = prices
        self.target = target
        self.prediction = None
        self.vos_weights = None
        self.top_texts = []
        self.explanation = ""
        self.reflections = []
        self.reflection_threshold = reflection_threshold
        self.finished = False
        
    def run_pen_model(self, model, sess, feed_dict):
        """Run the PEN model to get predictions and VoS weights"""
        y_pred, vos_weights, loss = sess.run(
            [model.y_T_, model.vos_weights, model.loss], 
            feed_dict=feed_dict
        )
        self.prediction = y_pred
        self.vos_weights = vos_weights
        return y_pred, vos_weights
    
    def extract_top_texts(self, n=2):
        """Extract top n texts based on VoS weights"""
        if self.vos_weights is None:
            return []
            
        # Get indices of top n weights
        top_indices = np.argsort(self.vos_weights.flatten())[-n:]
        
        # Extract the corresponding texts
        self.top_texts = [self.texts[i] for i in top_indices]
        return self.top_texts
    
    def generate_explanation(self, llm_model=None):
        """Generate explanation based on top texts"""
        if not self.top_texts:
            return "No explanation available"
            
        # If an LLM is provided, use it to generate an explanation
        if llm_model:
            prompt = f"Stock: {self.ticker}\n"
            prompt += f"Relevant information:\n"
            for i, text in enumerate(self.top_texts):
                prompt += f"- {text}\n"
            prompt += f"Based only on this information, explain why the stock price will likely "
            prompt += "increase" if self.prediction > 0.5 else "decrease"
            
            self.explanation = llm_model(prompt)
        else:
            # Simple rule-based explanation if no LLM available
            self.explanation = f"Stock {self.ticker} is predicted to "
            self.explanation += "increase" if self.prediction > 0.5 else "decrease"
            self.explanation += " based on: " + "; ".join(self.top_texts)
        
        return self.explanation
    
    def reflect(self, llm_model):
        """Generate reflection on the current explanation"""
        prompt = f"Stock: {self.ticker}\n"
        prompt += f"Prediction: {'increase' if self.prediction > 0.5 else 'decrease'}\n"
        prompt += f"Current explanation: {self.explanation}\n\n"
        prompt += "Please reflect on this explanation: Is it logical? Does it make sense given the information? What could be improved?"
        
        reflection = llm_model(prompt)
        self.reflections.append(reflection)
        return reflection
    
    def improve_explanation(self, llm_model):
        """Improve explanation based on reflections"""
        prompt = f"Stock: {self.ticker}\n"
        prompt += f"Relevant information:\n"
        for i, text in enumerate(self.top_texts):
            prompt += f"- {text}\n"
        
        prompt += f"Previous explanation: {self.explanation}\n\n"
        prompt += "Reflections on the explanation:\n"
        for i, reflection in enumerate(self.reflections):
            prompt += f"{i+1}. {reflection}\n"
        
        prompt += "\nPlease provide an improved explanation for the stock price movement prediction:"
        
        improved_explanation = llm_model(prompt)
        self.explanation = improved_explanation
        return improved_explanation
    
    def run_reflection_loop(self, llm_model, reward_model, max_iterations=3):
        """Run the full reflection loop to improve explanations"""
        current_score = 0
        
        for i in range(max_iterations):
            # Reflect on current explanation
            self.reflect(llm_model)
            
            # Improve explanation
            self.improve_explanation(llm_model)
            
            # Score the new explanation
            new_score = reward_model(self.explanation)
            
            # If score improvement is below threshold, stop
            if new_score - current_score < self.reflection_threshold and i > 0:
                break
                
            current_score = new_score
        
        self.finished = True
        return self.explanation
    
    def is_finished(self):
        return self.finished
        
    def is_correct(self):
        # Check if prediction matches target
        pred_class = 1 if self.prediction > 0.5 else 0
        return pred_class == self.target 