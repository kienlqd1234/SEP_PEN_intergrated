#!/usr/local/bin/python
import os
import openai
import time
import random
from typing import Optional, Dict, Any, List

class OpenAILLM:
    """Simple interface to OpenAI API"""
    
    def __init__(self, model="gpt-3.5-turbo", max_tokens=500, temperature=0.7, api_key=None):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Set API key from env var if not provided
        if api_key:
            openai.api_key = api_key
        elif "OPENAI_API_KEY" in os.environ:
            openai.api_key = os.environ["OPENAI_API_KEY"]
        else:
            raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")
    
    def __call__(self, prompt: str) -> str:
        """Call OpenAI API and return response"""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )
                return response.choices[0].message.content.strip()
            
            except Exception as e:
                print(f"Error calling OpenAI API: {e}")
                if attempt < max_retries - 1:
                    sleep_time = retry_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"Retrying in {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)
                else:
                    print("Failed to get response from OpenAI API after multiple attempts")
                    return "Error: Failed to generate response"


class MockLLM:
    """Mock LLM for testing without API calls"""
    
    def __init__(self):
        self.responses = {
            "reflection": [
                "This explanation lacks specific details about why the price movement is expected.",
                "The explanation doesn't connect the news to the predicted outcome clearly.",
                "The explanation could be improved by adding more context about the industry trends."
            ],
            "improvement": [
                "Based on the information, the stock price is likely to {direction} because the earnings report exceeded analyst expectations, showing strong revenue growth of 15% year-over-year.",
                "The stock will probably {direction} as the company announced a new partnership with a major industry player, which analysts see as a strategic advantage.",
                "Given the information about recent product launches and positive market reception, the stock is expected to {direction} in the short term."
            ]
        }
    
    def __call__(self, prompt: str) -> str:
        """Return mock response based on prompt content"""
        time.sleep(0.5)  # Simulate API delay
        
        if "reflect" in prompt.lower():
            return random.choice(self.responses["reflection"])
        
        if "improve" in prompt.lower() or "provide an improved explanation" in prompt.lower():
            direction = "increase" if "increase" in prompt.lower() else "decrease"
            response = random.choice(self.responses["improvement"])
            return response.format(direction=direction)
        
        # Default response
        return "The stock price is likely to change based on the provided information."


class TransformerLLM:
    """Interface to local transformer models"""
    
    def __init__(self, model, tokenizer, max_tokens=500, temperature=0.7):
        self.model = model
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.temperature = temperature
    
    def __call__(self, prompt: str) -> str:
        """Generate response using local transformer model"""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=0.95,
                do_sample=True
            )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part, not the prompt
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
                
            return response
        
        except Exception as e:
            print(f"Error generating response: {e}")
            return "Error: Failed to generate response"


# Factory function to create appropriate LLM based on configuration
def create_llm(config: Dict[str, Any]) -> Any:
    """Create LLM instance based on configuration"""
    llm_type = config.get("type", "mock")
    
    if llm_type == "openai":
        return OpenAILLM(
            model=config.get("model", "gpt-3.5-turbo"),
            max_tokens=config.get("max_tokens", 500),
            temperature=config.get("temperature", 0.7),
            api_key=config.get("api_key")
        )
    
    elif llm_type == "transformer":
        # This assumes model and tokenizer are passed in the config
        return TransformerLLM(
            model=config.get("model"),
            tokenizer=config.get("tokenizer"),
            max_tokens=config.get("max_tokens", 500),
            temperature=config.get("temperature", 0.7)
        )
    
    else:  # Default to mock for testing
        return MockLLM() 