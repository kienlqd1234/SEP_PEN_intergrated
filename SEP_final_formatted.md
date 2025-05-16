# Learning to Generate Explainable Stock Predictions using Self-Reflective Large Language Models

**Authors:** Kelvin J.L. Koa, Yunshan Ma*, Ritchie Ng, Tat-Seng Chua  
**Affiliations:** National University of Singapore; Eastspring Investments  
**Contact:** kelvin.koa@u.nus.edu, yunshan.ma@u.nus.edu, ritchie.ng@eastspring.com, dcscts@nus.edu.sg  

## Abstract
Traditional deep learning models struggle to provide explainable stock predictions. LLMs can generate natural language explanations but face challenges in reasoning with chaotic social texts. The proposed **Summarize-Explain-Predict (SEP)** framework uses self-reflection and Proximal Policy Optimization (PPO) to train LLMs to generate accurate and explainable stock predictions autonomously. SEP surpasses deep learning and existing LLM-based approaches in both prediction accuracy and explanation quality.

## 1. Introduction
- Stock prediction is challenging due to the vast and noisy nature of social data.
- Traditional models lack interpretability.
- LLMs show promise but need fine-tuning to handle explanation generation and text reasoning.
- SEP is introduced to autonomously train LLMs using their own self-reflections.

## 2. Related Works
- **Text analysis in stock prediction:** From shallow (BoW, Noun Phrases) to deep (VAEs, Transformers).
- **LLMs in finance:** BloombergGPT, FinGPT, etc. SEP differs by using self-reflection and PPO.

## 3. Methodology
### 3.1 Task Definition
Given T days of social text `C_s`, predict the binary stock movement `ŷ_s` and generate an explanation `ê_s`.

### 3.2 SEP Framework
- **Summarize:** LLMs summarize raw social texts into factual point-form data.
- **Explain:** Generates prediction and explanation, then improves through a **self-reflective loop**.
- **Predict:** Uses PPO to fine-tune a policy model on self-labeled training samples.

### 3.3 Fine-Tuning Process
- **Step 1:** Supervised fine-tuning with initial correct samples.
- **Step 2:** Reward model trained with reflection-based good vs bad samples.
- **Step 3:** PPO optimizes model for high-reward predictions.

### 3.4 Inference
- Input texts → summarized → multiple outputs sampled → highest reward selected.

## 4. Experiments
### Research Questions:
- **RQ1:** How does SEP perform vs baselines?
- **RQ2:** Contribution of each SEP component?
- **RQ3:** Can SEP generalize to portfolio construction?

### Baselines
- Deep learning: VAE+Att, GRU+Att, Transformer
- LLMs: GPT-3.5, Vicuna, FinGPT

### Metrics:
- Accuracy, Matthews Correlation Coefficient (MCC)

### Results:
- **SEP outperforms all baselines**.
- GPT-based SEP: best accuracy 54.35%, MCC 0.0993

### Explanation Quality:
- SEP explanations rated higher than GPT/Vicuna by GPT-4 on multiple criteria (clarity, relevance, etc.).

## 5. Ablation Studies
### Summarize Module:
- Using summarized facts significantly improves MCC.

### Explain Module:
- Self-reflection increases correct and decisive predictions over iterations.

### Predict Module:
- PPO fine-tuning and n-shot sampling improve performance.
- SEP with all components shows best results.

## 6. Portfolio Optimization
- SEP fine-tuned to assign portfolio weights from explanation text.
- Outperforms 1/N, S&P500, Positive-only, GPT-3.5, and Vicuna baselines.
- **SEP achieves best Sharpe Ratio (1.150)** and cumulative returns.

## 7. Conclusion
SEP introduces a self-reflective LLM framework for explainable stock prediction. It eliminates human annotation via self-training, improves interpretability, and generalizes to financial tasks like portfolio construction. Future work includes enhancing robustness, multi-modal input integration, and better evaluation metrics.

## Ethical Considerations
- Risks: manipulation, misinformation, LLM bias.
- Mitigations: human-in-the-loop, input/output validation, restricted access.

## Acknowledgments
Supported by the National Research Foundation, Singapore. Opinions are those of the authors.

---

**Keywords:** Stock Prediction, LLM, Explainability, Reinforcement Learning, Self-Reflection, PPO  
**Source:** ACM WWW’24 Conference  
**License:** Creative Commons Attribution 4.0  
GitHub: [https://github.com/koa-fin/sep](https://github.com/koa-fin/sep)
