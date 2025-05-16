# SEP-PEN Integration

This project integrates the Prediction-Explanation Network (PEN) with the Summarize-Explain-Predict (SEP) framework to create an enhanced stock prediction model with automated explanation generation and evaluation.

## Overview

The integration addresses a key limitation of PEN: its dependency on human experts to evaluate explanation quality through metrics like RoR, RTT, and Kappa. By incorporating SEP's self-reflection, automated evaluation, and reinforcement learning components, the integrated system can:

1. Generate explanations for stock price movement predictions using PEN's Vector of Salience (VoS)
2. Improve explanations through self-reflection without human intervention
3. Train a reward model to automatically evaluate explanation quality
4. Use multi-shot sampling to select the best explanations
5. Optimize the explanation generation through PPO

## Components

### Core Integration Files

- `SEP_Integration.py`: Main integration class that combines PEN and SEP
- `PENReflectAgent.py`: Agent class that implements self-reflection for PEN explanations
- `RewardModel.py`: Implements automated evaluation of explanations
- `LLMInterface.py`: Connects to LLMs (OpenAI API or local models) for explanation generation
- `PPOTrainer.py`: Optimizes explanation generation using reinforcement learning
- `run_sep_pen.py`: Script to run the integrated model

### Original PEN Files

- `Model.py`: Original PEN model implementation
- `MSINModule.py`: Market State Information Network module
- `VoSExplainModule.py`: Vector of Salience explanation module

### SEP Components

- `sep_main/`: Original SEP implementation directory

## Setup

### Requirements

- Python 3.7+
- TensorFlow 1.14
- PyTorch 1.9+ (for transformer models)
- transformers library
- OpenAI API key (optional, for using GPT models)

### Installation

```bash
pip install -r requirements.txt
```

If you want to use OpenAI models, set your API key:

```bash
export OPENAI_API_KEY=your_api_key_here
```

## Usage

### Training

To train the integrated model:

```bash
python src/run_sep_pen.py --mode train --epochs 5 --llm_type mock
```

For training with OpenAI:

```bash
python src/run_sep_pen.py --mode train --llm_type openai --model_name gpt-3.5-turbo --api_key your_api_key
```

### Evaluation

To evaluate the model:

```bash
python src/run_sep_pen.py --mode eval --eval_phase test
```

### Full Pipeline

To run both training and evaluation:

```bash
python src/run_sep_pen.py --mode full
```

## How It Works

1. **Initial Prediction**: PEN model predicts stock movement and identifies relevant texts via VoS
2. **Explanation Generation**: LLM generates explanations based on the top texts
3. **Self-Reflection**: For incorrect predictions, system reflects on explanations
4. **Explanation Improvement**: LLM improves explanations based on reflections
5. **Reward Model Training**: System trains on pairs of original and improved explanations
6. **Multi-shot Sampling**: During evaluation, multiple explanations are generated and the best is selected
7. **PPO Optimization**: LLM is fine-tuned to generate better explanations

## Key Benefits Over Original PEN

- **No Human Evaluation**: Eliminates need for expert evaluation (RoR, RTT, Kappa)
- **Automated Improvement**: Self-reflection improves explanations automatically
- **Better Explanations**: Multi-shot sampling and PPO optimization improve quality
- **Scalability**: Can process large datasets without human bottleneck

## Results Directory Structure

Results are saved to `results/sep_pen_integration/` with the following structure:

- `epoch_{n}/`: Results from each training epoch
  - `examples.json`: Example predictions and explanations
  - `stats.json`: Epoch statistics
- `{phase}_evaluation/`: Evaluation results
  - `examples.json`: Example predictions and explanations
  - `stats.json`: Evaluation statistics
- `reward_model/`: Saved reward model
- `run_config.json`: Configuration for the run

## References

- PEN: Li et al., "PEN: Prediction-Explanation Network to Forecast Stock Price Movement with Better Explainability"
- SEP: Koa et al., "Learning to Generate Explainable Stock Predictions using Self-Reflective Large Language Models" 