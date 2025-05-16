# Running SEP-PEN with OpenAI API

This guide explains how to run the SEP-PEN integration using OpenAI API without requiring PyTorch or transformers.

## Prerequisites

1. Python 3.6 (required by the PEN model)
2. TensorFlow 1.4.0
3. OpenAI API key

## Installation

1. Create a Python 3.6 environment:
   ```
   conda create -n sep_pen_openai python=3.6
   conda activate sep_pen_openai
   ```

2. Install the required dependencies:
   ```
   pip install -r src/requirements.txt
   ```

## Running the Integration

Use the `Main.py` script to run the integration with OpenAI:

```
python src/Main.py --openai_api_key YOUR_OPENAI_API_KEY --model gpt-3.5-turbo --mode eval
```

### Command-line Arguments

- `--openai_api_key`: Your OpenAI API key (required)
- `--model`: OpenAI model to use (default: gpt-3.5-turbo)
- `--mode`: Mode to run (choices: train, eval, full; default: eval)
- `--epochs`: Number of epochs for training (default: 3)
- `--output_dir`: Directory to save results (default: results/openai_run)

## How It Works

This implementation:

1. Uses OpenAI's API for the language model parts of SEP without requiring PyTorch or transformers
2. Uses the SimpleRewardModel instead of TransformerRewardModel for explanation evaluation
3. Uses the PEN model's Vector of Salience (VoS) for text importance scoring
4. Applies SEP's self-reflection approach to improve explanations using OpenAI

## Example Usage

### Evaluation Mode

To run only the evaluation phase:

```
python src/Main.py --openai_api_key YOUR_OPENAI_API_KEY --mode eval
```

### Training Mode

To run only the training phase:

```
python src/Main.py --openai_api_key YOUR_OPENAI_API_KEY --mode train --epochs 5
```

### Full Mode

To run both training and evaluation:

```
python src/Main.py --openai_api_key YOUR_OPENAI_API_KEY --mode full
```

## Results

Results are saved to the specified output directory (default: results/openai_run), including:
- Prediction accuracy
- Example predictions with explanations
- Reflection improvements 