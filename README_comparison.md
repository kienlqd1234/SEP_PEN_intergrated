# SEP-PEN vs PEN Model Comparison

This tool allows you to compare the performance of the SEP-PEN integrated model against the original PEN model on stock prediction tasks.

## Metrics Compared

- **Loss**: Training/validation loss values
- **Accuracy**: Prediction accuracy on stock movement directions
- **RTT (Related Text Tightness)**: How well the model concentrates attention on relevant texts
- **RoR (Rate of Relevance)**: Agreement with human expert judgment
- **Kappa (Fleiss' Kappa)**: Agreement with human annotators

## Files

- `run_and_compare.py`: Main Python script that runs both models and compares their performance
- `run_comparison.sh`: Shell script wrapper for easier execution

## Requirements

- Python 3.6+
- TensorFlow 1.x
- NumPy
- Pandas
- Matplotlib
- OpenAI API key (optional, for using OpenAI models)

## Usage

### Using the Shell Script

The simplest way to run the comparison is using the shell script:

```bash
./run_comparison.sh
```

Or with custom parameters:

```bash
./run_comparison.sh --phase test --config src/config.yml --api-key YOUR_API_KEY --model gpt-3.5-turbo
```

Make the script executable first:

```bash
chmod +x run_comparison.sh
```

### Using Python Directly

You can also run the Python script directly:

```bash
python run_and_compare.py --phase test --config src/config.yml --api-key YOUR_API_KEY --model gpt-3.5-turbo
```

### Command-line Arguments

- `--phase`: Dataset phase to use (train, valid, test). Default: test
- `--config`: Path to configuration file. Default: src/config.yml
- `--api-key`: OpenAI API key (if using OpenAI models)
- `--model`: LLM model to use. Default: gpt-3.5-turbo

## Output

The tool generates:

1. JSON files with detailed performance metrics in `results/comparison/`
2. Visualizations comparing metrics in `results/comparison/visualizations/`
3. Console output with a summary of key comparison metrics

## Implementation Details

The comparison tool:

1. Loads both models with the same configuration
2. Evaluates them on the same dataset
3. Collects metrics like accuracy, RTT, RoR, and Kappa
4. Saves comparative results and generates visualizations
5. Integrates SEP's self-reflection capabilities with PEN's Vector of Salience (VoS) approach

## Key Features

- Direct head-to-head comparison of model performance
- Automatic visualization of comparative metrics
- Support for OpenAI API-based LLM models
- Detailed logging of results for analysis

## Notes on Explainability Metrics

While accuracy can be automatically calculated, the explainability metrics (RTT, RoR, Kappa) have some limitations:

- **RTT**: Calculated automatically based on attention weight distribution
- **RoR and Kappa**: Traditionally require human evaluation, the tool uses default values from the PEN paper for comparison with SEP-PEN's automated metrics 