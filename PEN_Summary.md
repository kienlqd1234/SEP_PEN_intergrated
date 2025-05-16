# PEN: Prediction-Explanation Network to Forecast Stock Price Movement with Better Explainability

## Main Idea

PEN (Prediction-Explanation Network) is a neural network architecture designed to forecast stock price movements with enhanced explainability. The model integrates both numerical market data and textual information (such as tweets or news) to make predictions while providing explanations for its decisions through an attention mechanism.

## Model Architecture and Features

### Overview
PEN consists of several key components:

1. **Message Embedding Layer (MEL)**: Processes textual information (tweets/news articles)
   - Uses bidirectional GRU/LSTM to encode textual information
   - Converts words to word embeddings using GloVe pre-trained embeddings

2. **Market State Information Network (MSIN)**: A novel RNN-based architecture
   - Integrates textual and numerical market data
   - Contains several gates to process and combine different information sources
   - Uses attention mechanism to focus on relevant messages

3. **Variational Message Decoder (VMD)**: A generative model
   - Allows the model to generate explanations for its predictions
   - Based on a variational recurrent neural network architecture
   - Uses latent variables to capture hidden patterns

4. **Attention-based Temporal Aggregator (ATA)**: Assigns weights to different time steps
   - Helps identify important trading days for the final prediction
   - Provides temporal explanations

### Distinctive Features

- **Dual-purpose architecture**: Both predicts stock movements and explains its decisions
- **Hierarchical attention mechanism**: At both message level and temporal level
- **Variational approach**: Incorporates uncertainty into the prediction process
- **Multi-source fusion**: Effectively combines numerical (price) and textual (social media/news) information

## Experimental Results

The model was evaluated on two datasets:

1. **ACL18 dataset**: Stock tweets and prices
2. **DJIA dataset**: News headlines and prices

Performance metrics include:
- Accuracy
- Matthews Correlation Coefficient (MCC)
- F1 Score

The model demonstrates strong performance compared to baseline methods and state-of-the-art approaches:
- Better prediction accuracy than traditional methods and many deep learning models
- More interpretable predictions through attention visualization
- Ability to identify key messages and time periods that influence predictions
- Robust performance across different market sectors and conditions

## Conclusion

PEN successfully addresses two critical challenges in stock movement prediction:

1. **Performance**: By integrating multiple information sources (market data and text) through a sophisticated architecture, PEN achieves competitive prediction accuracy.

2. **Explainability**: More importantly, PEN provides interpretable explanations for its predictions, making it more trustworthy for decision-making in financial contexts. The model can highlight:
   - Which messages (tweets/news) had the most impact on the prediction
   - Which trading days were most significant
   - How different information sources were weighted

The architecture represents a step forward in developing financial prediction systems that are both accurate and transparent, addressing the "black box" problem that limits the adoption of many machine learning approaches in critical domains like finance.

The attention mechanisms in PEN not only improve prediction accuracy but also enable users to understand the reasoning behind specific forecasts, which is crucial for building trust in algorithmic trading systems. 