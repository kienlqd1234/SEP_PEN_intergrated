# PEN: Prediction-Explanation Network to Forecast Stock Price Movement with Better Explainability

**Authors:** Shuqi Li, Weiheng Liao, Yuhan Chen, Rui Yan  
**Affiliations:**
- Gaoling School of Artificial Intelligence (GSAI), Renmin University of China
- MADE by DATA
- Engineering Research Center of Next-Generation Intelligent Search and Recommendation, Ministry of Education

**Contact:** shuqili@ruc.edu.cn, weiheng@madebydata.com, yuhanchen@ruc.edu.cn, ruiyan@ruc.edu.cn

## Abstract
Stock price movement prediction needs better explainability due to auditing and regulatory requirements. PEN (Prediction-Explanation Network) jointly models text streams and price streams via alignment and shared representation learning. It identifies relevant text messages that influence stock price changes and explains predictions effectively. PEN outperforms state-of-the-art models in both accuracy and interpretability.

## Introduction
Traditional models in stock prediction (e.g., time series analysis) are valued for simplicity and interpretability. Deep learning has gained popularity due to higher accuracy with large datasets. Texts from news and social media are increasingly used in prediction, but not all texts are equally useful. PEN is proposed to identify and use only the most relevant texts for better predictions and explanations.

### Contributions
- Proposes a novel framework combining price and text data.
- Develops a Shared Representation Learning (SRL) module.
- Introduces a salient regulator to focus on important texts.

## Related Work
### Stock Price Prediction
Traditional methods use price and volume data. Recent models integrate textual information like news and tweets. Hybrid models (e.g., StockNet) combine price and text data. PEN improves upon these by learning better representations.

### Explainability
Previous work uses attention mechanisms for interpretability. PEN uses SRL to generate a Vector of Salience (VoS), making explanations more precise.

## Prediction-Explanation Network (PEN)
### Problem Definition
Classifies next-day price movement (`1` for up, `0` for down) using:
- Historical price data `P[t−L, t−1]`
- Text corpora `C`

### Architecture Components
1. **Text Embedding Layer (TEL)**: Bi-directional GRU encodes daily texts.
2. **Shared Representation Learning (SRL)**:
   - **Text Selection Unit (TSU)**: Weights text importance.
   - **Text Memory Unit (TMU)**: Preserves useful historical text.
   - **Information Fusion Unit (IFU)**: Fuses text and price data.
3. **Deep Recurrent Generation (DRG)**: Variational auto-encoder for latent representation.
4. **Temporal Attention Prediction (TAP)**: Uses attention on time dimension for final prediction.

## Learning Objective
Loss function includes:
- Prediction loss (`L1`): Combines cross-entropy and KL divergence.
- Explainability loss (`L2`): KL divergence between VoS and uniform distribution.

Overall loss: `L = L1 + L2`

## Experiments
### Datasets
- **ACL18**: Tweets + stock prices (2014–2016)
- **DJIA**: Reddit headlines + Dow Jones index (2008–2016)

### Metrics
- Accuracy (ACC)
- Matthews Correlation Coefficient (MCC)

### Results
- PEN outperforms all baselines (e.g., StockNet, HAN).
- Best accuracy: 59.9% (ACL18), 60.5% (DJIA)

### Explainability
- **RTT**: Top 2 texts account for >95% attention weights in 99.5% of samples.
- **RoR**: 89.3% of PEN’s top picks matched human expert judgment.
- **Fleiss' Kappa**: 0.591 agreement with human annotators.

### Qualitative Examples
PEN correctly identifies most relevant news in specific dates for both Apple stock and DJIA.

## Ablation Studies
- Removing SRL, TAP, DRG, or KL-loss reduces performance.
- Price components: High/Low prices are more informative than adjusted close.

## Conclusion
PEN introduces a novel explainable stock prediction model by aligning and jointly representing text and price data. It achieves state-of-the-art performance and strong interpretability using SRL and VoS.

## References
*(Included at the end of original paper)*
