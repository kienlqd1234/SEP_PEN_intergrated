# code
This is the source code and some evaluation scripts for our paper [Causal and Dual-Path Enhanced Prediction-Explanation Network for Interpretable Stock Price Forecasting]().
Our code is based on https://github.com/Shuqi-li/PEN

## 🧠 Cognitive Causal Dual-Stream for Financial Prediction

This project implements a **Cognitive Causal Dual-Stream (CDP-PEN)** framework for stock movement prediction using financial text (news, social media) and market data.

Unlike traditional approaches that rely on correlation, this model focuses on **causal reasoning** by separating information into two distinct learning pathways:

- 🔵 **Causal Pathway (text → price):** captures how news impacts market movements  
- 🔴 **Responsive Pathway (price → text):** models how market trends influence narratives  

To further enhance performance, the framework introduces:

- ✂️ **Causal Text Selection (Ca-TSU):** filters out noisy or irrelevant news, keeping only causally meaningful signals  
- 🔀 **Dual-Path Learning (DP-SRL):** jointly learns bidirectional interactions between text and price  
- 🌪️ **Volatility-Aware Fusion (VAF):** dynamically adapts model behavior under different market conditions  

---

## 🎯 Key Objectives

- Improve prediction accuracy for stock/index movement  
- Reduce noise from financial text data  
- Provide **interpretable, causally grounded insights**  
- Build a more **robust model under market volatility**

---


## 💡 Key Idea

> Financial markets are driven not just by *what is said*, but by *what truly causes price changes*.  
> This project bridges that gap by explicitly modeling **causality + interaction + market dynamics**.

## Dependencies
- Python 3.6.11
- Tensorflow 1.4.0
- Scipy 1.0.0
- NLTK 3.2.5


## Directories
- src: source files;
    - The core code of our model is in `MSINModule_caTSU.py` and `MSINModule_dual_path.py`
- res: resource files including,
    - Vocabulary file `vocab.txt`;
    - Pre-trained embeddings of [GloVe](https://github.com/stanfordnlp/GloVe). We used the GloVe obtained from the Twitter corpora which you could download [here](http://nlp.stanford.edu/data/wordvecs/glove.twitter.27B.zip).
- data:
    - ACL18 consisting of tweets and prices which you could download [here](https://github.com/yumoxu/stocknet-dataset).
    - DJIA consisting of news and prices which you could download [here](https://www.kaggle.com/datasets/aaron7sun/stocknews).


## Configurations
All details about hyper-parameters are listed in `src/config_tx_lf.yml` and `src/config_tx_lf_dual_path.yml`. 

See more information in 'Experimental Setup' of our paper.

## Running
Use `python src/Main_tx_lf.py`/ `python src/Main_tx_lf_dual_path.py` in your terminal to start model training and testing. 

The default code corresponds to ACL18.
For DJIA, simply replace `Executor` to `Executor_d` in `src/Main.py`.

## Result
## 📊 Results

<p align="center">
  <img src="https://img.shields.io/badge/Model-CDP--PEN-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/ACL18-58.54%25_ACC-success?style=for-the-badge" />
  <img src="https://img.shields.io/badge/DJIA-59.31%25_ACC-success?style=for-the-badge" />
</p>

We evaluate the **Cognitive Causal Dual-Stream (CDP-PEN)** framework on:

- 📈 **ACL18** (stock-level prediction with Twitter data)  
- 📰 **DJIA** (index-level prediction with news headlines)

**Metrics:**
- Accuracy (ACC)
- Matthews Correlation Coefficient (MCC)

---

## 🔢 Quantitative Results

| Model | ACL18 ACC (%) | ACL18 MCC | DJIA ACC (%) | DJIA MCC |
|------|--------------|----------|--------------|----------|
| Random | 48.76 | -0.002 | 49.22 | 0.0003 |
| RF | 50.04 | 0.010 | 51.33 | 0.050 |
| HAN | 54.33 | 0.052 | 51.34 | 0.059 |
| StockNet | 54.55 | 0.080 | 52.91 | 0.129 |
| CPC | 54.61 | 0.179 | - | - |
| PEN | 54.99 | 0.153 | 55.53 | 0.220 |
| Ca-TSU | 57.44 | 0.162 | 58.14 | 0.235 |
| DP-SRL | 58.28 | 0.155 | 58.89 | 0.227 |
| **CDP-PEN** | **58.54** | **0.170** | **59.31** | **0.220** |


All information and results are detailed in the report file `BaoCao.pdf`
