## Stock Price Prediction Project

### Overview

This project aims to predict stock price movements based on the sentiment derived from financial news and forum discussions. By leveraging the power of natural language processing (NLP), specifically the term frequency-inverse document frequency (tf-idf) method, we extract relevant keywords that indicate Bullish (positive) or Bearish (negative) sentiment.

### Methodology

The core of our approach involves constructing a term set through tf-idf to abstract significant keywords from our corpus of financial texts. These keywords serve as features for our predictive models, which are then trained to forecast stock market trends.

### Prediction and Accuracy

We utilize these text-derived features to predict whether a stock's price will rise or fall. Our models have demonstrated remarkable proficiency, accurately capturing market trends with an 85% accuracy rate.

### Conclusion

The high accuracy rate underscores the efficacy of using tf-idf for keyword abstraction in financial sentiment analysis and its potency in predicting stock market movements.

## Module Structure

```plaintext
root/
  |──src/
  |   |── __init__.py
  |   |── main.py
  |   |── preproc.py
  |   └── predict.py  
  |── .gitignore
  |── README.md
  └── requirements.txt
```
