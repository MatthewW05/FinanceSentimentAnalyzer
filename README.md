# Finance Sentiment Analyzer

This project aims to predict the sentiment (positive or negative) from financial news headlines using machine learning techniques.

## Installation

Install `FinanceSentimentAnalyzer` using `pip`:

``` {.sourceCode .bash}
pip install git+https://github.com/MatthewW05/FinanceSentimentAnalyzer.git
```

[Required dependencies](./requirements.txt) , [All dependencies](./setup.py).


## Usage

### Predicting a headline's sentiment using the pre-trained model

```python
from FinanceSentimentAnalyzer import predict_headline_sentiment

headline = "Nvidia Stock Rises. How Earnings From Microsoft and Apple Could Drive It Higher."

# by default a pre-trained model is pre-loaded when imported
# the function will return float between 0 and 1
prediction = predict_headline_sentiment(headline)
prediction = "Positive" if round(prediction) == 1 else "Negative"
   
print(f"Prediction for \'{headline}\': {prediction}")
```

### Loading your own model
When loading your own model, you must include both the model and and the used vocabulary

```python
from FinanceSentimentAnalyzer import load_model_and_vocab, predict_headline_sentiment

headline = "Nvidia Stock Rises. How Earnings From Microsoft and Apple Could Drive It Higher."
model, vocab = load_model_and_vocab('path/to/your/model', 'path/to/your/vocab')
prediction = predict_headline_sentiment(headline, model, vocab)
prediction = "Positive" if round(prediction) == 1 else "Negative"

print(f"Prediction for \'{headline}\': {prediction}")
```


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.


## Data Sources

This project uses data from several third-party sources. We acknowledge and thank the creators of these datasets:

- **[Sentiment Analysis for Financial News](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news)** - Ankur Sinha
- **[Financial Sentiment Analysis](https://www.kaggle.com/datasets/sbhatti/financial-sentiment-analysis)** - sbhatti
- **[Sentiment Analysis - Labelled Financial News Data](https://www.kaggle.com/datasets/aravsood7/sentiment-analysis-labelled-financial-news-data)** - Arav Sood7
- **[financial_phrasebank](https://huggingface.co/datasets/takala/financial_phrasebank)** - Ankur Sinha
- **[Aspect based Sentiment Analysis for Financial News](https://www.kaggle.com/datasets/ankurzing/aspect-based-sentiment-analysis-for-financial-news)** - Ankur Sinha


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

**Matthew Wong**
