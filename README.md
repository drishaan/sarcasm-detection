# Sarcasm Across Text Mediums
Over the years, there have been many theories crafted about how to best detect the presence of sarcasm in text. Indeed, this task is complex as the limited range of expression of text combined with the ambiguity that characterizes sarcasm makes text analysis for even humans difficult. It is this set of challenges that we take on with our language models.

Our project consists of gathering a variety of models to investigate how well they can detect sarcasm across different text mediums as a proxy to judge how difficult it may be for a reader to detect sarcasm. Our secondary objective looks into generative models as another avenue of investigation. 

## Data Processing and Exploration
Run the `DataProcessingHeadlines.ipynb` and `DataProcessingTweets.ipynb` Jupyter notebook.

The resulting cleaned text data will be in `data/cleaned_headlines.csv` and `data/tweets_headlines.csv`.

The text based columns for the tweet will be the following columns:
- `headline` or `tweet`: the raw text
- `is_sarcastic`: denotes as sarcastic (1) or not sarcastic (0)
- `no_stopwords`: remove stopwords
- `tokenized`: tokenizes text
- `tokenized_no_stopwords`: tokenizes text, then removes stopwords

## Text Classification
We ran Naive Bayes and Logistic Regression models as low power text classifiers as well as LSTM for high power classification. In the following files, we train models on both the headline and tweet data (separately) and print out the result metrics for different input parameters. To run the models, run
- `NaiveBayes.ipynb`
- `LogisticRegression.ipynb`
- `LSTM_Headlines.ipynb`
- `LSTM_Tweets.ipynb`

## Generative Model
We implemented a Markov Chain to generate sarcastic and non-sarcastic headlines and tweets. The user can select what type of text to generate and the number of desired outputs. To generate, run `MarkovChain.ipynb`
