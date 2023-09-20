# NLP Amazon Reviews

Data was collected from this <a href="https://www.kaggle.com/datasets/bittlingmayer/amazonreviews">Amazon reviews dataset</a>, which contains Amazon products' reviews (in text), and their corresponding star-rating.

<strong>'bert' notebook</strong> - Predicts the star-rating with BERT model after performing the tokenization with 'bert-base-uncased' model tokenizer.  

<strong>'lstm-lda-topic-modelling' notebook</strong> - Predicts the star-rating with a bi-directional LSTM and performs topic-modelling with Latent Dirichlet Allocation (LDA). With enough computing power, it would be possible to expand this notebook to associate reviews' topics with sentiment and star-ratings, in order for the seller to prioritize the points of improvement in the product.  
For example, within the collection of reviews of a given smartphone, 'low battery autonomy' could be associated to a very negative sentiment and 1-star reviews, while 'camera quality' could be associated with a slightly negative sentiment and 3-star reviews, which would indicate that the battery autonomy is the major fault to address.
