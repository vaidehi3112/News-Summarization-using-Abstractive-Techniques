# News-Summarization-using-Abstractive-Techniques
The following code has been trained on Cnn/Daily-Mail dataset.

Link to Dataset: https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail

### Seq2Seq RNN with Attention Mechanism
Deep learning models that take a sequence of items as input and produce another sequence of items as output are known as sequence-to-sequence (Seq2Seq) models. Utilizing Long Short-Term Memory (LSTM) networks, which are a type of recurrent neural network, capable of learning long-term dependencies, is particularly effective in applications where sequences are interdependent, such as text summarization.

The drawback of LSTM model is that, sometimes it predicts the wrong sequence due to ambiguity. Hence, we have combined the "attention" mechanism along with the existing model. It is based on this exact concept of directing the focus on important factors while predicting the output in Sequence to Sequence models. This way, it helps the encoder in searching most relevant information. 

For the baseline, Bahdanau's attention is implemented.

### Transformer Model
One of the Seq2Seq Model's shortcomings is that it cannot manage long-term dependencies. Hence, we implemented a transformer Architecture to mitigate this issue. The transformer's goal is to prevent recursion in order to facilitate parallel computing (reduce training time). 
Unlike the Seq2Seq approach, which relies on prior hidden states to capture relationships with previous words, the sentences are analyzed as a whole. Due to the parallelization ability of the transformer mechanism, much more data can be processed in the same amount of time with transformer models. Furthermore, it employs self attention, which is used to compute similarity scores between words in a phrase, and positional embedding, which replaces the recurrence and allows us to encode information related to a specific position of a token in a sentence. 

### T5 Model
