# Neural-Machine-Translation
The goals:
•	To understand the steps to train/test the model for natural language processing.
•	Understand architecture of RNN and how to connect each layer together by using TensorFlow.
•	To implement and understand RNN using TensorFlow.

The English-Czech, English-German, and English-Vietnamese datasets can be found at: https://nlp.stanford.edu/projects/nmt/  under Preprocessed Data.

First Creat folder ./model

Use command "Python NMT train" to train the model, save the model in a folder named “model” after finish the training.

Use command "Python NMT test" to test the model. This command will (1) load data from (tst2012.en or tst2013.en) to translate sentences; and (2) will calculate BLEU score (https://www.nltk.org/_modules/nltk/translate/bleu_score.html) with smoothing method1. 

Use command "Python NMT translate" run the translate function. Given a sentence, this model will be able to translate it into the corresponding language. 

Sample code found at: https://github.com/chiphuyen/stanford-tensorflow-tutorials/tree/master/assignments/chatbot
