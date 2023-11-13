# Experiment: Spam Classifier with Naive Bayes model

In this experiment, I used a Naive Bayes model from `sklearn` to classify SMS messages as spam or not spam - based on instructions from an assignment on Udacity.

In [this notebook](spam_classifier.ipynb), I converted the messages in the dataset into Bag-of-Word representations manually first as examples. Then I used `CountVectorizer` class in sklearn to automatically convert the data into Bag-of-Word form that will be input into the Naive Bayes model. 

Spam messages tend to contain words like "free," "winner," "discount," etc. so based on the probabilities of those words in a large volume of spam messages, a Naive Bayes model can determine the probability that a new message is spam or not.

The notebook can be executed sequentially. New messages can be entered at the end in order to see the classification (1 being spam, 0 being not spam).

Four types of metrics are displayed at the end (accuracy, precision, recall and F1 score). In this example, the scenario where spams are mis-classified as not-spam are far less serious than if an email from your mother is mis-classified as spam. So accuracy alone is not enough to assess the performance of the model. Instead, precision score would be much more important.

## Examples
```
new_message = "first month free - no credit card needed"
test_instance = count_vectorizer.transform([new_message])
naive_bayes.predict(test_instance)
```
returns 1 (spam).

```
new_message = "win big prize - this weekend only. Terms and conditions apply."
test_instance = count_vectorizer.transform([new_message])
naive_bayes.predict(test_instance)
```
returns 1 (spam).

```
new_message = "hey, when're you picking me up?"
test_instance = count_vectorizer.transform([new_message])
naive_bayes.predict(test_instance)
```
returns 0 (not spam).
