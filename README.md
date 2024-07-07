# Sentiment Analysis of Restaurant Reviews using NLP
This project involves performing sentiment analysis on restaurant reviews using Natural Language Processing (NLP) techniques. The aim is to classify reviews as positive or negative.

## Project Overview
The project includes the following steps:

### Importing Libraries
### Loading the Dataset
### Text Cleaning and Preprocessing
### Creating the Bag of Words Model
### Splitting the Dataset
### Training the Logistic Regression Model
### Evaluating the Model
### Predicting New Reviews
### Saving and Loading the Model

1. Importing Libraries
The following libraries are required for the project:

python
Copy code
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import pickle
2. Loading the Dataset
Load the dataset from a .tsv file:

python
Copy code
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)
3. Text Cleaning and Preprocessing
Clean the text data by removing non-alphabetic characters, converting to lowercase, removing stopwords, and stemming:

python
Copy code
nltk.download('stopwords')
corpus = []
for i in range(0, 1007):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)
4. Creating the Bag of Words Model
Transform the cleaned text data into feature vectors:

python
Copy code
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values
5. Splitting the Dataset
Split the dataset into training and testing sets:

python
Copy code
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
6. Training the Logistic Regression Model
Train a logistic regression model:

python
Copy code
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)
7. Evaluating the Model
Evaluate the model's performance using a confusion matrix and accuracy score:

python
Copy code
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print('Accuracy:', accuracy_score(y_test, y_pred))
8. Predicting New Reviews
Predict whether a new review is positive or negative:

python
Copy code
def predict_review(review):
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    new_corpus = [review]
    new_X_test = cv.transform(new_corpus).toarray()
    new_y_pred = classifier.predict(new_X_test)
    return 'Positive' if new_y_pred == 1 else 'Negative'

print(predict_review('I love this restaurant'))
print(predict_review('I hate this restaurant'))
9. Saving and Loading the Model
Save the trained model and the CountVectorizer:

python
Copy code
pickle.dump(classifier, open('nlp.pkl', 'wb'))
pickle.dump(cv, open('cv.pkl', 'wb'))

# Loading the model
loaded_model = pickle.load(open('nlp.pkl', 'rb'))
cv_model = pickle.load(open('cv.pkl', 'rb'))
Repository Structure
Sentimental_Analysis_of_Restaurant_Review_using_NLP.ipynb: Jupyter notebook containing the full code.
Restaurant_Reviews.tsv: Dataset used for training and testing the model.
nlp.pkl: Serialized logistic regression model.
cv.pkl: Serialized CountVectorizer.
Conclusion
This project demonstrates how to perform sentiment analysis on text data using NLP techniques and logistic regression. The model can classify new reviews as positive or negative with a certain degree of accuracy.
