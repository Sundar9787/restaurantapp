import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

# Load the CountVectorizer and model
cv = CountVectorizer(max_features=1500)
loaded_model = pickle.load(open('nlp.pkl', 'rb'))
cv= pickle.load(open('cv.pkl','rb'))

new_review = 'I hate this restaurant '
new_review = re.sub('[^a-zA-Z]', ' ', new_review)
new_review = new_review.lower()
new_review = new_review.split()
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')
new_review = [ps.stem(word) for word in new_review if not word in set(all_stopwords)]
new_review = ' '.join(new_review)
new_corpus = [new_review]
new_X_test = cv.transform(new_corpus).toarray()
new_y_pred = loaded_model.predict(new_X_test)
if (new_y_pred==1):
    print("Its a positive review")
else:
    print("Its a negative review")