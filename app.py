import pickle
import base64
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import streamlit as st
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
loaded_model = pickle.load(open('nlp.pkl', 'rb'))
cv= pickle.load(open('cv.pkl','rb'))

def review_prediction(new_review):

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
    if new_y_pred == 1:
        return 'Its a Positive review'
    else:
        return 'Its a Negative review'

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
        f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: 100% 100%;
    }}
    </style>
    """,
        unsafe_allow_html=True,
    )

# Add background image
add_bg_from_local("new_restaurant.jpg")

def main():
  
    # Giving title
    st.title(':orange[Restaurant Review Detection] ')
    
    # Text input for review
    Review = st.text_area("*:blue[Your Review]*", height=100)
   
    # Code for prediction
    analysis = ''
    
    # Creating button for prediction
    if st.button("Predict"):
        analysis = review_prediction(Review)
        
    if "Its a Negative review" in analysis:
        st.markdown(f'<p style="color:red;background-color:#FFD2D2;padding:10px;border-radius:5px;">{analysis}</p>', unsafe_allow_html=True)
    elif "Its a Positive review" in analysis:
        st.markdown(f'<p style="color:green;background-color:#C2F9C2;padding:10px;border-radius:5px;">{analysis}</p>', unsafe_allow_html=True)
    else:
        st.success(analysis)


if __name__ == '__main__':
    main()
    
    
    
    
