import sys
import pandas as pd 
import numpy as np
from nltk.tokenize import word_tokenize
import re 
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
import nltk
import pickle 
nltk.download(['punkt', 'wordnet'])
from sklearn.datasets import make_multilabel_classification
from sklearn.metrics import classification_report 
from sklearn.model_selection import GridSearchCV


def load_data(database_filepath):
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('etl_clean',engine)
    X_old = df['message']
    Y_old = df.iloc[:,4:41]
    Not_NaN = [i for i in range(len(Y_old)) if not any(Y_old.iloc[i].isnull())]
    X = X_old.iloc[Not_NaN]
    Y = Y_old.iloc[Not_NaN]
    category_name = Y.columns
    
    return X,Y,category_name
    
   
    
def tokenize(text):
    url_regx = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regx,text)
    
    for url in detected_urls:
        text = text.replace(url,'urlplaceholder')
        
    tokens = word_tokenize(text)
    
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    return clean_tokens


def build_model():
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    
    parameters = {
    'vect__ngram_range': ((1, 1), (1, 2)),
#     'vect__max_df': (0.8, 1.0),
#     'vect__max_features': (None, 10000),
#     'clf__estimator__n_estimators': [50, 100],
#     'clf__estimator__learning_rate': [0.1, 1.0]
    }
    
    cv = GridSearchCV(pipeline, parameters, cv=3, n_jobs=-1)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    for i in range(len(category_names)):
        Y_pred = model.predict(X_test)[i]
        Y_true = np.array(Y_test.iloc[i].values)
        Target = [category_names[i]]
        print(classification_report(Y_true, Y_pred, target_names= Target))
        


def save_model(model, model_filepath):
    pkl_file = model_filepath
    model_pickle = open(pkl_file,'wb')
    pickle.dump(model, model_pickle)
    model_pickle.close()

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()