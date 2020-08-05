import sys
# import libraries
import pandas as pd 
import re
from sqlalchemy import create_engine

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
import pickle



def load_data(database_filepath):
    """
    Loads disaster data from SQLite database
    
    Parameters:
        database_filepath(str): path to the SQLite database
        table_name(str): name of the table where the data is stored
        
    Returns:
        (DataFrame) X: Independent Variables , array which contains the text messages
        (DataFrame) Y: Dependent Variables , array which contains the labels to the messages
        (DataFrame) category: Data Column Labels , a list with the target column names, i.e. the category names
    """
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterResponse', con=engine)
    X = df.message
    Y = df.loc[:, 'related':'direct_report']
    category = Y.columns
    return X, Y, category

def tokenize(text):
    """
    Tokenizes message data
    
    INPUT:
       text (string): message text data
    
    OUTPUT:
        (DataFrame) clean_messages: array of tokenized message data
    """
    #Case Normalization & remove punctuation 
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower()) 
    
    #tokenization methods 
    
    words = word_tokenize(text)
    
    words = [w for w in words if w not in stopwords.words("english")]
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
    words = [WordNetLemmatizer().lemmatize(w, pos='v') for w in lemmed]

    
    return words


def build_model():
    """
    This function output is a SciKit ML Pipeline that process text messages
    according to NLP best-practice and apply a classifier.
        
    Returns:
        gs_cv(obj): an estimator which chains together a nlp-pipeline with
                    a multi-class classifier
    
    """
    moc = MultiOutputClassifier(RandomForestClassifier())

    pipeline = Pipeline([
                            ('vect', CountVectorizer(tokenizer=tokenize)),
                            ('tfidf', TfidfTransformer()),
                            ('clf', moc)
                                            ])
    # specify parameters for grid search - only limited paramter, as the training takes to much time,
    # more testing was done in the jupyter notebooks
    parameters = {'clf__estimator__max_depth': [10, 50, None],
              'clf__estimator__min_samples_leaf':[2, 5, 10]}

    cv = GridSearchCV(estimator=pipeline, param_grid=parameters)

    return cv
    


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluates models performance in predicting message categories
    
    INPUT:
        model (Classification): stored classification model
        X_test (string): Independent Variables
        Y_test (string): Dependent Variables
        category_names (DataFrame): Stores message category labels
    OUTPUT:
        None. But prints a classification report for every category
    """
    y_pred = model.predict(X_test)
    for i, col in enumerate(category_names):
        print(f'-----------------------{i, col}----------------------------------')
        print(classification_report(list(Y_test.values[:, i]), list(y_pred[:, i])))
        


def save_model(model, model_filepath):
    """
    Saves trained classification model to pickle file
    
    INPUT:
        model (Classification): stored classification model
        model_filepath (string): Filepath to pickle file
    """
    with open(model_filepath, 'wb') as file:  
        pickle.dump(model, file)


def main():
    """
    Train Classifier Main function
    
    This function applies the Machine Learning Pipeline:
        1) Extract data from SQLite db
        2) Train ML model on training set
        3) Estimate model performance on test set
        4) Save trained model as Pickle
        
    Args:
        None
        
    Returns:
        Nothing
 
    """
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