# Disaster Response Pipeline Project
## Udacity Data Scientist Nanodegree

### Description:

This Project is a part of Data Science Nanodegree Program by Udacity in collaboration with Figure Eight. The initial dataset contains pre-labelled tweet and messages from real-life disasters. The aim of this project is to build a Natural Language Processing tool that categorize messages.

The Project is divided in the following Sections:

1. Building an ETL pipeline to extract data, clean the data and save the data in a SQLite Database
2. Building a ML pipeline to train our model
3. Run a Web App to show our model results

### Installation: 
Clone the GitHub repository and use Anaconda distribution of Python 3.6.7.

    $ git clone https://github.com/Swatichanchal/Disaster-Response-Pipeline.git

In addition This will require pip installation of the following:

    $ pip install SQLAlchemy
    $ pip install nltk

1. Python 3+
2. ML Libraries: NumPy,  Pandas, SciPy, SkLearn
3. NLP Libraries: NLTK
4. SQLlite Libraries: SQLalchemy
5. Model Loading and Saving Library: Pickle
6. Web App and Visualization: Flask, Plotly

The code can be viewed and modified with Jupyter Notebooks.


### Data:

The data in this project comes from Figure Eight - Multilingual Disaster Response Messages. This dataset contains 30,000 messages drawn from events including an earthquake in Haiti in 2010, an earthquake in Chile in 2010, floods in Pakistan in 2010, super-storm Sandy in the U.S.A. in 2012, and news articles spanning a large number of years and 100s of different disasters.

The data has been encoded with 36 different categories related to disaster response and has been stripped of messages with sensitive information in their entirety.

Data includes 2 csv files:

1. disaster_messages.csv: Messages data.
2. disaster_categories.csv: Disaster categories of messages.

### Files:
process_data.py : ETL script write a data cleaning pipeline that:
- Loads the messages and categories datasets
- Merges the two datasets
- Cleans the data
- Stores it in a SQLite database

train_classifier.py : script write a machine learning pipeline that:
- Loads data from the SQLite database
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file

run.py : Main file to run Flask app that classifies messages based on the model and shows data visualizations.

- Link of my git hub repository : https://github.com/Swatichanchal/Disaster-Response-Pipeline.git

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

#### Note: Notebook Folder is not essential to run the Web App.

### Built With:
- Visual Studio Code
- Udacity Project Workspace IDE

### Screenshots:

https://github.com/Swatichanchal/Disaster-Response-Pipeline/blob/master/Screenshots/Web_App.png

https://github.com/Swatichanchal/Disaster-Response-Pipeline/blob/master/Screenshots/Web_App1.png

https://github.com/Swatichanchal/Disaster-Response-Pipeline/blob/master/Screenshots/Web_App2.png


### Acknowledgements:
1. [Udacity](www.udacity.com) for this Data Science Nanodegree Program.
2. [Figure-Eight](www.figure-eight.com) for the relevant dataset.

