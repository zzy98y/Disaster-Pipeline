# Disaster Response Pipeline Project

### Package 
* Pandas 
* SQLAlchemy (Database Package) 
* Regular Expression (No need to install, import re) 
* NLTK (Natural Language Toolkit) 
* Sklearn (Machine Learning Package) 

### Our Goal
We're creating a Disaster Response that take a disaster message and apply Machine Learning technique to predict which of the 36 category is this message about. 
(Please note, the message can be about one more category at the same time) 



### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
