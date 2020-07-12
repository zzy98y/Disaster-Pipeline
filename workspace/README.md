# Disaster Response Pipeline Project

## Package 
* Pandas 
* SQLAlchemy (Database Package) 
* Regular Expression (No need to install, import re) 
* NLTK (Natural Language Toolkit) 
* Sklearn (Machine Learning Package) 

## Our Goal
We're creating a Disaster Response that take a disaster message and apply Machine Learning technique to predict which of the 36 category is this message about. 
(Please note, the message can be about one more category at the same time) 

### File Explained
**app Folder**: 
**template folder**: There are two html file inside this folder, this is the two file that build the front end of the website, and it was given to me in the beginning of the project. 

**run.py**: This file is the file that used to the launch the website and make sure all the algorithm and methods are implemented in the back end of the website, feel free to modify any visualization in this file. 

**data folder**: 
**disaster_categories.csv**: the csv file that contains the data about all the 36 categories. 

**disaster_messages.csv**: the csv file that contains the data about the disaster messages. 

**DisasterResponse.db**: the sqlite database use to store the cleaned data through the ETL pipeline.

**process_data.py**: the python script that run through the ETL pipeline that help clean the data. 

**model folder**:
**train_classifier**: the python script file that run through machine learning pipeline and categorize the message in the correct category. 

**classifier.pkl**: the pickle file that store the machine learning model. 



## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
