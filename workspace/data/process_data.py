import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    "this function load the data messages and categories and merge into one called df and return this df"
    messages = pd.read_csv(messages_filepath)   # load messages data
    categories = pd.read_csv(categories_filepath) # load categories data
    df = pd.merge(messages,categories,on = 'id') # merge both dataframe on the common key 'id'
    return df

def clean_data(df):
    "this function expand the categories column and extract the letter and set as column name and number as the result and drop the original categories column, and drop duplicates"
    cate = df['categories'].str.split(';',expand = True) 
    # seperate each category seperated by colon and form a seperate column 
    row = cate.iloc[0].tolist()
    # extract the first row of the new categories dataframe 
    category_colnames = [i[0:-2] for i in row]
    # get rid of the number at the end of each categories 
    cate.columns = category_colnames
    # set the column name for the data 
    for column in cate:
        # set each value to be the last character of the string
        cate[column] = [i[-1] for i in cate[column].tolist()]
        # convert column from string to numeric
        cate[column] = cate[column].astype(int)
    # drop the original categories column from `df`
    df.drop(columns = ['categories'],inplace = True)
    # merge the cleaned categories and the cleaned df into one df 
    df = pd.concat([df,cate],axis = 1)
    # find the duplicate row in the dataframe
    df_dup = df[df.duplicated()]
    # drop duplicates
    df.drop(df_dup.index,inplace = True)
    # find the rows with related is 2 which doesn't make any sense 
    df_related_2 = df[df['related'] == 2] 
    # drop the row that related is 2 
    df.drop(df_related_2.index,inplace = True) 
    
    return df

def save_data(df, database_filename):
    "This function save the cleaned data into a SQLite Database "
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('etl_clean', engine, index=False,if_exists = 'replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
