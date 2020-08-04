import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    
    """
    Load dataframe from filepaths
    INPUT
    messages_filepath (string): Filepath to messages.csv file
        categories_filepath (string): Filepath to categories.csv file
   
    OUTPUT
    (DataFrame) df: Consolidated Pandas dataframe
    """
    
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = messages.merge(categories, on="id")
    
    return df


def clean_data(df):
    """
    Clean data included in the DataFrame and transform categories part
    INPUT
        df (DataFrame): Merged DataFrame
    
    OUTPUT
        (DataFrame) df: Returns Cleaned DataFrame
    """
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(";", expand=True)
    # select the first row of the categories dataframe
    row = categories.loc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda i: i[:-2])
    # rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] =  categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    # drop the original categories column from `df`
    df = df.drop(columns = 'categories') 
    # concatenate the original dataframe with the new `categories` dataframe
    df = df.join(categories)
    # drop duplicates
    df = df.drop_duplicates()
    
    return df
     


def save_data(df, database_filename):
    """
    Stores a Data Frame in a SQLite database
    Args:
        df(obj): a pandas Data Frame
        table_name(str): name of the table
        database_filename(str): Name of the SQLite database file
    
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterResponse', engine , index = False)


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