#import libraries
import pandas as pd
from openai import OpenAI

#loading data
df = pd.read_csv("Roget's Thesaurus.csv")

# Verify data
df.head()

# Input the api key here
client = OpenAI(api_key=" ")

def get_embedding(text, model="text-embedding-3-large"):
   return client.embeddings.create(input = [text], model=model).data[0].embedding

# creating embeddings
df['embeddings'] = df['Words'].apply(lambda x: get_embedding(x, model='text-embedding-3-large'))

# Saving the data
df.to_csv('Rogets_Thesaurus_Embeddings(OpenAI).csv', index=False)


#Centroids -------
#CLASS
# loading data
df = pd.read_csv("Class Centroids.csv")

# creating embeddings
df['embeddings'] = df['Words'].apply(lambda x: get_embedding(x, model='text-embedding-3-large'))

# Initialize an empty list to store flattened embeddings
flattened_embeddings = []

# Iterate over each row
for index, row in df.iterrows():
    # Append to the flattened_embeddings list
    flattened_embeddings.append(row['embeddings'])

# Create DataFrame from flattened embeddings
embeddings_df = pd.DataFrame(flattened_embeddings)

# Concatenate DataFrames
result_df = pd.concat([df, embeddings_df], axis=1)

# Drop the original embeddings column
result_df.drop('embeddings', axis=1, inplace=True)

result_df.head()

# Saving the data
result_df.to_csv('Class Centroids(OpenAI).csv', index=False)


#SECTION	
# loading data
df = pd.read_csv("Section Centroids.csv")

# creating embeddings
df['embeddings'] = df['Words'].apply(lambda x: get_embedding(x, model='text-embedding-3-large'))

# Initialize an empty list to store flattened embeddings
flattened_embeddings = []

# Iterate over each row
for index, row in df.iterrows():
    # Append to the flattened_embeddings list
    flattened_embeddings.append(row['embeddings'])

# Create DataFrame from flattened embeddings
embeddings_df = pd.DataFrame(flattened_embeddings)

# Concatenate DataFrames
result_df = pd.concat([df, embeddings_df], axis=1)

# Drop the original embeddings column
result_df.drop('embeddings', axis=1, inplace=True)

result_df.head()

# Saving the data
result_df.to_csv('Section Centroids(OpenAI).csv', index=False)
