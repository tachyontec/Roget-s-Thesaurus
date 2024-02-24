# Gemini
import google.generativeai as genai

# Read the data from the DataFrame
df = pd.read_csv("Roget's Thesaurus.csv")

API_KEY = ""
genai.configure(api_key=API_KEY)

Words_list = df['Words'].tolist()

#model
model = genai.embed_content(model="models/embedding-001",
                            content=Words_list,
                           task_type='clustering')

#Extracting the embeddings
Embeddings = model['embedding']

#Subsetting the first 600 vectors
Embeddings_600 = [sublist[:600] for sublist in Embeddings]

# Convert the list of lists to a DataFrame
embedding_df = pd.DataFrame(Embeddings_600)

# Concatenate embedding DataFrame with original DataFrame
df = pd.concat([df, embedding_df], axis=1)

# Save DataFrame to CSV file
df.to_csv("Rogets_Thesaurus_Embeddings(Gemini).csv", index=False)


#Now get the centroids --------------------

#CLASS
# Read the data from the DataFrame
df = pd.read_csv("Class Centroids.csv") # File with each class name

Words_list = df['Words'].tolist()

#model
model = genai.embed_content(model="models/embedding-001",
                            content=Words_list,
                           task_type='clustering')

#Extracting the embeddings
Embeddings = model['embedding']

#Subsetting the first 600 vectors
Embeddings_600 = [sublist[:600] for sublist in Embeddings]

# Convert the list of lists to a DataFrame
embedding_df = pd.DataFrame(Embeddings_600)

# Concatenate embedding DataFrame with original DataFrame
df = pd.concat([df, embedding_df], axis=1)

# Save DataFrame to CSV file
df.to_csv("Class Centroids embed(Gemini).csv", index=False)


#SECTION
# Read the data from the DataFrame
df = pd.read_csv("Section Centroids.csv") # File with each section name

Words_list = df['Words'].tolist()

#model
model = genai.embed_content(model="models/embedding-001",
                            content=Words_list,
                           task_type='clustering')

#Extracting the embeddings
Embeddings = model['embedding']

#Subsetting the first 600 vectors
Embeddings_600 = [sublist[:600] for sublist in Embeddings]

# Convert the list of lists to a DataFrame
embedding_df = pd.DataFrame(Embeddings_600)

# Concatenate embedding DataFrame with original DataFrame
df = pd.concat([df, embedding_df], axis=1)

# Save DataFrame to CSV file
df.to_csv("Section Centroids embed(Gemini).csv", index=False)
