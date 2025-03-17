Option 2: File Selection

I built a research paper searching engine. 

I construct a database of papers from arXiv by automatically downloading and extracting the contents of papers from the site. By default, the datastore has papers related to large language models; however, the user can decide to generate a new database on papers relevant to their field.

Then, I have two search methods: dense/FAISS search and TF-IDF based search. I use cosine similarity as a similarity metric to find the closest matches to the query, and then generate a relevance score based on the z-scores of the similarities. The threshold mechanism can be set to this value. The user can also set the maximum number of papers returned (top k). 

As an extension, I allow for natural language search instead keyword-based queries, using a keyword extraction model to generate a keywork query from a search like "I need papers to reference on scaling inference compute in language models". 

The search will return the titles of the top relevant results with their scores and their location in the database.

With more time, I would like to implement a weighted method for the TF-IDF search to prioritize words in the title or abstract more, explore further ways to generate a relevance score, and create a more streamlined front-end that is beyond the terminal.

Note: To start the paper search enginer, run 'python search.py'. 

