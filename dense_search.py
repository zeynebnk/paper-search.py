from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import numpy as np
import argparse

class dense_searcher:
    def __init__(self, papers_dir, embedding_model="all-MiniLM-L6-v2"):
        self.papers_dir = papers_dir
        self.embedding_model = embedding_model
        self.db = None
        self.embeddings = None


    def construct_searcher(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
        
        loader = DirectoryLoader(self.papers_dir, glob="*.txt", loader_cls=TextLoader)
        documents = loader.load()
        
  
        self.db = FAISS.from_documents(documents, self.embeddings)

    def similarity_to_relevance(self, scores):
        mean = np.mean(scores)
        std = np.std(scores)

        if std == 0: 
            return np.ones_like(scores)
        
        z_scores = (scores - mean) / std
        relevance = 1 / (1 + np.exp(-z_scores))
        
        return relevance

    def search(self, query, k=3, threshold=None):
        
        docs_and_scores = self.db.similarity_search_with_score(query, k=k if threshold is None else 100)

        results = []
        for doc, distance in docs_and_scores:

            similarity = np.exp(-distance)
            results.append((doc, similarity))

        relevance_scores = self.similarity_to_relevance([c[1] for c in results])
        results = [(doc, score) for doc, score in zip([c[0] for c in results], relevance_scores)]
        
        if threshold is not None:
            results = [(doc, score) for doc, score in results if score >= threshold]
            results = sorted(results, key=lambda x: x[1], reverse=True)[:k]
        
        result_str = ""
        for doc, score in results:
            title = doc.page_content.split("\n")[0]
            source = doc.metadata["source"]
            result_str += f"{title} (relevance: {score:.3f}) @ {source}\n\n"
            
        return result_str

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dense Vector Search")
    parser.add_argument("--k", default=3, type=int, help="Top k papers to retrieve")
    parser.add_argument("--papers_dir", type=str, default="data/", help="Path of datastore")
    parser.add_argument("--query", type=str, default="retrieval models", help="Query to search")
    parser.add_argument("--embedding_model", type=str, default="all-MiniLM-L6-v2", help="HuggingFace embedding model to use")
    parser.add_argument("--threshold", type=float, default=None, help="Minimum relevance threshold (0-1)")
    
    args = parser.parse_args()
    
    searcher = dense_searcher(args.papers_dir, args.embedding_model)
    searcher.construct_searcher()
    results = searcher.search(args.query, args.k, args.threshold)
    print(results)