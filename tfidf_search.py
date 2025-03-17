from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from langchain_community.document_loaders import DirectoryLoader, TextLoader
import argparse

class tfidf_searcher:
    def __init__(self, papers_dir):
        self.papers_dir = papers_dir
        self.vectorizer = TfidfVectorizer(
                lowercase=True,
                stop_words='english', 
                max_df=0.9,  
                min_df=3     
        )
        self.tfidf_matrix = None
        self.documents = []
        self.doc_contents = []
        self.doc_metadata = []

    def construct_searcher(self):
        loader = DirectoryLoader(self.papers_dir, glob="*.txt", loader_cls=TextLoader)
        self.documents = loader.load()

        for doc in self.documents:
            self.doc_contents.append(doc.page_content)
            title = doc.page_content.split("\n")[0]
            self.doc_metadata.append({
                "title": title,
                "source": doc.metadata["source"]
            })
        
        self.tfidf_matrix = self.vectorizer.fit_transform(self.doc_contents)

    
    def similarity_to_relevance(self, scores):
        mean = np.mean(scores)
        std = np.std(scores)

        
        z_scores = (scores - mean) / std
        relevance = 1 / (1 + np.exp(-z_scores))
        
        return relevance

    def search(self, query, k=3, threshold=None):

            
        q_vec = self.vectorizer.transform([query])
        
        results = []
        for idx, doc_vec in enumerate(self.tfidf_matrix):
            
            relevance = cosine_similarity(q_vec, doc_vec)[0][0]

            results.append((idx, relevance))
        

        relevance_scores = self.similarity_to_relevance([c[1] for c in results])
        results = [(doc, score) for doc, score in zip([c[0] for c in results], relevance_scores)]

        if threshold is not None:
            results = [(idx, score) for idx, score in results if score >= threshold]
        
        results = [(doc, self.similarity_to_relevance(score)) for doc, score in results]

        results = sorted(results, key=lambda x: x[1], reverse=True)[:k]
        
        result_str = ""
        for idx, score in results:
            metadata = self.doc_metadata[idx]
            result_str += f"{metadata['title']} (relevance: {score:.3f}) @ {metadata['source']}\n\n"
             
        return result_str
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TFIDF Search")
    parser.add_argument("--k", default=3, type=int, help="Top k papers to retrieve")
    parser.add_argument("--papers_dir", type=str, default="data/", help="Path of datastore")
    parser.add_argument("--query", type=str, default="retrieval models", help="Query to search")
    parser.add_argument("--threshold", type=float, default=None, help="Minimum relevance threshold (0-1)")
    parser.add_argument("--verbose", action="store_true", help="Show detailed relevance computation")
    
    args = parser.parse_args()
    
    searcher = tfidf_searcher(args.papers_dir)
    searcher.construct_searcher()
    results = searcher.search(args.query, args.k, args.threshold, args.verbose)
    print(results)
