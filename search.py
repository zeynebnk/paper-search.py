from keyword_extractor import nl_to_kw
from dense_search import dense_searcher
from tfidf_search import tfidf_searcher
from create_database import create_paper_repo
import os
papers_dir = "data/"
class search:
    def __init__(self, mode, searcher):
        self.mode = "KW"
        self.search_type = "DENSE"

        self.dense_searcher = dense_searcher(papers_dir)
        self.dense_searcher.construct_searcher()

        self.tfidf_searcher = tfidf_searcher(papers_dir)
        self.tfidf_searcher.construct_searcher()

        self.searcher = None

        self.k = 3
        self.threshold = None
    
    def set_mode(self):
        self.mode = input("Enter your search mode: Natural Language (NL) search OR Keywords (KW) search: ")
        while self.mode != "NL" and self.mode != "KW":
            self.mode = input("Invalid mode. \n Enter your search mode: Natural Language (NL) search OR Keywords (KW) search: ")
        
    def set_search_type(self):
        self.search_type = input("Enter your search method: TF-IDF (TFIDF) search OR Dense (DENSE) search: ")
        while self.search_type != "TFIDF" and self.search_type != "DENSE":
            self.search_type = input("Invalid search method. \n Enter your search method: TF-IDF (TFIDF) search OR Dense (DENSE) search: ")
        
        if self.search_type == "TFIDF":
            self.searcher = self.tfidf_searcher
        if self.search_type == "DENSE":
            self.searcher = self.dense_searcher

    def set_args(self):
        self.k = input("Enter the max number of results to return: ")
        self.k = int(self.k)
        self.threshold = input("Enter the relevance threshold (0-1) for the results or NA for no threshold: ")
        if self.threshold != "NA":
            self.threshold = float(self.threshold)
        else:
            self.threshold = None
    
    def search(self, query):
        if self.mode == "NL":
            extractor = nl_to_kw() # default model
            query = extractor.extract_keywords(query)
        
        if self.mode == "KW":
            query = query
        
        results = self.searcher.search(query, self.k, self.threshold)
        
        return results

    def perform_search(self):
        self.set_mode()
        self.set_search_type()
        self.set_args()
        q = input("Enter your search query (q to quit): ")
        if q == "q":
            return "q"
        return self.search(q)
        

def main():
    print("\n\n\n\n")
    print("__________________________________________________________")
    print("                    Paper Search Engine")
    print("__________________________________________________________")

    print("\n") 

    choice = input("C to continue with default database or type a research field to create a new database: ")
    
    if choice != "C":
        create_paper_repo(output_dir="data/",field=choice)

    searcher = search(mode="KW", searcher="DENSE")
    res = searcher.perform_search()
    while res != "q":
        print(res)
        res = searcher.perform_search()
    
    print("Thank you for using the Paper Search Engine!")
    

main()