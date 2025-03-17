from transformers import pipeline
import argparse

class nl_to_kw:
    def __init__(self, model_name="yanekyuk/bert-uncased-keyword-extractor"):
        self.extractor = pipeline(
            "token-classification",
            model=model_name,
            aggregation_strategy="simple"
        )

    def extract_keywords(self, query):
        
        results = self.extractor(query)
        keywords = [result["word"] for result in results]
        
        return " ".join(keywords)

def main():
    
    extractor = nl_to_kw(args.model)
    kws = extractor.extract_keywords(args.query)
    print(kws)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Natural Language to Keywords Converter")
    parser.add_argument("--model", type=str, 
                       default="yanekyuk/bert-uncased-keyword-extractor",
                       help="HuggingFace model to use for keyword extraction")
    parser.add_argument("--query", type=str,
                       default="I want papers about transformer architectures in deep learning",
                       help="Query to extract keywords from")
    
    args = parser.parse_args()
    main()
