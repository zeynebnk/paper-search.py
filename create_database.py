import arxiv
import requests
import PyPDF2
import argparse

def download_pdf(url, output_path):
  response = requests.get(url)
  if response.status_code == 200:
    with open(output_path, "wb") as f:
      f.write(response.content)
      return True
  else:
    return False

def extract_text_from_pdf(pdf_path):
  with open(pdf_path, "rb") as file:
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
      text += page.extract_text() + "\n"
    return text

def create_paper_repo(output_dir,field,n=20):
  
  client = arxiv.Client()
  search = arxiv.Search(
        query=field,
        max_results=n
    )
  
  for paper in client.results(search):
    title = paper.title
    path = "_".join(paper.title.split(" ")[:5]) 

    if download_pdf(paper.pdf_url, path + ".pdf"):
      text = extract_text_from_pdf(path + ".pdf")
      if text:
        with open(output_dir + path + ".txt", "w", encoding="utf-8") as f:
          f.write(text)
  

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Construct paper repository")
  parser.add_argument("--datastore_size",default=20, type=int,help="Max number of papers to load")
  parser.add_argument("--output_dir", type=str, default="",help="Path to save")
  parser.add_argument("--field", type=str, default="large language models",help="Field of papers to retrieve")
  
  args = parser.parse_args()
  create_paper_repo(args.output_dir,args.field,n=args.datastore_size)
