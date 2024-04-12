import os
import fitz 
from tqdm.auto import tqdm 
import re
import random
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from spacy.lang.en import English
import numpy as np
import matplotlib.pyplot as plt

nlp = English()
nlp.add_pipe("sentencizer")


class pdfReader:
    
    def __init__(self, files_dir):
        self.dir = files_dir
        self.files = os.listdir(self.dir)
        self.SENTENCE_CHUNK_SIZE = 10
        self.MIN_TOKEN_LENGTH = 50
    
    def readPDF(self, pdf_path):
        doc = fitz.open(pdf_path)
        pages_and_texts = []
        for page_num, page_content in tqdm(enumerate(doc)):
            text = page_content.get_text()
            text = self.text_clean(text)
            page_info = {
                "page_number": page_num,
                "num_chars": len(text),
                "num_words": len(re.sub("\s","",text)),
                "num_sentences": len(text.split(". ")),
                "num_tokens": len(text)/4,
                "text": text
            }
            pages_and_texts.append(page_info)
        return pages_and_texts
    
    def text_clean(self, text):
        cleaned_text = text.replace("\n", " ").strip()
        return cleaned_text
    
    def correctFormat(self, pages_and_texts):
        if not isinstance(pages_and_texts, pd.DataFrame):
            df = pd.DataFrame(pages_and_texts)
        return df

    def getPdfStatisticDescription(self,pages_and_texts):
        df = self.correctFormat(pages_and_texts)
        return df.describe().round(2)

    def sentencize(self, pages_and_texts):
        # pages_and_texts = self.correctFormat(pages_and_texts)
        for item in tqdm(pages_and_texts):
            item["sentences"] = list(nlp(item["text"]).sents)
            item["sentences"] = [str(sentence) for sentence in item["sentences"]]
            # Count the sentences 
            item["num_sentences_spacy"] = len(item["sentences"])
        return pages_and_texts
    
    def sentencesToChunks(self, pages_and_texts):
        def splitList(list_obj, stride):
            return [list_obj[i:i + stride] for i in range(0, len(list_obj), stride)]
        
        for item in tqdm(pages_and_texts):
            item["sentence_chunks"] = splitList(item["sentences"], self.SENTENCE_CHUNK_SIZE)
            item["num_chunks"] = len(item["sentence_chunks"])
        
        return pages_and_texts

    def getChunkRecord(self, pages_and_texts):
        pages_and_chunks = []
        for item in tqdm(pages_and_texts):
            for sentence_chunk in item["sentence_chunks"]:
                chunk_dict = {}
                chunk_dict["page_number"] = item["page_number"]
                
                # Join the sentences together into a paragraph-like structure, aka a chunk (so they are a single string)
                joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
                joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk) # ".A" -> ". A" for any full-stop/capital letter combo 
                chunk_dict["sentence_chunk"] = joined_sentence_chunk

                # Get stats about the chunk
                chunk_dict["chunk_num_chars"] = len(joined_sentence_chunk)
                chunk_dict["chunk_num_words"] = len([word for word in joined_sentence_chunk.split(" ")])
                chunk_dict["chunk_num_tokens"] = len(joined_sentence_chunk) / 4 
                
                pages_and_chunks.append(chunk_dict)

        df = self.correctFormat(pages_and_chunks)
        
        return df[df["chunk_num_tokens"] > self.MIN_TOKEN_LENGTH].to_dict(orient="records")

    def encodeChunkText(self, pages_and_chunks, model_name="all-mpnet-base-v2", ifSave=True, save_path = "./hengda_report/pages_and_chunks_with_embeddings.csv"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        embedding_model = SentenceTransformer(model_name_or_path=model_name, device=device) 
        
        text_chunks = [item["sentence_chunk"] for item in pages_and_chunks]
        text_chunk_embeddings = embedding_model.encode(text_chunks,
                                               batch_size=32, # you can use different batch sizes here for speed/performance, I found 32 works well for this use case
                                               convert_to_tensor=True) # optional to return embeddings as tensor instead of array

        embeddings_list = [embedding.numpy() if isinstance(embedding, torch.Tensor) else embedding for embedding in text_chunk_embeddings]

        # Add embeddings to the original data structure
        for item, embedding in zip(pages_and_chunks, embeddings_list):
            item['embedding'] = embedding.tolist() 
            
        pages_and_chunks_with_embeddings = pd.DataFrame(pages_and_chunks)
        
        if ifSave:
            pages_and_chunks_with_embeddings.to_csv(save_path, index=False)
        
        return pages_and_chunks_with_embeddings
    

        
if __name__ == "__main__":
    
    files_dir = "./hengda_report"
    
    hengdaReader = pdfReader(files_dir=files_dir)
    
    pages_and_texts = hengdaReader.readPDF("./hengda_report/intrep.pdf")
    # print(random.sample(pages_and_texts, k=3))
    # print(hengdaReader.getPdfStatisticDescription(pages_and_texts))
    
    pages_and_texts = hengdaReader.sentencize(pages_and_texts)
    # print(random.sample(pages_and_texts, k=1))
    # print(hengdaReader.getPdfStatisticDescription(pages_and_texts))
    
    pages_and_texts = hengdaReader.sentencesToChunks(pages_and_texts)
    # print(random.sample(pages_and_texts, k=1))
    # print(hengdaReader.getPdfStatisticDescription(pages_and_texts))
    
    pages_and_chunks = hengdaReader.getChunkRecord(pages_and_texts)
    print(hengdaReader.encodeChunkText(pages_and_chunks))
    