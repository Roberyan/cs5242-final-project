import random
import torch
import numpy as np 
import pandas as pd
from sentence_transformers import util, SentenceTransformer
import textwrap
import matplotlib.pyplot as plt
import fitz
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import is_flash_attn_2_available 
from transformers import BitsAndBytesConfig
quantization_config = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=torch.float16)

RETRIEVAL_CANDIDATES_NUM = 5
EMBEDDING_MODEL = "all-mpnet-base-v2"

MODEL_ID = "google/gemma-2b"

def print_wrapped(text, wrap_length=80):
    wrapped_text = textwrap.fill(text, wrap_length)
    print(wrapped_text)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def getArrayFromString(array_str):
    return np.array(array_str.strip("[]").split(", "), dtype=float)

def getAimFileWithEmbedding(file_path):
    text_chunks_and_embedding_df = pd.read_csv(file_path)
    text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df["embedding"].apply(getArrayFromString)
    pages_and_chunks = text_chunks_and_embedding_df.to_dict(orient="records")
    embeddings = torch.tensor(np.array(text_chunks_and_embedding_df["embedding"].tolist()), dtype=torch.float32).to(device)
    return pages_and_chunks, embeddings

def getAnswer(query, model, embeddings, sim_metric="cosine"):
    query_embedding = model.encode(query, convert_to_tensor=True)
    if sim_metric == "cosine":
        sim_relations = util.cos_sim(a=query_embedding, b=embeddings)[0]
    elif sim_metric == "dot":
        sim_relations = util.dot_score(a=query_embedding, b=embeddings)[0]
    scores, indices = torch.topk(sim_relations, k=RETRIEVAL_CANDIDATES_NUM)

    return scores, indices

def getPageImg(file_path, page_num):
    doc = fitz.open(file_path)
    page = doc.load_page(page_num)
    img = page.get_pixmap(dpi=300)
    doc.close()
    img_array = np.frombuffer(img.samples_mv, dtype=np.uint8).reshape((img.h, img.w, img.n))

    return img_array

def showAnswerPageForQuery(query, answer_page):
    plt.figure(figsize=(13, 10))
    plt.imshow(answer_page)
    plt.title(f"Query: '{query}' | Most relevant page:")
    plt.axis('off') # Turn off axis
    plt.show()

def showGPU():
    gpu_memory_bytes = torch.cuda.get_device_properties(0).total_memory
    gpu_memory_gb = round(gpu_memory_bytes / (2**30))
    print(f"Available GPU memory: {gpu_memory_gb} GB")



if __name__ == "__main__":

    pdf_path = "./hengda_report/intrep.pdf"
    
    pages_and_chunks, embeddings = getAimFileWithEmbedding(file_path="./hengda_report/pages_and_chunks_with_embeddings.csv" )
    
    embedding_model = SentenceTransformer(model_name_or_path=EMBEDDING_MODEL, device=device) # choose the device to load the model to
    

    query = "BORROWINGS of Hengda in 2023"

    # Get the top-k results 
    scores, indices = getAnswer(query,model=embedding_model,embeddings=embeddings)
    # print(top_results_dot_product)

    print(f"Query: '{query}'\n")
    print("Results:")
    # Loop through zipped together scores and indicies from torch.topk
    for score, idx in zip(scores, indices):
        print(f"Score: {score:.4f}")
        # Print relevant sentence chunk (since the scores are in descending order, the most relevant chunk will be first)
        print("Text:")
        print_wrapped(pages_and_chunks[idx]["sentence_chunk"])
        # Print the page number too so we can reference the textbook further (and check the results)
        print(f"Page number: {pages_and_chunks[idx]['page_number']}")
        print("\n")
    
    img_array = getPageImg(pdf_path, 50)
    
    showAnswerPageForQuery(query, img_array)
    
    