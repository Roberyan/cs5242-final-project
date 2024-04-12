import random
import torch
import numpy as np 
import pandas as pd
from sentence_transformers import util, SentenceTransformer
# Define helper function to print wrapped text 
import textwrap

def print_wrapped(text, wrap_length=80):
    wrapped_text = textwrap.fill(text, wrap_length)
    print(wrapped_text)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def getArrayFromString(array_str):
    return np.array(array_str.strip("[]").split(", "), dtype=float)

if __name__ == "__main__":
    aim_file_with_embedding = "./hengda_report/pages_and_chunks_with_embeddings.csv" 
    text_chunks_and_embedding_df = pd.read_csv(aim_file_with_embedding)
    
    text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df["embedding"].apply(getArrayFromString)
    
    pages_and_chunks = text_chunks_and_embedding_df.to_dict(orient="records")
    
    embeddings = torch.tensor(np.array(text_chunks_and_embedding_df["embedding"].tolist()), dtype=torch.float32).to(device)
    
    embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", device=device) # choose the device to load the model to
    
    # 1. Define the query
    # Note: This could be anything. But since we're working with a nutrition textbook, we'll stick with nutrition-based queries.
    query = "BORROWINGS of Hengda in 2023"
    print(f"Query: {query}")

    # 2. Embed the query to the same numerical space as the text examples 
    # Note: It's important to embed your query with the same model you embedded your examples with.
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)

    # 3. Get similarity scores with the dot product (we'll time this for fun)
    from time import perf_counter as timer

    start_time = timer()
    dot_scores = util.dot_score(a=query_embedding, b=embeddings)[0]
    end_time = timer()

    print(f"Time take to get scores on {len(embeddings)} embeddings: {end_time-start_time:.5f} seconds.")

    # 4. Get the top-k results (we'll keep this to 5)
    top_results_dot_product = torch.topk(dot_scores, k=5)
    print(top_results_dot_product)
    larger_embeddings = torch.randn(100*embeddings.shape[0], 768).to(device)
    print(f"Embeddings shape: {larger_embeddings.shape}")

    # Perform dot product across 168,000 embeddings
    start_time = timer()
    dot_scores = util.dot_score(a=query_embedding, b=larger_embeddings)[0]
    end_time = timer()

    print(f"Time take to get scores on {len(larger_embeddings)} embeddings: {end_time-start_time:.5f} seconds.")
    
    print(f"Query: '{query}'\n")
    print("Results:")
    # Loop through zipped together scores and indicies from torch.topk
    for score, idx in zip(top_results_dot_product[0], top_results_dot_product[1]):
        print(f"Score: {score:.4f}")
        # Print relevant sentence chunk (since the scores are in descending order, the most relevant chunk will be first)
        print("Text:")
        print_wrapped(pages_and_chunks[idx]["sentence_chunk"])
        # Print the page number too so we can reference the textbook further (and check the results)
        print(f"Page number: {pages_and_chunks[idx]['page_number']}")
        print("\n")