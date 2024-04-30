import random
import torch
import numpy as np
import pandas as pd
import textwrap
import matplotlib.pyplot as plt
from sentence_transformers import util, SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import is_flash_attn_2_available
from transformers import BitsAndBytesConfig
import os
from askGPT import askGPT

os.environ['HF_HOME'] = "************"

quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

RETRIEVAL_CANDIDATES_NUM = 5
EMBEDDING_MODEL = "all-mpnet-base-v2"


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
    embeddings = torch.tensor(np.array(text_chunks_and_embedding_df["embedding"].tolist()), dtype=torch.float32).to(
        device)
    return pages_and_chunks, embeddings


def getAnswer(query, model, embeddings, sim_metric="cosine"):
    query_embedding = model.encode(query, convert_to_tensor=True)
    if sim_metric == "cosine":
        sim_relations = util.cos_sim(a=query_embedding, b=embeddings)[0]
    elif sim_metric == "dot":
        sim_relations = util.dot_score(a=query_embedding, b=embeddings)[0]
    scores, indices = torch.topk(sim_relations, k=RETRIEVAL_CANDIDATES_NUM)

    return scores, indices



def showAnswerPageForQuery(query, answer_page):
    plt.figure(figsize=(13, 10))
    plt.imshow(answer_page)
    plt.title(f"Query: '{query}' | Most relevant page:")
    plt.axis('off')  # Turn off axis
    plt.show()


def showGPU():
    gpu_memory_bytes = torch.cuda.get_device_properties(0).total_memory
    gpu_memory_gb = round(gpu_memory_bytes / (2 ** 30))
    print(f"Available GPU memory: {gpu_memory_gb} GB")


from tqdm import tqdm

if __name__ == "__main__":

    files_dir = "./hengda_report"
    # retriever = pdfReader(files_dir)

    pdf_path = "mamba.pdf"
    pdf_path = os.path.join(files_dir, pdf_path)

    # retriever.encodeChunkText(retriever.getChunkRecord(retriever.sentencesToChunks(retriever.sentencize(retriever.readPDF(pdf_path)))))

    pages_and_chunks, embeddings = getAimFileWithEmbedding(
        file_path="./hengda_report/pages_and_chunks_with_embeddings.csv")

    embedding_model = SentenceTransformer(model_name_or_path=EMBEDDING_MODEL,
                                          device=device)  # choose the device to load the model to


    showGPU()
    print("*" * 50)


    # let's explore LLM
    questions = []
    with open("./questions.txt", "r", encoding='utf-8') as f:
        for line in f:
            if len(line.strip()) > 5:
                questions.append(line.strip())
    answers = []
    for i, question in enumerate(tqdm(questions)):
        input_text = question
        query = input_text
        # print(f"Input text:\n{input_text}",end="\n\n")

        # Create prompt template for instruction-tuned model
        dialogue_template = [
            {"role": "user",
             "content": input_text}
        ]


        from utils import prompt_formatter, prompt_formatter_withoutRAG

        scores, indices = getAnswer(query=query, model=embedding_model, embeddings=embeddings, sim_metric="cosine")
        context_items = [pages_and_chunks[i] for i in indices]
        prompt = prompt_formatter(query=query,
                                  context_items=context_items)
        prompt_withoutRAG = prompt_formatter_withoutRAG(query=query)
        # Turn the output tokens into text
        output_text = askGPT(prompt)
        output_withoutRAG = askGPT(prompt_withoutRAG)

        answer = ""
        # print(f"Query: {query}",end="\n\n")
        answer = answer + f"Query {i + 1}: {query}\n\n"
        # print(f"RAG answer:\n{output_text.replace(prompt, '')}", end="\n\n")
        answer = answer + f"RAG answer:\n{output_text.replace(prompt, '')}\n\n"
        # print(f"Without RAG answer:\n{output_withoutRAG.replace(prompt_withoutRAG, '')}")
        answer = answer + f"Without RAG answer:\n{output_withoutRAG.replace(prompt_withoutRAG, '')}"
        answers.append(answer)
        torch.cuda.empty_cache()
    with open("gpt_3.5_answers.txt", "w", encoding='utf-8') as f:
        f.write("\n".join(answers))
