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

quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

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
    plt.axis('off')  # Turn off axis
    plt.show()


def showGPU():
    gpu_memory_bytes = torch.cuda.get_device_properties(0).total_memory
    gpu_memory_gb = round(gpu_memory_bytes / (2 ** 30))
    print(f"Available GPU memory: {gpu_memory_gb} GB")


if __name__ == "__main__":

    pdf_path = "./hengda_report/intrep.pdf"

    pages_and_chunks, embeddings = getAimFileWithEmbedding(
        file_path="./hengda_report/pages_and_chunks_with_embeddings.csv")

    embedding_model = SentenceTransformer(model_name_or_path=EMBEDDING_MODEL,
                                          device=device)  # choose the device to load the model to

    query = "How to determine whether a person has depression?"

    # Get the top-k results 
    scores, indices = getAnswer(query, model=embedding_model, embeddings=embeddings)
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

    # Now we try to use dot product as similarity
    print("=" * 50)

    scores, indices = getAnswer(query=query, model=embedding_model, embeddings=embeddings, sim_metric="dot")
    print(scores)
    print(indices)
    for score, idx in zip(scores, indices):
        print(f"Score: {score:.4f}")
        # Print relevant sentence chunk (since the scores are in descending order, the most relevant chunk will be first)
        print("Text:")
        print_wrapped(pages_and_chunks[idx]["sentence_chunk"])
        # Print the page number too so we can reference the textbook further (and check the results)
        print(f"Page number: {pages_and_chunks[idx]['page_number']}")
        print("\n")
    # cos dot has completed


    showGPU()
    print("*" * 50)
    from utils import get_llm
    tokenizer, llm_model = get_llm(MODEL_ID)
    print(llm_model)

    # let's explore LLM
    # TODO 增加更多的询问例句
    questions = ["What is depression?"]
    input_text = random.choice(questions)
    query = input_text
    print(f"Input text:\n{input_text}")

    # Create prompt template for instruction-tuned model
    dialogue_template = [
        {"role": "user",
         "content": input_text}
    ]

    # Apply the chat template
    prompt = tokenizer.apply_chat_template(conversation=dialogue_template,
                                           tokenize=False,  # keep as raw text (not tokenized)
                                           add_generation_prompt=True)
    print(f"\nPrompt (formatted):\n{prompt}")

    # Tokenize the input text (turn it into numbers) and send it to GPU
    input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")
    # print(f"Model input (tokenized):\n{input_ids}\n")

    # Generate outputs passed on the tokenized input
    # See generate docs: https://huggingface.co/docs/transformers/v4.38.2/en/main_classes/text_generation#transformers.GenerationConfig
    outputs = llm_model.generate(**input_ids,
                                 max_new_tokens=256)  # define the maximum number of new tokens to create
    outputs_decoded = tokenizer.decode(outputs[0])
    print(f"Model output (decoded):\n{outputs_decoded}\n")
    print("^" * 50)
    print(f"Input text: {input_text}\n")
    print(f"Output text:\n{outputs_decoded.replace(prompt, '').replace('<bos>', '').replace('<eos>', '')}")

    from utils import prompt_formatter, prompt_formatter_withoutRAG

    scores, indices = getAnswer(query=query, model=embedding_model, embeddings=embeddings, sim_metric="dot")
    context_items = [pages_and_chunks[i] for i in indices]
    prompt = prompt_formatter(query=query,
                              context_items=context_items,
                              tokenizer=tokenizer)
    prompt_withoutRAG = prompt_formatter_withoutRAG(query=query,
                              tokenizer=tokenizer)
    input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_ids_withoutRAG = tokenizer(prompt_withoutRAG, return_tensors="pt").to("cuda")
    # final
    print("#" * 60)
    # Generate an output of tokens
    outputs = llm_model.generate(**input_ids,
                                 temperature=0.7,
                                 # lower temperature = more deterministic outputs, higher temperature = more creative outputs
                                 do_sample=True,
                                 # whether or not to use sampling, see https://huyenchip.com/2024/01/16/sampling.html for more
                                 max_new_tokens=256)  # how many new tokens to generate from prompt
    outputs_withoutRAG = llm_model.generate(**input_ids_withoutRAG,
                                 temperature=0.7,
                                 # lower temperature = more deterministic outputs, higher temperature = more creative outputs
                                 do_sample=True,
                                 # whether or not to use sampling, see https://huyenchip.com/2024/01/16/sampling.html for more
                                 max_new_tokens=256)
    # Turn the output tokens into text
    output_text = tokenizer.decode(outputs[0])
    output_withoutRAG = tokenizer.decode(outputs_withoutRAG[0])
    print(f"Query: {query}")
    print(f"RAG answer:\n{output_text.replace(prompt, '')}")
    print(f"Without RAG answer:\n{output_withoutRAG.replace(prompt, '')}")

