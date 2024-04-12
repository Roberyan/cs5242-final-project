import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import is_flash_attn_2_available
# 1. Create quantization config for smaller model loading (optional)
# Requires !pip install bitsandbytes accelerate, see: https://github.com/TimDettmers/bitsandbytes, https://huggingface.co/docs/accelerate/
# For models that require 4-bit quantization (use this if you have low GPU memory available)
import huggingface_hub
huggingface_hub.login("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")
from transformers import BitsAndBytesConfig
def get_llm(model_id, use_quantization_config=True):
    quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                             bnb_4bit_compute_dtype=torch.float16)

    # Bonus: Setup Flash Attention 2 for faster inference, default to "sdpa" or "scaled dot product attention" if it's not available
    # Flash Attention 2 requires NVIDIA GPU compute capability of 8.0 or above, see: https://developer.nvidia.com/cuda-gpus
    # Requires !pip install flash-attn, see: https://github.com/Dao-AILab/flash-attention
    if (is_flash_attn_2_available()) and (torch.cuda.get_device_capability(0)[0] >= 8):
      attn_implementation = "flash_attention_2"
    else:
      attn_implementation = "sdpa"
    print(f"[INFO] Using attention implementation: {attn_implementation}")

    # 2. Pick a model we'd like to use (this will depend on how much GPU memory you have available)
    #model_id = "google/gemma-7b-it"
    model_id = model_id # (we already set this above)
    print(f"[INFO] Using model_id: {model_id}")

    # 3. Instantiate tokenizer (tokenizer turns text into numbers ready for the model)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_id)

    # 4. Instantiate the model
    llm_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_id,
                                                     torch_dtype=torch.float16, # datatype to use, we want float16
                                                     quantization_config=quantization_config if use_quantization_config else None,
                                                     low_cpu_mem_usage=False, # use full memory
                                                     attn_implementation=attn_implementation) # which attention version to use

    return tokenizer, llm_model

def prompt_formatter(query: str,
                     context_items: list[dict],
                     tokenizer) -> str:
    """
    Augments query with text-based context from context_items.
    """
    # Join context items into one dotted paragraph
    context = "- " + "\n- ".join([item["sentence_chunk"] for item in context_items])

    # Create a base prompt with examples to help the model
    # Note: this is very customizable, I've chosen to use 3 examples of the answer style we'd like.
    # We could also write this in a txt file and import it in if we wanted.
    base_prompt = """Based on the following context items, please answer the query.
Give yourself room to think by extracting relevant passages from the context before answering the query.
Don't return the thinking, only return the answer.
Make sure your answers are as explanatory as possible.
Use the following examples as reference for the ideal answer style.
\nExample 1:
Query: How is the performance of Nvidia in 2023?
Answer: NVIDIA had a strong performance in various sectors throughout the year.
\nExample 2:
Query: How is the performance of Moderna in 2023?
Answer: In 2023, Moderna's performance can be considered robust in several key aspects, despite a significant decline in its stock price.
\nExample 3:
Query: Whether a company with a significant decline in profits and revenue will have a good market prospect?
Answer: Generally speaking, a significant decline in profits and revenue can signal challenges for a company's market prospects. However, market prospects are not solely determined by current financial metrics. 
\nNow use the following context items to answer the user query:
{context}
\nRelevant passages: <extract relevant passages from the context here>
User query: {query}
Answer:"""

    # Update base prompt with context items and query
    base_prompt = base_prompt.format(context=context, query=query)

    # Create prompt template for instruction-tuned model
    dialogue_template = [
        {"role": "user",
        "content": base_prompt}
    ]

    # Apply the chat template
    prompt = tokenizer.apply_chat_template(conversation=dialogue_template,
                                          tokenize=False,
                                          add_generation_prompt=True)
    return prompt

def prompt_formatter_withoutRAG(query: str,
                     tokenizer) -> str:
    """
    Augments query with text-based context from context_items.
    """
    # Create a base prompt with examples to help the model
    # Note: this is very customizable, I've chosen to use 3 examples of the answer style we'd like.
    # We could also write this in a txt file and import it in if we wanted.
    base_prompt = """Based on the following context items, please answer the query.
Give yourself room to think by extracting relevant passages from the context before answering the query.
Don't return the thinking, only return the answer.
Make sure your answers are as explanatory as possible.
Use the following examples as reference for the ideal answer style.
\nExample 1:
Query: How is the performance of Nvidia in 2023?
Answer: NVIDIA had a strong performance in various sectors throughout the year.
\nExample 2:
Query: How is the performance of Moderna in 2023?
Answer: In 2023, Moderna's performance can be considered robust in several key aspects, despite a significant decline in its stock price.
\nExample 3:
Query: Whether a company with a significant decline in profits and revenue will have a good market prospect?
Answer: Generally speaking, a significant decline in profits and revenue can signal challenges for a company's market prospects. However, market prospects are not solely determined by current financial metrics. 
User query: {query}
Answer:"""

    # Update base prompt with context items and query
    base_prompt = base_prompt.format(query=query)

    # Create prompt template for instruction-tuned model
    dialogue_template = [
        {"role": "user",
        "content": base_prompt}
    ]

    # Apply the chat template
    prompt = tokenizer.apply_chat_template(conversation=dialogue_template,
                                          tokenize=False,
                                          add_generation_prompt=True)
    return prompt