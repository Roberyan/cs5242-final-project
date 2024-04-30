import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import huggingface_hub
import os

os.environ['HF_HOME'] = "******"
huggingface_hub.login("hf_*********************************")
from transformers import BitsAndBytesConfig


def get_llm(model_id, use_quantization_config=True):
    quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                             bnb_4bit_compute_dtype=torch.float16)


    model_id = model_id  # (we already set this above)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_id, cache_dir="./",
                                              trust_remote_code=True)

    llm_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_id,
                                                     torch_dtype=torch.float16,  # datatype to use, we want float16
                                                     quantization_config=quantization_config if use_quantization_config else None,
                                                     low_cpu_mem_usage=False,  # use full memory
                                                     trust_remote_code=True,
                                                     cache_dir="./")  # which attention version to use

    if not use_quantization_config:  # quantization takes care of device setting automatically, so if it's not used, send model to GPU
        llm_model.to("cuda")
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
    base_prompt = """Assuming you're an expert in artificial intelligence and health care, and you know a lot about AI, Programming Languages and medicine. Please answer the query.
        Give yourself room to read some passages which may be useful for answering the query. If it is not useful, please you are an expert, and you can answer the question by yourself.
        Don't return the thinking, only return the answer.
        If you can't answer the query based on the passages, please try to use what you have learned to answer the query.
        To be precise, the following text is just your assistance in answering questions, but do not rely entirely on it. If you have existing knowledge to answer, you can also try to answer as much as possible.
        \nNow Here are the passages which may be useful for you. If it is useless, try to answer the question by yourself:
        {context}Variable Spacing. Selectivity allows filtering out irrelevant noise tokens that may occur between inputs ofinterest. 'This is exemplified by the Selective Copying task, but occurs ubiquitously in common data modalitiesparticularly for discrete data for example the presence of language fillers such as "um". This property arisesbecause the model can mechanistically filter out any particular input x, for example in the gated RNN case(Theorem l)when g - 0.Filtering Context. It has been empirically observed that many sequence models do not improve with longercontext (F. Shi et al. 2023), despite the principle that more context should lead to strictly better performance. Anexplanation is that many sequence models cannot elfectively ignore irrelevant context when necessary; an intuitiveexample are global convolutions (and general l'I'l models). On the other hand, selective models can simply resettheir state at any time to remove extraneous history, and thus their performance in principle improves monotoniclwith context length (e.g. Section 4.3.2).Boundary Resetting. In settings where multiple independent sequences are stitched together, Transformerscan keep them separate by instantiating a particular attention mask, while lIi models will bleed informationbetween the sequences. Selective SSMls can also reset their state at boundaries (e.g. A,- co or 'Theorem l when8,- 1). 'These settings may occur artificially (e.g. packing documents together to improve hardware utilizationor naturally (e.g. episode boundaries in reinforcement learning (Lu et al. 2023)).Additionally, we elaborate on effects of each selective parameter.Interpretation of A. in general, A controls the balanee between how much to focus or ignore the current inputx,. lt generalizes RNN gates (e.g. g in 'T'heorem l), mechanically, a large A resets the state h and focuses on thecurrent input x, while a small A persists the state and ignores the current input. SSMs (1)-(2) can be interpreted asa continuous system discretized by a timestep A, and in this context the intuition is that large A -> co representsthe system focusing on the current input for longer (thus "seleeting" it and forgetting its current state) while asmall A-0 represents a transient input that is ignored.
        Please remember to answer questions as much as possible, even without referring to relevant paragraphs, you can rely on your own knowledge to answer questions.
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
    base_prompt = """
        Assuming you're an expert in artificial intelligence and health care.
        Don't return the thinking, only return the answer.
        Try to answer as much as possible, even if you are not very sure about the answer.
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