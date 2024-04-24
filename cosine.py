import torch
from transformers import MarianTokenizer, MarianMTModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

def analyze_translation(src_sentence, target_sentence, model_name):
    # Load tokenizer and model
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    # print special tokens for the model
    print("special tokens: ", tokenizer.special_tokens_map)
    # print tokenized input_ids of special tokens
    print("special tokens input_ids: ", tokenizer.convert_tokens_to_ids(tokenizer.special_tokens_map.values()))

    # Tokenize source and target sentences
    src_tokens = tokenizer(src_sentence, return_tensors="pt").input_ids
    print("src_sentence:  ", src_sentence)
    print("src_tokens", src_tokens)
    target_tokens = tokenizer(target_sentence, return_tensors="pt").input_ids
    print("target_sentence:  ", target_sentence)
    print("target_tokens", target_tokens)

    # Generate embeddings for source and target tokens
    with torch.no_grad():
        src_outputs = model.base_model(src_tokens, decoder_input_ids=target_tokens)
        target_outputs = model.base_model(target_tokens, decoder_input_ids=src_tokens)

        src_embeddings = src_outputs.last_hidden_state[0, :, :].numpy()
        target_embeddings = target_outputs.last_hidden_state[0, :, :].numpy()

    # print shapes of embeddings
    print("src_embeddings shape: ", src_embeddings.shape)
    print("target_embeddings shape: ", target_embeddings.shape)
    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(src_embeddings, target_embeddings)
    # print("similarity_matrix", similarity_matrix)
    # print similarity_matrix as a dataframe
    df = pd.DataFrame(similarity_matrix)
    columns = [tokenizer.decode(src_tokens[0][i].item(), skip_special_tokens=False) for i in range(len(src_tokens[0]))]
    df.columns = columns
    df.index = [tokenizer.decode(target_tokens[0][i].item(), skip_special_tokens=False) for i in range(len(target_tokens[0]))]
    # df.index = tokenizer.decode(target_tokens[0], skip_special_tokens=False).split()
    print(df)


    # Find best matching tokens based on cosine similarity
    src_indices = np.argmax(similarity_matrix, axis=1)
    target_indices = np.arange(len(target_tokens[0]))

    # Decode tokens to words
    src_words = tokenizer.decode(src_tokens[0], skip_special_tokens=True).split()
    target_words = tokenizer.decode(target_tokens[0], skip_special_tokens=True).split()

    # Print word-to-word mapping based on cosine similarity
    print("Source Sentence:", src_words)
    print("Target Sentence:", target_words)
    print("\nCosine Similarity-based Word-to-Word Mapping:")
    
    for src_index, target_index in enumerate(src_indices):
        src_word = src_words[src_index]
        target_word = target_words[target_index]
        print(f"{src_word} -> {target_word}")

# Example usage
src_sentence = "I will buy a washing machine tomorrow"
target_sentence = "Ich werde morgen eine Waschmaschine kaufen"
model_name = "Helsinki-NLP/opus-mt-en-de"

analyze_translation(src_sentence, target_sentence, model_name)
