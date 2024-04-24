from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def analyze_translation(src_sentence, target_sentence, model_name):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Tokenize source and target sentences
    src_tokens = tokenizer(src_sentence, return_tensors="pt").input_ids
    target_tokens = tokenizer(target_sentence, return_tensors="pt").input_ids

    # Generate translation
    with tokenizer.as_target_tokenizer():
        target_input = tokenizer(target_sentence, return_tensors="pt").input_ids

    translation = model.generate(src_tokens, decoder_input_ids=target_input)

    # Decode the generated tokens back to words
    translated_words = tokenizer.decode(translation[0], skip_special_tokens=True).split()
    src_words = tokenizer.decode(src_tokens[0], skip_special_tokens=True).split()
    target_words = target_sentence.split()

    # Print word-to-word mapping
    print("Source Sentence:", src_words)
    print("Target Sentence:", target_words)
    print("Translated Sentence:", translated_words)

    print("\nApproximate Word-to-Word Translation Mapping:")
    
    src_index = 0
    target_index = 0
    while src_index < len(src_words) and target_index < len(translated_words):
        src_word = src_words[src_index]
        target_word = translated_words[target_index]
        
        print(f"{src_word} -> {target_word}")
        
        # Move to the next source word
        src_index += 1
        
        # If the target word is a part of the source word, move to the next target word
        while not src_word.startswith(target_word) and target_index < len(translated_words) - 1:
            target_index += 1
            target_word += translated_words[target_index]
        
        # Move to the next target word
        target_index += 1

# Example usage
src_sentence = "I will buy a washing machine tomorrow."
target_sentence = "Ich werde morgen eine Waschmaschine kaufen."
model_name = "Helsinki-NLP/opus-mt-en-de"

analyze_translation(src_sentence, target_sentence, model_name)
