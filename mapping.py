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
    translated_words = tokenizer.decode(translation[0], skip_special_tokens=True)
    src_words = tokenizer.decode(src_tokens[0], skip_special_tokens=True).split()
    target_words = target_sentence.split()

    # Print word-to-word mapping
    print("Source Sentence:", src_words)
    print("Target Sentence:", target_words)
    print("Translated Sentence:", translated_words.split())

    # Print word-to-word mapping
    print("\nWord-to-Word Translation Mapping:")
    for src_word, target_word in zip(src_words, translated_words.split()):
        print(f"{src_word} -> {target_word}")

# Example usage
# src_sentence = "Hello, how are you?"
# target_sentence = "Hallo, wie geht es dir?"

# src_sentence = "Top 10 new game releases in 2022"
# target_sentence = "Top 10 neues Spiel Veröffentlichungen im Jahr 2022"

src_sentence = "I will buy a washing machine tomorrow."
target_sentence = "Ich werde morgen eine Waschmaschine kaufen."

model_name = "Helsinki-NLP/opus-mt-en-de"

analyze_translation(src_sentence, target_sentence, model_name)
