import markovify

def generate_text(data, classification, text_type, num_out):
    # Specify which type of text to generate
    if classification == "sarcastic":
        sub_data = data[data['is_sarcastic'] == 1]
    elif classification == "non sarcastic":
        sub_data = data[data['is_sarcastic'] == 0]

    # Specify model type
    combined = sub_data.tokenized.str.cat(sep='\n')

    text_model = markovify.NewlineText(combined)

    # Generate and print output
    for i in range(num_out):
        if text_type == "headline":
            print(text_model.make_sentence())
        elif text_type == "tweet":
            print(text_model.make_short_sentence(140))