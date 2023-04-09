import markovify

def generate_text(data, classification, text_type, num_out):
    if classification == "sarcastic":
        sub_data = data[data['is_sarcastic'] == 1]
    elif classification == "non sarcastic":
        sub_data = data[data['is_sarcastic'] == 0]

    combined = sub_data.tokenized.str.cat(sep='\n')

    text_model = markovify.NewlineText(combined)

    for i in range(num_out):
        if text_type == "headline":
            print(text_model.make_sentence())
        elif text_type == "tweet":
            print(text_model.make_short_sentence(140))