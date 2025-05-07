import nltk
nltk.download('punkt')
from textblob import TextBlob
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize, sent_tokenize


jar = "YOUR PATH TO stanford-ner.jar"
model = "YOUR PATH TO english.all.3class.distsim.crf.ser.gz" #Under classifiers folder

st = StanfordNERTagger(model, jar)

# test dataset type (for each question) [{question: question, human: therapist_response, base: base_response, fine_tuned: fine_tuned_response}]
dataset = [
    {"question": "How are you feeling today?", "human": "James, I'm feeling a bit down.", "base": "Bob, I am okay.", "fine_tuned": "I am feeling really sad"},
]

def analyze_responses(data):
    results = []
    for entry in data:
        human_response = entry['human']
        base_response = entry['base']
        fine_tuned_response = entry['fine_tuned']

        delta_base = len(human_response.split()) - len(base_response.split())
        delta_fine_tuned = len(human_response.split()) - len(fine_tuned_response.split())

        sentiment_human = TextBlob(human_response).sentiment.polarity
        sentiment_base = TextBlob(base_response).sentiment.polarity
        sentiment_fine_tuned = TextBlob(fine_tuned_response).sentiment.polarity

        def contains_name(text):
            for sent in sent_tokenize(text):
                tokens = word_tokenize(sent)
                tags = st.tag(tokens)
                for tag in tags:
                    if tag[1] == 'PERSON':
                        return True
            return False

        human_contains_name = contains_name(human_response)
        base_contains_name = contains_name(base_response)
        fine_tuned_contains_name = contains_name(fine_tuned_response)

        results.append({
            'question': entry['question'],
            'delta_base': delta_base,
            'delta_fine_tuned': delta_fine_tuned,
            'sentiment_human': sentiment_human,
            'sentiment_base': sentiment_base,
            'sentiment_fine_tuned': sentiment_fine_tuned,
            'human_contains_name': human_contains_name,
            'base_contains_name': base_contains_name,
            'fine_tuned_contains_name': fine_tuned_contains_name
        })
    return results

if __name__ == "__main__":
    results = analyze_responses(dataset)
    for result in results:
        print(result)

