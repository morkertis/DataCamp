import re
import spacy
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

#%%
bot_template = "BOT : {0}"
user_template = "USER : {0}"

def send_message(message):
    print(user_template.format(message))
    response = respond(message)
    print(bot_template.format(response))

#%%
keywords={'goodbye': ['bye', 'farewell'],
          'greet': ['hello', 'hi', 'hey'],
          'thankyou': ['thank', 'thx']}

#%%


# Define a dictionary of patterns
patterns = {}

# Iterate over the keywords dictionary
for intent, keys  in keywords.items():
    # Create regular expressions and compile them into pattern objects
    patterns[intent] = re.compile('|'.join(keys))
    
# Print the patterns
print(patterns)

#%%

responses={'default': 'default message',
           'goodbye': 'goodbye for now',
           'greet': 'Hello you! :)',
            'thankyou': 'you are very welcome'}

#%%
# Define a function to find the intent of a message
def match_intent(message):
    matched_intent = None
    for intent, pattern in patterns.items():
        # Check if the pattern occurs in the message 
        if pattern.search(message):
            matched_intent = intent
    return matched_intent

# Define a respond function
def respond(message):
    # Call the match_intent function
    intent = match_intent(message)
    # Fall back to the default response
    key = "default"
    if intent in responses:
        key = intent
    return responses[key]

# Send messages
send_message("hello!")
send_message("bye byeee")
send_message("thanks very much!")

#%%

# Define find_name()
def find_name(message):
    name = None
    # Create a pattern for checking if the keywords occur
    name_keyword = re.compile("name|call")
    # Create a pattern for finding capitalized words
    name_pattern = re.compile('[A-Z]{1}[a-z]+')
    if name_keyword.search(message):
        # Get the matching words in the string
        name_words = name_pattern.findall(message)
        if len(name_words) > 0:
            # Return the name if the keywords are present
            name = ' '.join(name_words)
    return name

# Define respond()
def respond(message):
    # Find the name
    name = find_name(message)
    if name is None:
        return "Hi there!"
    else:
        return "Hello, {0}!".format(name)

# Send messages
send_message("my name is David Copperfield")
send_message("call me Ishmael")
send_message("People call me Cassandra")

#%%
sentences=[' i want to fly from boston at 838 am and arrive in denver at 1110 in the morning',
           ' what flights are available from pittsburgh to baltimore on thursday morning',
           ' what is the arrival time in san francisco for the 755 am flight leaving washington',
           ' cheapest airfare from tacoma to orlando',
           ' round trip fares from pittsburgh to philadelphia under 1000 dollars',
           ' i need a flight tomorrow from columbus to minneapolis',
           ' what kind of aircraft is used on a flight from cleveland to dallas',
           ' show me the flights from pittsburgh to los angeles on thursday',
           ' all flights from boston to washington',
           ' what kind of ground transportation is available in denver',
           ' show me the flights from dallas to san francisco',
           ' show me the flights from san diego to newark by way of houston',
           ' what is the cheapest flight from boston to bwi',
           ' all flights to baltimore after 6 pm',
           ' show me the first class fares from boston to denver',
           ' show me the ground transportation in denver',
           ' all flights from denver to pittsburgh leaving after 6 pm and before 7 pm',
           ' i need information on flights for tuesday leaving baltimore for dallas dallas to boston and boston to baltimore',
           ' please give me the flights from boston to pittsburgh on thursday of next week',
           ' i would like to fly from denver to pittsburgh on united airlines',
           ' show me the flights from san diego to newark',
           ' please list all first class flights on united from denver to baltimore',
           ' what kinds of planes are used by american airlines',
           " i'd like to have some information on a ticket from denver to pittsburgh and atlanta",
           " i'd like to book a flight from atlanta to denver",
           ' which airline serves denver pittsburgh and atlanta',
           " show me all flights from boston to pittsburgh on wednesday of next week which leave boston after 2 o'clock pm",
           ' atlanta ground transportation',
           ' i also need service from dallas to boston arriving by noon',
           ' show me the cheapest round trip fare from baltimore to dallas']

#%%
# Load the spacy model: nlp
nlp = spacy.load('en_core_web_md')

n_sentences = len(sentences)

# Calculate the dimensionality of nlp
embedding_dim = nlp.vocab.vectors_length

# Initialize the array with zeros: X
X = np.zeros((n_sentences, embedding_dim))

## Iterate over the sentences
for idx, sentence in enumerate(sentences):
    # Pass each each sentence to the nlp object to create a document
    doc = nlp(sentence)
    # Save the document's .vector attribute to the corresponding row in X
    X[idx, :] = doc.vector


#%%

import pandas as pd
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df_train=pd.read_csv('atis/atis_intents_train.csv',header=None)
df_test=pd.read_csv('atis/atis_intents_test.csv',header=None)

X_train_text=df_train[1].values
y_train=le.fit_transform(df_train[0].values)

X_test_text=df_test[1].values
y_test=le.transform(df_test[0].values)

#%%

X_train = np.zeros((len(X_train_text), embedding_dim))
for idx, sentence in enumerate(X_train_text):
    # Pass each each sentence to the nlp object to create a document
    doc = nlp(sentence)
    # Save the document's .vector attribute to the corresponding row in X
    X_train[idx, :] = doc.vector

#%%

X_test = np.zeros((len(X_test_text), embedding_dim))
for idx, sentence in enumerate(X_test_text):
    # Pass each each sentence to the nlp object to create a document
    doc = nlp(sentence)
    # Save the document's .vector attribute to the corresponding row in X
    X_test[idx, :] = doc.vector

#%%

# Import SVC
from sklearn.svm import SVC

# Create a support vector classifier
clf = SVC()

# Fit the classifier using the training data
clf.fit(X_train,y_train)

# Predict the labels of the test set
y_pred = clf.predict(X_test)

# Count the number of correct predictions
n_correct = 0
for i in range(len(y_test)):
    if y_pred[i] == y_test[i]:
        n_correct += 1

print("Predicted {0} correctly out of {1} test examples".format(n_correct, len(y_test)))


#%%

# Define included entities
include_entities = ['DATE', 'ORG', 'PERSON']

# Define extract_entities()
def extract_entities(message):
    # Create a dict to hold the entities
    ents = dict.fromkeys(include_entities)
    # Create a spacy document
    doc = nlp(message)
    for ent in doc.ents:
        if ent.label_ in include_entities:
            # Save interesting entities
            ents[ent.label_] = ent.text
    return ents

print(extract_entities('friends called Mary who have worked at Google since 2010'))
print(extract_entities('people who graduated from MIT in 1999'))


#%%

colors = ['black', 'red', 'blue']
items = ['shoes', 'handback', 'jacket', 'jeans']

def entity_type(word):
    _type = None
    if word.text in colors:
        _type = "color"
    elif word.text in items:
        _type = "item"
    return _type

#%%
# Create the document
doc = nlp("let's see that jacket in red and some blue jeans")

# Iterate over parents in parse tree until an item entity is found
def find_parent_item(word):
    # Iterate over the word's ancestors
    for parent in word.ancestors:
        # Check for an "item" entity
        if entity_type(parent) == "item":
            return parent.text
    return None

# For all color entities, find their parent item
def assign_colors(doc):
    # Iterate over the document
    for word in doc:
        # Check for "color" entities
        if entity_type(word) == "color":
            # Find the parent
            item = find_parent_item(word) 
            print("item: {0} has color : {1}".format(item, word))

# Assign the colors
assign_colors(doc) 

#%%


