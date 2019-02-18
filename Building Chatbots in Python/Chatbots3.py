import re
import spacy
import numpy as np
import sqlite3

#%%
import sqlite3
conn = sqlite3.connect('hotels.db')
cursor = conn.cursor()


print((cursor.execute("SELECT name from hotels where price = 'expensive' AND area = 'center'")).fetchall())

print((cursor.execute("SELECT name from hotels where price = 'mid' AND area = 'north'")).fetchall())

print((cursor.execute("SELECT name from hotels where price = 'expensive'")).fetchall())

#%%

cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print(cursor.fetchall())

#%%

# Import sqlite3
import sqlite3

# Open connection to DB
conn = sqlite3.connect('hotels.db')

# Create a cursor
c = conn.cursor()

# Define area and price
area, price = "south", "hi"
t = (area, price)

# Execute the query
c.execute('SELECT * FROM hotels WHERE area=? AND price=?', t)

# Print the results
print(c.fetchall())


#%%

# Define find_hotels()
def find_hotels(params):
    # Create the base query
    query = 'SELECT * FROM hotels'
    # Add filter clauses for each of the parameters
    if len(params) > 0:
        filters = ["{}=?".format(k) for k in params]
        query += " WHERE " + " and ".join(filters)
    # Create the tuple of values
    t = tuple(params.values())
    
    # Open connection to DB
    conn = sqlite3.connect("hotels.db")
    # Create a cursor
    c = conn.cursor()
    # Execute the query
    c.execute(query,t)
    # Return the results
    return c.fetchall()

#%%
    
# Create the dictionary of column names and values
params = {"area": "south", "price": "lo"}

# Find the hotels that match the parameters
print(find_hotels(params))

#%%
entities=[{'end': 19,
           'entity': 'price',
           'extractor': 'ner_crf',
           'processors': ['ner_synonyms'],
           'start': 10,
           'value': 'hi'},
            {'end': 38,
            'entity': 'area',
            'extractor': 'ner_crf',
            'start': 33,
            'value': 'south'}]

responses=["I'm sorry :( I couldn't find anything like that",
            '{} is a great hotel!',
            '{} or {} would work!',
            '{} is one option, but I know others too :)']

#%%

# Define respond()
def respond(message):
    # Extract the entities
#    entities = interpreter.parse(message)["entities"]
    # Initialize an empty params dictionary
    params = {}
    # Fill the dictionary with entities
    for ent in entities:
        params[ent["entity"]] = str(ent["value"])

    # Find hotels that match the dictionary
    results = find_hotels(params)
    # Get the names of the hotels and index of the response
    names = [r[0] for r in results]
    n = min(len(results),3)
    # Select the nth element of the responses array
    return responses[n].format(*names)

print(respond("I want an expensive hotel in the south of town"))


#%%

entities=[{'end': 19,
           'entity': 'price',
           'extractor': 'ner_crf',
           'processors': ['ner_synonyms'],
           'start': 10,
           'value': 'hi'}]
    
# Define a respond function, taking the message and existing params as input
def respond(message,params):
    # Extract the entities
#    entities = interpreter.parse(message)["entities"]
    # Fill the dictionary with entities
    for ent in entities:
        params[ent["entity"]] = str(ent["value"])

    # Find the hotels
    results = find_hotels(params)
    names = [r[0] for r in results]
    n = min(len(results), 3)
    # Return the appropriate response
    return responses[n].format(*names), params

# Initialize params dictionary
params = {}

# Pass the messages to the bot
for message in ["I want an expensive hotel", "in the north of town"]:
    print("USER: {}".format(message))
    response, params = respond(message, params)
    print("BOT: {}".format(response))
    
    entities=[{'end': 12,
          'entity': 'area',
          'extractor': 'ner_crf',
          'start': 7,
          'value': 'north'}]

#%%

tests=[("no I don't want to be in the south", {'south': False}),
       ('no it should be in the south', {'south': True}),
       ('no in the south not the north', {'north': False, 'south': True}),
       ('not north', {'north': False})]
    
#%%
    
# Define negated_ents()
def negated_ents(phrase):
    # Extract the entities using keyword matching
    ents = [e for e in ["south", "north"] if e in phrase]
    # Find the index of the final character of each entity
    ends = sorted([phrase.index(e) + len(e) for e in ents])
    # Initialise a list to store sentence chunks
    chunks = []
    # Take slices of the sentence up to and including each entitiy
    start = 0
    for end in ends:
        chunks.append(phrase[start:end])
        start = end
    result = {}
    # Iterate over the chunks and look for entities
    for chunk in chunks:
        for ent in ents:
            if ent in chunk:
                # If the entity is preceeded by a negation, give it the key False
                if "not" in chunk or "n't" in chunk:
                    result[ent] = False
                else:
                    result[ent] = True
    return result  

# Check that the entities are correctly assigned as True or False
for test in tests:
    print(negated_ents(test[0]) == test[1])

#%%

def negated_ents(phrase, ent_vals):
    ents = [e for e in ent_vals if e in phrase]
    ends = sorted([phrase.index(e)+len(e) for e in ents])
    start = 0
    chunks = []
    for end in ends:
        chunks.append(phrase[start:end])
        start = end
    result = {}
    for chunk in chunks:
        for ent in ents:
            if ent in chunk:
                if "not" in chunk or "n't" in chunk:
                    result[ent] = False
                else:
                    result[ent] = True
    return result

def find_hotels(params, neg_params):
    query = 'SELECT * FROM hotels'
    if len(params) > 0:
        filters = ["{}=?".format(k) for k in params] + ["{}!=?".format(k) for k in neg_params] 
        query += " WHERE " + " and ".join(filters)
    t = tuple(params.values())
    
    # open connection to DB
    conn = sqlite3.connect('hotels.db')
    # create a cursor
    c = conn.cursor()
    c.execute(query, t)
    return c.fetchall()

#%%

entities=[{'end': 14,
          'entity': 'price',
          'extractor': 'ner_crf',
          'processors': ['ner_synonyms'],
          'start': 9,
          'value': 'lo'}]

# Define the respond function
def respond(message,params,neg_params):
    # Extract the entities
#    entities = interpreter.parse(message)["entities"]
    ent_vals = [e["value"] for e in entities]
    # Look for negated entities
    negated = negated_ents(message,ent_vals)
    for ent in entities:
        if ent["value"] in negated and negated[ent["value"]]:
            neg_params[ent["entity"]] = str(ent["value"])
        else:
            params[ent["entity"]] = str(ent["value"])
    # Find the hotels
    results = find_hotels(params,neg_params)
    names = [r[0] for r in results]
    n = min(len(results),3)
    # Return the correct response
    return responses[n].format(*names), params, neg_params

# Initialize params and neg_params
params = {}
neg_params = {}

# Pass the messages to the bot
for message in ["I want a cheap hotel", "but not in the north of town"]:
    print("USER: {}".format(message))
    response, params, neg_params = respond(message, params, neg_params)
    print("BOT: {}".format(response))
    
    entities=[{'end': 20,
               'entity': 'area',
               'extractor': 'ner_crf',
               'start': 15,
               'value': 'north'}]



