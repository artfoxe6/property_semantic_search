from fontTools.misc.cython import returns

from sbert import SentenceBert
from ollama import ollama
from property import Property
from vector_db import VectorDB


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    prop = Property()
    prop.generate_property()
    description = ollama.generate_description(prop)
    if description != "":
        prop.description = description

    vdb = VectorDB()
    vdb.create_collection()
    p_dict = prop.to_dict()
    p_dict["desc_vector"] = SentenceBert.text2vector(prop.description)
    vdb.upsert([p_dict])
