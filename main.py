from ollama import ollama
from property import Property


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    prop = Property()
    prop.generate_property()
    ollama.generate_description(prop)
    print(prop.description)
