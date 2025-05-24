import os
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from groq import Groq

# Initialize Groq client
os.environ['GROQ_API_KEY'] = "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"
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

class Agent:
    def __init__(self, client: Groq, system: str = "") -> None:
        self.client = client
        self.system = system
        self.messages: list = []
        if self.system:
            self.messages.append({"role": "system", "content": system})

    def __call__(self, message=""):
        if message:
            self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result

    def execute(self):
        completion = self.client.chat.completions.create(
            model="llama3-70b-8192",
            messages=self.messages
        )
        return completion.choices[0].message.content

def class_words(frase):
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    tokens = word_tokenize(frase)
    tags = pos_tag(tokens)
    return str(tags)

def loop():
    system_prompt = """
    You run in a loop of Thought, Action, PAUSE, Observation.
    At the end of the loop you output an Answer.
    Use Thought to describe your thoughts about the question you have been asked.
    Use Action to run one of the actions available to you - then return PAUSE.
    Observation will be the result of running those actions.

    Your available actions are:
    class_words:
    e.g. class_words(frase)
    returns a list with the words and their grammatical class

    Example session:
    Frase: The black cat jumped over the tall brick wall.
    Thought: I need to analyze the grammatical structure of this sentence.
    Action: class_words(The black cat jumped over the tall brick wall)
    PAUSE

    Observation: [('The', 'DT'), ('black', 'JJ'), ('cat', 'NN'), ('jumped', 'VBD'), ('over', 'IN'), ('the', 'DT'), ('tall', 'JJ'), ('brick', 'NN'), ('wall', 'NN')]

    Answer: Here are the grammatical classifications:
    [('The', 'DT'), ('black', 'JJ'), ('cat', 'NN'), ('jumped', 'VBD'), ('over', 'IN'), ('the', 'DT'), ('tall', 'JJ'), ('brick', 'NN'), ('wall', 'NN')]
    """.strip()

    agent = Agent(client, system_prompt)
    print("Welcome to the Grammatical Classifier Agent!")
    print("Enter a sentence to analyze or 'quit' to exit.")

    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
            
        response = agent(f"Analyze this sentence: {user_input}")
        
        while True:
            print(f"\nAgent: {response}")
            
            if "PAUSE" in response:
                # Extract action more robustly
                action_line = next((line for line in response.split('\n') if line.startswith("Action:")), None)
                if action_line:
                    action = action_line.split("Action:")[1].strip()
                    if action.startswith("class_words("):
                        frase = action[len("class_words("):-1].strip()
                        if not frase:  # Use original input if extraction fails
                            frase = user_input
                        observation = class_words(frase)
                        response = agent(f"Observation: {observation}")
                    else:
                        response = agent("Observation: Unknown action")
                else:
                    response = agent("Observation: No action specified")
            
            elif "Answer:" in response:
                answer = response.split("Answer:")[1].strip()
                print(f"\nFinal Answer: {answer}")
                break
            
            else:
                response = agent()

if __name__ == "__main__":
    loop()