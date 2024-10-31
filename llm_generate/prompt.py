import os
import re
from openai import OpenAI
from ollama import Client

client = OpenAI(
    api_key="",  # Replace with a real DashScope API_KEY
    base_url="",  # Fill in the DashScope service endpoint
)

# Define the prompts without unnecessary instructions.
prompt_1 = "Now you are a software testing engineer."
prompt_2 = "Design a complete code generation plan. Generate test code with assertions and exception predictors."
prompt_3 = "Understand the method function, extract variables and function names."
prompt_4= "Please combine the mutated model network structure."
prompt_5 = "Reply with only the complete test oracle code."
prompt_6 = "Please add a third-party dependency package reference."


def mutatePromptGenerator(mutation_function):
    mutate_prompt = f"""{mutation_function}
# test case
def test():
    assertTrue
    assertFalse
    assertIsNotNone
    assertIsNone
    assertIs
    assertIsNot
    assertIsInstance
    assertNotIsInstance
"""
    return mutate_prompt

def get_mutant_content(mutant_path):
    abs_mutant_path = os.path.abspath(mutant_path)
    with open(abs_mutant_path, encoding='utf-8') as input_file:
        mutant_content = input_file.read()
    return mutant_content

def modelPromptGenerator(mutated_model):
    model_prompt = f"""{mutated_model}
"""
    return model_prompt

mutationimport = '''
from mut.guassian_fuzz_splayers import gaussian_fuzzing_splayer
from mut.random_shuffle import random_shuffle_weight
from mut.remove_activation import remove_activations
from mut.replace_activation import replace_activations
from mut.uniform_fuzz import uniform_fuzz_weight
'''

def mutationImportPrompt(mutationimport):
    return mutationimport

model_name = "qwen2-7b-instruct"
temperature = 0.7


def call_llms(model_name, messages, temperature):
    completion = client.chat.completions.create(
        seed=100,
        model=model_name,
        messages=messages,
        temperature=temperature,
        stream=False,
        top_p=0.7
    )

    generated_content = completion.choices[0].message.content

    # Remove markdown code block indicators
    generated_content = re.sub(r'```python\s*', '', generated_content)
    generated_content = re.sub(r'```\s*', '', generated_content)

    # Ensure there is an assertion in the output before returning.
    if not re.search('assert', generated_content):
        return call_llms(model_name, messages, temperature)

    return generated_content

def call_llms_ollama(model_name, messages, temperature):
    client = Client(host='')
    response = client.chat(
        model=model_name,
        messages=messages,
        options={
            "temperature": temperature
        }
    )

    generated_content = response['message']['content']

    # Remove markdown code block indicators
    generated_content = re.sub(r'```python\s*', '', generated_content)
    generated_content = re.sub(r'```\s*', '', generated_content)

    # Ensure there is an assertion in the output before returning.
    if not re.search('assert', generated_content):
        return call_llms_ollama(model_name, messages, temperature)

    return generated_content



