from langchain_groq import ChatGroq
import os
from openai import OpenAI
api_key=os.environ['OPENAI_API_KEY']
client = OpenAI(api_key=api_key)

def llm_api(prompt, model, api_key):
    if "gpt" in model:
        if not isinstance(prompt, str):
            prompt_new = ""
            for i in prompt.messages:
                prompt_new += i.content
        else:
            prompt_new = prompt
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt_new},
            ]
        )
        with open("output.txt", "a") as f:
            f.write("Prompt tokens: " + str(response.usage.prompt_tokens) + "\n")
            f.write("Completion tokens: " + str(response.usage.completion_tokens) + "\n")
        return response.choices[0].message.content
    return ChatGroq(model="llama3-70b-8192", api_key=api_key)
