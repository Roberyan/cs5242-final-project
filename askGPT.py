import os
from openai import OpenAI
def askGPT(message):
    os.environ["OPENAI_API_KEY"] = "sk-*******************"

    client = OpenAI()

    completion = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[
        {"role": "system", "content": "You are a good doctor and AI expert, skilled in explaining complex problems with health care."},
        {"role": "user", "content": "{:s}".format(message)},
      ]
    )

    return str(completion.choices[0].message.content)