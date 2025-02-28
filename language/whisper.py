from openai import OpenAI

print("What would you like to put into chatgpt")
user_input = input()

client = OpenAI(
    api_key = "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"
)

completion = client.chat.completions.create(
    model="gpt-4o-mini",
    store=True,
    messages=[
        {"role": "user", "content": user_input}
    ]
)

print(completion.choices[0].message.content)