from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:11434/v1/',
    api_key='ollama', # ключ не нужен, но поле не должно быть пустым
)

response = client.chat.completions.create(
    model="gpt-oss:20b-cloud",
    messages=[{"role": "user", "content": "Привет!"}]
)
print(response.choices[0].message.content)