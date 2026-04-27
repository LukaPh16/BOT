import ollama

response = ollama.chat(model = "qwen2.5:0.5b", messages=[
    {
        'role': 'user',
        'content': ""
    }
])

print(response['message']['content'])