import ollama

response = ollama.chat(model = "qwen2.5:0.5b", messages=[
    {
        'role': 'user',
        'content': "what is the capital of france"
    }
])

print(response['message']['content'])