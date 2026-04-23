import ollama

response = ollama.chat(model = "phi3", messages=[
    {
        'role': 'user',
        'content': "what is the capital of france"
    }
])

print(response['message']['content'])