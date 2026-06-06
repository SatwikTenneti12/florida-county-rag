import requests
resp2 = requests.post(
    "http://localhost:8000/api/ask", 
    json={"question": "hi how things are going?"},
    headers={"Authorization": f"Bearer $(curl -s -X POST http://localhost:8000/api/login -d '{\"email\": \"test3@example.com\", \"password\": \"password\", \"captcha_token\": \"mock\"}' -H 'Content-Type: application/json' | grep -o '\"access_token\":\"[^\"]*' | cut -d'\"' -f4)"},
)
print(resp2.text)
