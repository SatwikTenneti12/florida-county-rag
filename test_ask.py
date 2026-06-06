import requests

# 1. Signup to get user
resp = requests.post("http://localhost:8000/api/signup", json={
    "name": "Test",
    "email": "test2@example.com", 
    "password": "password", 
    "captcha_token": "mock"
})

import sqlite3
conn = sqlite3.connect('data/auth.db')
c = conn.cursor()
c.execute("UPDATE users SET is_verified = 1 WHERE email = 'test2@example.com'")
conn.commit()
conn.close()

# 2. Login
resp = requests.post("http://localhost:8000/api/login", json={
    "email": "test2@example.com", 
    "password": "password", 
    "captcha_token": "mock"
})
if resp.status_code != 200:
    print(f"Login failed: {resp.text}")
else:
    token = resp.json()["access_token"]
    
    # 3. Call /api/ask
    resp2 = requests.post(
        "http://localhost:8000/api/ask", 
        json={"question": "hi how things are going?"},
        headers={"Authorization": f"Bearer {token}"},
        stream=True
    )
    print(f"Ask Status: {resp2.status_code}")
    for line in resp2.iter_lines():
        if line:
            print(line.decode('utf-8'))
