import asyncio
from fastapi.testclient import TestClient
from api import app, check_input_safety

client = TestClient(app)

# We can directly invoke ask_question to see the exception
import traceback

def test():
    resp = requests.post("http://localhost:8000/api/signup", json={
        "name": "Test",
        "email": "test3@example.com", 
        "password": "password", 
        "captcha_token": "mock"
    })

    import sqlite3
    conn = sqlite3.connect('data/auth.db')
    c = conn.cursor()
    c.execute("UPDATE users SET is_verified = 1 WHERE email = 'test3@example.com'")
    conn.commit()
    conn.close()

    resp = requests.post("http://localhost:8000/api/login", json={
        "email": "test3@example.com", 
        "password": "password", 
        "captcha_token": "mock"
    })
    token = resp.json()["access_token"]
    
    resp2 = requests.post(
        "http://localhost:8000/api/ask", 
        json={"question": "hi how things are going?"},
        headers={"Authorization": f"Bearer {token}"},
        stream=True
    )
    print(resp2.text)

