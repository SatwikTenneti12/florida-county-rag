import asyncio
from fastapi.testclient import TestClient
from api import app, check_input_safety
import json

client = TestClient(app)

def test_ask():
    # Login as existing user or create one
    resp = client.post("/api/signup", json={
        "name": "Test", "email": "test4@example.com", "password": "password", "captcha_token": "mock"
    })
    
    import sqlite3
    conn = sqlite3.connect('data/auth.db')
    c = conn.cursor()
    c.execute("UPDATE users SET is_verified = 1 WHERE email = 'test4@example.com'")
    conn.commit()
    conn.close()

    resp = client.post("/api/login", json={
        "email": "test4@example.com", "password": "password", "captcha_token": "mock"
    })
    token = resp.json()["access_token"]
    
    # Trigger 500
    try:
        resp2 = client.post(
            "/api/ask", 
            json={"question": "hi how things are going?", "top_k": 8},
            headers={"Authorization": f"Bearer {token}"}
        )
        print(f"Status: {resp2.status_code}")
        print(resp2.text)
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ask()
