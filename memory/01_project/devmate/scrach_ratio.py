import tiktoken

enc = tiktoken.get_encoding("cl100k_base")
samples = {
    "english": "I'm building an e-commerce application using React, FastAPI and PostgreSQL.",
    "python": "def add_to_cart(user_id: int, item_id: int) -> dict:\n    return {'ok': True, 'items': [item_id]}\n",
    "json": '{"id": "msg_001", "role": "user", "token_count": 12, "tags": ["db", "auth"]}',
    "hindi": "मैं React, FastAPI और PostgreSQL का उपयोग करके एक ई-कॉमर्स एप्लिकेशन बना रहा हूँ।",
}
for name, text in samples.items():
    real = len(enc.encode(text))
    guess = len(text) // 4
    print(f"{name:8} chars={len(text):3}  real={real:3}  len//4={guess:3}  chars/token={len(text)/real:.2f}")
