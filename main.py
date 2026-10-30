import os, json, time
from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq,APITimeoutError, APIError
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb

load_dotenv()
app = FastAPI()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

print("Loading RAG...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="rsd_knowledge")
print(f"RAG ready, {collection.count()} chunks loaded")

# --- YOUR BUSINESS DATA ---
def get_shifting_rate(bhk: str):
    """This is your TOOL - just a normal function"""
    rates = {"1BHK": 3500, "2BHK": 6000, "3BHK": 9000, "1RK": 2500}
    return {"bhk": bhk, "rate": rates.get(bhk.upper(), 4000), "note": "Chennai local"}

# --- TOOL DESCRIPTION FOR AI ---
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_shifting_rate",
            "description": "Get exact shifting rate from RSD Packers rate card. Use when user asks price, rate, cost, charge, kitna lagega for any BHK.",
            "parameters": {
                "type": "object",
                "properties": {
                    "bhk": {"type": "string", "description": "1BHK, 2BHK, 3BHK, 1RK etc"}
                },
                "required": ["bhk"]
            }
        }
    }
]

# --- RAM PER USER ---
chat_histories = {}

class ChatRequest(BaseModel):
    message: str
    user_id: str


def call_groq(messages, tools_arg=None):
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            print(f"Trying Groq... attempt {attempt+1}")
            if tools_arg:
                return client.chat.completions.create(
                    model="openai/gpt-oss-20b",
                    messages=messages,
                    tools=tools_arg,
                    tool_choice="auto",
                    temperature=0.2,
                    max_tokens=300,
                    timeout=10
                )
            else:
                return client.chat.completions.create(
                    model="openai/gpt-oss-20b",
                    messages=messages,
                    temperature=0.2,
                    max_tokens=300,
                    timeout=10
                )
        
        except APITimeoutError:
            print(f"Timeout on attempt {attempt+1}, will retry")
            if attempt < max_retries - 1:
                wait = 2 ** attempt  # 1 sec, 2 sec, 4 sec
                print(f"Waiting {wait} sec before retry...")
                time.sleep(wait)
            else:
                raise  # last attempt failed, give up
                
        except APIError as e:
            print(f"API Error: {e}")
            # If rate limit, retry. If model not found, don't retry
            if "rate_limit" in str(e).lower() and attempt < max_retries - 1:
                time.sleep(2)
                continue
            else:
                raise

@app.post("/chat")
def chat(req: ChatRequest):
    start = time.time()

    if req.user_id not in chat_histories:
        chat_histories[req.user_id] = []
    history = chat_histories[req.user_id]

    # 1. CONTEXT WINDOW - keep last 10 only
    recent_history = history[-10:]

    messages = [
        {"role": "system", "content": "You are RSD Packers assistant Chennai. Use tool get_shifting_rate for any price query. Be friendly. If out of context, say: Sorry I only help with Packers queries."},
    ] + recent_history + [
        {"role": "user", "content": req.message}
    ]

    # 2. TEMPERATURE + TOOL LOOP - First call
    res1 = call_groq(messages, tools)

    # 3. TOOL CALLING LOOP
    if res1.choices[0].message.tool_calls:
        tool_call = res1.choices[0].message.tool_calls[0]
        func_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        print(f"--- AI decided to call tool: {func_name} with {args} ---")

        # YOUR Python function actually runs
        if func_name == "get_shifting_rate":
            tool_result = get_shifting_rate(**args)

        # Send tool result back to AI
        messages.append(res1.choices[0].message)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(tool_result)
        })

        # Second call to get final reply
        res2 = call_groq(messages)
        final_res = res2
    else:
        final_res = res1

    ai_reply = final_res.choices[0].message.content

    # 4. TOKENS + COST + LATENCY
    tokens_used = final_res.usage.total_tokens
    latency = round(time.time() - start, 2)
    print(f"Tokens: {tokens_used} | Latency: {latency}s | Cost: ~${tokens_used * 0.00000005}")

    # Save to memory
    history.append({"role": "user", "content": req.message})
    history.append({"role": "assistant", "content": ai_reply})
    chat_histories[req.user_id] = history[-10:]

    return {"reply": ai_reply, "tokens": tokens_used, "latency": latency}

# Run: uvicorn main:app --reload