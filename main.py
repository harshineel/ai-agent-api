import os
import re
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from groq import AsyncGroq

app = FastAPI()
client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

class SolveRequest(BaseModel):
    query: str
    assets: Optional[List[str]] = []

class SolveResponse(BaseModel):
    output: str

@app.post("/solve")
async def solve_problem(request: SolveRequest):
    try:
        # Step 1: Detect if it's a Yes/No question
        is_yes_no = re.match(r'^(Is|Are|Was|Were|Do|Does|Did|Can|Could|Will|Would|Has|Have|Had)', request.query.strip(), re.I)

        response = await client.chat.completions.create(
            model="llama-3.1-8b-instant", # Faster for better latency scores
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise answering engine. Output ONLY the raw value or answer. NO sentences. NO periods. NO explanations. If math, output ONLY the number."
                },
                {"role": "user", "content": request.query}
            ],
            temperature=0,
            max_tokens=20,
            stop=["\n", "."] # Force stop before any punctuation or extra lines
        )

        answer = response.choices[0].message.content.strip()

        # Step 2: Strict Post-Processing Logic
        
        # Force YES/NO format
        if is_yes_no:
            return SolveResponse(output="YES" if "YES" in answer.upper() else "NO")

        # If the output contains a number, and the query looks like math/counting
        # Extract just the number (e.g., "10" instead of "The sum is 10")
        if any(char.isdigit() for char in answer):
            # This regex finds the last number in the string (usually the answer)
            numbers = re.findall(r"[-+]?\d*\.\d+|\d+", answer)
            if numbers:
                return SolveResponse(output=numbers[-1])

        # For extraction or short factual answers, remove the trailing period
        return SolveResponse(output=answer.rstrip('.'))

    except Exception:
        return SolveResponse(output="Error processing request")