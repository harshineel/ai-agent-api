from dotenv import load_dotenv
load_dotenv()

import os
import re
import logging
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional
from groq import AsyncGroq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Agent API", version="1.0.0")

client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

class SolveRequest(BaseModel):
    query: str = Field(..., description="The question to answer")
    assets: Optional[List[str]] = Field(default=[], description="Asset URLs")

class SolveResponse(BaseModel):
    output: str

def handle_number_list(query: str):
    q = query.lower()
    numbers = re.findall(r'-?\d+\.?\d*', query)
    if not numbers:
        return None
    nums = [float(n) for n in numbers]

    if 'sum even' in q or 'sum of even' in q:
        result = sum(n for n in nums if n % 2 == 0)
        return str(int(result) if result.is_integer() else result)
    if 'sum odd' in q or 'sum of odd' in q:
        result = sum(n for n in nums if n % 2 != 0)
        return str(int(result) if result.is_integer() else result)
    if 'count even' in q or 'how many even' in q:
        return str(len([n for n in nums if n % 2 == 0]))
    if 'count odd' in q or 'how many odd' in q:
        return str(len([n for n in nums if n % 2 != 0]))
    if 'average' in q or 'mean' in q:
        result = sum(nums) / len(nums)
        return str(int(result) if result.is_integer() else round(result, 2))
    if 'max' in q or 'largest' in q or 'greatest' in q or 'highest' in q or 'most' in q:
        return str(int(max(nums)) if max(nums).is_integer() else max(nums))
    if 'min' in q or 'smallest' in q or 'lowest' in q or 'least' in q:
        return str(int(min(nums)) if min(nums).is_integer() else min(nums))
    if 'sum' in q or 'total' in q:
        result = sum(nums)
        return str(int(result) if result.is_integer() else result)
    return None

def handle_comparison(query: str):
    q = query.lower()
    # Pattern: "X scored N, Y scored M. Who scored highest/lowest?"
    pattern = r'([A-Z][a-z]+)\s+(?:scored|has|got|earned|received)\s+(\d+\.?\d*)'
    matches = re.findall(pattern, query)
    if matches and len(matches) >= 2:
        if any(w in q for w in ['highest', 'most', 'max', 'greatest', 'best', 'top']):
            winner = max(matches, key=lambda x: float(x[1]))
            return winner[0]
        if any(w in q for w in ['lowest', 'least', 'min', 'smallest', 'worst', 'bottom']):
            winner = min(matches, key=lambda x: float(x[1]))
            return winner[0]
    return None

@app.get("/")
async def health_check():
    return {"status": "ok"}

@app.post("/solve", response_model=SolveResponse)
async def solve_problem(request: SolveRequest):
    try:
        query = request.query
        q = query.lower()

        # Handle comparison questions (who scored highest etc)
        comparison_result = handle_comparison(query)
        if comparison_result:
            return SolveResponse(output=comparison_result)

        # Handle number list operations
        number_result = handle_number_list(query)
        if number_result:
            return SolveResponse(output=number_result)

        # Detect question type for LLM
        if q.startswith(('is ', 'are ', 'was ', 'were ', 'do ', 'does ',
                          'did ', 'can ', 'could ', 'will ', 'would ',
                          'has ', 'have ', 'had ')):
            system_prompt = "Answer ONLY with 'YES' or 'NO' in capitals. Nothing else."

        elif any(w in q for w in ['extract', 'find the date', 'what is the date']):
            system_prompt = "Extract and return ONLY the requested value. No extra words."

        elif 'who' in q and any(w in q for w in ['highest', 'lowest', 'most', 'least', 'best', 'worst', 'scored', 'earned']):
            system_prompt = "Return ONLY the person's name. No extra words."

        else:
            system_prompt = """Answer in one short sentence ending with a period.
For math: 'The sum is 25.'
For facts: 'The capital of France is Paris.'
No extra explanation."""

        response = await client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0,
            max_tokens=50
        )

        content = response.choices[0].message.content or "Could not process accurately"
        return SolveResponse(output=content.strip())

    except Exception as e:
        logger.error(f"Error: {type(e).__name__}: {e}")
        return SolveResponse(output="Could not process accurately")
