import asyncio
from src.classification.guardrails import check_input_safety

async def main():
    status, msg = await check_input_safety("hi how things are going?")
    print(f"Status: {status}")
    print(f"Message: {msg}")

if __name__ == "__main__":
    asyncio.run(main())
