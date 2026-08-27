import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("❌ GROQ_API_KEY not found")
    print("Create a .env file with:")
    print("GROQ_API_KEY=your_api_key_here")
    exit()

client = Groq(api_key=api_key)

print("Testing Groq model...")

try:
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "user",
                "content": "Explain AgriPattern in one short sentence."
            }
        ],
        temperature=0.2,
        max_tokens=100,
    )

    print("\n✅ GROQ MODEL WORKING!")
    print("Response:")
    print(response.choices[0].message.content)

except Exception as e:
    print("\n❌ GROQ ERROR:")
    print(type(e).__name__)
    print(e)