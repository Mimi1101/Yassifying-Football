import ollama

# Read match data from file
with open("llm_context.txt", "r", encoding="utf-8") as f:
    match_data = f.read()

# Read Gen-Z slangs from file
with open("genz_slang.txt", "r", encoding="utf-8") as f:
    genz_slangs = f.read()


prompt = """
You're a Gen-Z football commentator. Generate a post-match summary using the match data below and the example Gen-Z slang.
Make it fun, sharp, and engaging like a TikTok or Instagram Reel voiceover. Use the provided Gen-Z slangs when they match the moment.
Feel free to include your own Gen-Z expressions if they fit the tone naturally.
**Stick strictly to the stats and facts in the match data. Do NOT make up goals, assists, or cards.**
Only reference things that are explicitly in the data. Use slang meaningfully, not randomly.

Make sure to include goals and assists from both teams.
Match Data:
{match_data}
Gen-Z Slangs (examples with meaning):
{genz_slangs}
"""



response = ollama.chat(
    model='llama3',
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

print("\nGen-Z Match Summary:\n")
print(response['message']['content'])
