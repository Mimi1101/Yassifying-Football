import json
import ollama


def format_team_stats_for_prompt(team_data, team_label):
    with open("explain_like_five.json", "r", encoding="utf-8") as f:
        stats = json.load(f)
    output = f"📊 {team_label} ({team_data['team_name']}):\n"
    for key, value in team_data['stats'].items():
        if value is not None:
            output += f"- {key}: {value}\n"
    return output.strip()

with open("explain_like_five.txt", "r", encoding="utf-8") as f:
    team_stats = f.read()

prompt = """
You are writing a stat focused explainer for a football match. Your job is to go through each stat and explain, in a clear and engaging way, 
what the number means and what it tells us about how the game played out. This is NOT a match summary or narrative, just a breakdown of the stats, one by one. 
Make it enjoyable for both football fans, casual fans, and people who are new to the sport, no jargon without explanation, but also no oversimplifying. 
Write it like a confident, knowledgeable football analyst who knows how to make numbers interesting without sounding like a robot or a sportscaster.
Cover every stat listed below. You may compare the two teams and highlight interesting differences. Do not invent anything beyond what's in the stats.

Match Stats:
{team_stats}
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

print("Explain like I am five:\n")
print(response['message']['content'])




