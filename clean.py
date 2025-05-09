import json

def clean_json(data):
    if isinstance(data, dict):
        cleaned = {}
        for key, value in data.items():
            cleaned_value = clean_json(value)
            if cleaned_value in [None, {}, [], 0, 0.0, "0", "0.0"]:
                continue
            cleaned[key] = cleaned_value
        return cleaned if cleaned else None
    elif isinstance(data, list):
        cleaned_list = [clean_json(item) for item in data]
        cleaned_list = [item for item in cleaned_list if item not in [None, {}, [], 0, 0.0, "0", "0.0"]]
        return cleaned_list if cleaned_list else None
    else:
        return data
    


def summarize_player(p, team):
    name = p['name']
    pos = p['position']
    minutes = p['minutes']
    summary = f"- {name} ({team}, {pos}, {minutes} mins)"
    details = []

    if g := p.get("goals", {}):
        if "total" in g: details.append(f"scored {g['total']} goals")
        if "assists" in g: details.append(f"provided {g['assists']} assists")
        if "conceded" in g: details.append(f"conceded {g['conceded']} goals")
        if "saves" in g: details.append(f"made {g['saves']} saves")
    if p.get("assists"): details.append(f"provided {p['assists']} assists")

    passes = p.get("passes", {})
    if "total" in passes: details.append(f"{passes['total']} passes")
    if "key" in passes: details.append(f"{passes['key']} key passes")
    if "accuracy" in passes: details.append(f"{passes['accuracy']}% pass accuracy")

    duels = p.get("duels", {})
    if "total" in duels: details.append(f"{duels['total']} duels")
    if "won" in duels: details.append(f"{duels['won']} won")

    cards = p.get("cards", {})
    if "yellow" in cards: details.append(f"{cards['yellow']} yellow cards")
    if "red" in cards: details.append(f"{cards['red']} red cards")

    return summary + " – " + ", ".join(details)

def llm_prompt_readable():
    with open("summary_cleaned.json", "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    data = clean_json(raw_data)

    # Match Overview
    date = data["date"][:10]
    match_info = (
        f"Match Overview:\n"
        f"On {date}, Barcelona played at home against {data['opponent']} "
        f"in the {data['league']} Final ({data['season']} season). "
        f"The match ended in a {data['result']} for Barcelona "
        f"({data['barca_goals']}-{data['opponent_goals']}).\n"
    )

    # Team Stats
    barca = data["team_stats"]["barcelona"]["stats"]
    madrid = data["team_stats"]["opponent"]["stats"]
    all_keys = sorted(set(barca.keys()).union(madrid.keys()))
    team_stats = "Team Stats:\n"
    for key in all_keys:
        team_stats += f"- {key}: Barcelona {barca.get(key, 'N/A')} | Real Madrid {madrid.get(key, 'N/A')}\n"

    # Player Summaries
    players = data["players"]
    player_section = "🧑‍💼 Player Performances:\n\n🔵 Barcelona:\n"
    player_section += "\n".join([summarize_player(p, "Barcelona") for p in players["Barcelona"]])
    player_section += "\n\n⚪ Real Madrid:\n"
    player_section += "\n".join([summarize_player(p, "Real Madrid") for p in players["Real Madrid"]])

    # Combine all parts
    context = f"{match_info}\n{team_stats}\n{player_section}"

    # Optional: Save to file
    with open("llm_context.txt", "w", encoding="utf-8") as out:
        out.write(context)

    return context  # You can also print() or return this

# To generate and see the result:
if __name__ == "__main__":
    prompt_text = llm_prompt_readable()
    print(prompt_text[:1000] + "\n...")  # printing first 1000 charcters
