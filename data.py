import requests
import json

api_key = ""
headers = {
    'x-apisports-key': api_key
}

team_id = 529  # FC Barcelona

def last_30_matches():
    # Step 1: Get last 30 fixtures
    response = requests.get(
        'https://v3.football.api-sports.io/fixtures',
        headers=headers,
        params={'team': team_id, 'last': 30}
    )

    fixtures = response.json()['response']
    all_matches = []

    for match in fixtures:
        fixture_id = match['fixture']['id']
        date = match['fixture']['date']
        league = match['league']['name']
        season = match['league']['season']
        home_team = match['teams']['home']['name']
        away_team = match['teams']['away']['name']
        home_goals = match['goals']['home']
        away_goals = match['goals']['away']

        is_home = home_team == "Barcelona"
        opponent = away_team if is_home else home_team
        barca_goals = home_goals if is_home else away_goals
        opponent_goals = away_goals if is_home else home_goals
        result = (
            "Win" if barca_goals > opponent_goals else
            "Loss" if barca_goals < opponent_goals else
            "Draw"
        )
        home_or_away = "Home" if is_home else "Away"

        # Step 2: Get statistics for the match
        stats_resp = requests.get(
            'https://v3.football.api-sports.io/fixtures/statistics',
            headers=headers,
            params={'fixture': fixture_id}
        )
        stats_data = stats_resp.json().get('response', [])
        barca_stats = next((s for s in stats_data if s['team']['id'] == team_id), None)

        stats_dict = {}
        if barca_stats:
            for stat in barca_stats['statistics']:
                stats_dict[stat['type']] = stat['value']

        # Step 3: Build match record
        match_data = {
            'date': date,
            'league': league,
            'season': season,
            'home_or_away': home_or_away,
            'opponent': opponent,
            'result': result,
            'barca_goals': barca_goals,
            'opponent_goals': opponent_goals,
            'stats': stats_dict
        }

        all_matches.append(match_data)

    # Step 4: Save to JSON file
    with open('barcelona_last_30_matches.json', 'w', encoding='utf-8') as f:
        json.dump(all_matches, f, ensure_ascii=False, indent=4)

    print("Data saved to barcelona_last_30_matches.json")

def fixture_id():
    response = requests.get(
    'https://v3.football.api-sports.io/fixtures',
    headers=headers,
    params={
        'team': 529,  # Barcelona
        'date': '2025-01-12',
        'season': 2024
    }
    )
    data = response.json() 
    for match in data['response']:
        fixture = match['fixture']
        home = match['teams']['home']['name']
        away = match['teams']['away']['name']
        print(f"Fixture ID: {fixture['id']} | {home} vs {away}")

def key_player_stats():
    response = requests.get(
    'https://v3.football.api-sports.io/fixtures/players',
    headers=headers,
    params={'fixture': 1334517}
    )
    data = response.json()
    players_stats = []

    for team in data['response']:
        team_name = team['team']['name']
        for player in team['players']:
            p = player['player']
            stats = player['statistics'][0]

            players_stats.append({
                'player_id': p['id'],
                'name': p['name'],
                'team': team_name,
                'position': stats['games']['position'],
                'minutes': stats['games']['minutes'],
                'rating': float(p.get('rating') or 0),
                'captain': stats['games'].get('captain', False),
                'substitute': stats['games'].get('substitute', False),
                'goals': stats.get('goals', {}),
                'shots': stats.get('shots', {}),
                'passes': stats.get('passes', {}),
                'tackles': stats.get('tackles', {}),
                'duels': stats.get('duels', {}),
                'dribbles': stats.get('dribbles', {}),
                'fouls': stats.get('fouls', {}),
                'cards': stats.get('cards', {}),
                'saves': stats.get('goalkeeper', {}),
            })

    # Save to JSON
    with open('key_players_stats.json', 'w', encoding='utf-8') as f:
        json.dump(players_stats, f, ensure_ascii=False, indent=4)

    print("All player stats saved to key_players_stats.json")

def last_30_matches_barca_players():
    # Step 1: Get last 30 fixtures for Barcelona
    response = requests.get(
        'https://v3.football.api-sports.io/fixtures',
        headers=headers,
        params={'team': team_id, 'last': 30}
    )

    fixtures = response.json()['response']
    all_players_stats = []

    for match in fixtures:
        fixture_id = match['fixture']['id']

        # Step 2: Fetch players statistics for this fixture
        stats_resp = requests.get(
            'https://v3.football.api-sports.io/fixtures/players',
            headers=headers,
            params={'fixture': fixture_id}
        )
        stats_data = stats_resp.json()['response']

        # Step 3: Extract only Barcelona players' stats
        for team in stats_data:
            if team['team']['id'] == team_id:  # Only Barcelona
                for player in team['players']:
                    p = player['player']
                    stats = player['statistics'][0]

                    all_players_stats.append({
                        'fixture_id': fixture_id,
                        'match_date': match['fixture']['date'],
                        'opponent': match['teams']['away']['name'] if match['teams']['home']['id'] == team_id else match['teams']['home']['name'],
                        'player_id': p['id'],
                        'name': p['name'],
                        'position': stats['games']['position'],
                        'minutes': stats['games']['minutes'],
                        'rating': float(p.get('rating') or 0),
                        'captain': stats['games'].get('captain', False),
                        'substitute': stats['games'].get('substitute', False),
                        'goals': stats.get('goals', {}),
                        'shots': stats.get('shots', {}),
                        'passes': stats.get('passes', {}),
                        'tackles': stats.get('tackles', {}),
                        'duels': stats.get('duels', {}),
                        'dribbles': stats.get('dribbles', {}),
                        'fouls': stats.get('fouls', {}),
                        'cards': stats.get('cards', {}),
                        'saves': stats.get('goalkeeper', {}),
                    })
                break  # Once we find Barcelona's players, stop checking other teams

    # Step 4: Save to JSON
    with open('barcelona_next_30_matches_players.json', 'w', encoding='utf-8') as f:
        json.dump(all_players_stats, f, ensure_ascii=False, indent=4)


def get_latest_fixture():
    # Get the most recent fixture
    response = requests.get(
        "https://v3.football.api-sports.io/fixtures",
        headers=headers,
        params={"team": team_id, "last": 2}
    )
    data = response.json()
    return data['response'][1]

def get_team_stats(fixture_id, barca_id, opponent_id):
    # Get both Barcelona and Opponent team stats
    response = requests.get(
        "https://v3.football.api-sports.io/fixtures/statistics",
        headers=headers,
        params={"fixture": fixture_id}
    )
    data = response.json()
    stats_response = data.get('response', [])

    team_stats = {}

    for stat in stats_response:
        team_id_from_api = stat['team']['id']
        team_name = stat['team']['name']
        stats = {s['type']: s['value'] for s in stat['statistics']}

        if team_id_from_api == barca_id:
            team_stats['barcelona'] = {
                'team_id': team_id_from_api,
                'team_name': team_name,
                'stats': stats
            }
        elif team_id_from_api == opponent_id:
            team_stats['opponent'] = {
                'team_id': team_id_from_api,
                'team_name': team_name,
                'stats': stats
            }

    return team_stats

def get_player_stats(fixture_id):
    # Get players' statistics for both Barcelona and Opponent
    response = requests.get(
        'https://v3.football.api-sports.io/fixtures/players',
        headers=headers,
        params={'fixture': fixture_id}
    )
    data = response.json()
    players_by_team = {}

    for team in data['response']:
        team_name = team['team']['name']
        players_by_team[team_name] = []

        for player in team['players']:
            p = player['player']
            s = player['statistics'][0]

            players_by_team[team_name].append({
                "player_id": p['id'],
                "name": p['name'],
                "position": s['games'].get('position'),
                "minutes": s['games'].get('minutes'),
                "rating": float(p.get('rating') or 0),
                "captain": s['games'].get('captain', False),
                "substitute": s['games'].get('substitute', False),
                "goals": s.get('goals', {}),
                "assists": s.get('goals', {}).get('assists'),
                "shots": s.get('shots', {}),
                "passes": s.get('passes', {}),
                "tackles": s.get('tackles', {}),
                "duels": s.get('duels', {}),
                "dribbles": s.get('dribbles', {}),
                "fouls": s.get('fouls', {}),
                "cards": s.get('cards', {}),
                "saves": s.get('goalkeeper', {}),
            })

    return players_by_team

def generate_summary_data():
    fixture = get_latest_fixture()
    fixture_id = fixture['fixture']['id']
    date = fixture['fixture']['date']
    season = fixture['league']['season']
    league = fixture['league']['name']
    home_team = fixture['teams']['home']['name']
    away_team = fixture['teams']['away']['name']
    home_goals = fixture['goals']['home']
    away_goals = fixture['goals']['away']
    is_home = fixture['teams']['home']['id'] == team_id
    opponent = away_team if is_home else home_team
    home_or_away = "Home" if is_home else "Away"
    barca_goals = home_goals if is_home else away_goals
    opponent_goals = away_goals if is_home else home_goals

    result = (
        "Win" if barca_goals > opponent_goals else
        "Loss" if barca_goals < opponent_goals else
        "Draw"
    )

    team_stats = get_team_stats(fixture_id)
    player_stats = get_player_stats(fixture_id, opponent)

    summary = {
        "fixture_id": fixture_id,
        "date": date,
        "league": league,
        "season": season,
        "opponent": opponent,
        "home_or_away": home_or_away,
        "barca_goals": barca_goals,
        "opponent_goals": opponent_goals,
        "result": result,
        "team_stats": team_stats,
        "players": player_stats
    }

    with open("barca_summary_data.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)

    print("Saved match summary to barca_summary_data.json")

def generate_summary_data():
    # Main function to orchestrate everything
    fixture = get_latest_fixture()
    fixture_id = fixture['fixture']['id']
    date = fixture['fixture']['date']
    season = fixture['league']['season']
    league = fixture['league']['name']
    home_team = fixture['teams']['home']['name']
    away_team = fixture['teams']['away']['name']
    home_goals = fixture['goals']['home']
    away_goals = fixture['goals']['away']

    is_home = fixture['teams']['home']['id'] == team_id
    opponent = away_team if is_home else home_team
    home_or_away = "Home" if is_home else "Away"
    barca_goals = home_goals if is_home else away_goals
    opponent_goals = away_goals if is_home else home_goals

    result = (
        "Win" if barca_goals > opponent_goals else
        "Loss" if barca_goals < opponent_goals else
        "Draw"
    )

    # Need both team IDs
    barca_id = team_id
    opponent_id = fixture['teams']['away']['id'] if is_home else fixture['teams']['home']['id']

    # Fetch team stats for both teams
    team_stats = get_team_stats(fixture_id, barca_id, opponent_id)

    # Fetch player stats for both teams
    player_stats = get_player_stats(fixture_id)

    # Bundle everything nicely
    summary = {
        "fixture_id": fixture_id,
        "date": date,
        "league": league,
        "season": season,
        "opponent": opponent,
        "home_or_away": home_or_away,
        "barca_goals": barca_goals,
        "opponent_goals": opponent_goals,
        "result": result,
        "team_stats": team_stats,
        "players": player_stats
    }

    # Save to file
    with open("barca_summary_full.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)

    print("Saved full match summary to barca_summary_full.json")

def get_last_10_fixture_ids():
    response = requests.get(
        'https://v3.football.api-sports.io/fixtures',
        headers=headers,
        params={
            'team': team_id,
            'last': 15
        }
    )

    data = response.json()
    print("Last 10 Barcelona Fixtures:\n")
    for match in data['response']:
        fixture = match['fixture']
        date = fixture['date'][:10]
        fixture_id = fixture['id']
        home = match['teams']['home']['name']
        away = match['teams']['away']['name']
        home_goals = match['goals']['home']
        away_goals = match['goals']['away']
        print(f"Fixture ID: {fixture_id} | {date} | {home} {home_goals} - {away_goals} {away}")

def get_team_stats_for_fixture(fixture_id):
    # Step 1: Get match info to extract team IDs
    fixture_info = requests.get(
        'https://v3.football.api-sports.io/fixtures',
        headers=headers,
        params={'id': fixture_id}
    ).json()['response'][0]

    home_team = fixture_info['teams']['home']
    away_team = fixture_info['teams']['away']

    # Identify Barça and opponent correctly
    barca_id = home_team['id'] if home_team['name'] == "Barcelona" else away_team['id']
    opponent_id = away_team['id'] if barca_id == home_team['id'] else home_team['id']

    # Step 2: Get stats
    stats = get_team_stats(fixture_id, barca_id, opponent_id)

    # Step 3: Save to file
    with open("explain_like_five.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=4)

    print(f"Team stats saved for Fixture ID {fixture_id} (Barcelona vs Real Madrid)")

if __name__ == "__main__":
    #get_team_stats_for_fixture(1367758)
    get_last_10_fixture_ids()
