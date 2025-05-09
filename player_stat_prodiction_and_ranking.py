from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Polygon, Arc
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Polygon, Arc
import matplotlib.patheffects as path_effects


def calculate_score(player_stats, position):
    score = 0
    
    # Helper function to safely calculate ratios to avoid division by zero
    def safe_ratio(numerator, denominator, default=0):
        if denominator == 0:
            return default
        return numerator / denominator
    
    # Convert position code from numeric to letter code if needed
    pos_code = position
    if isinstance(position, (int, float)):
        pos_map = {0: 'G', 1: 'D', 2: 'M', 3: 'F'}
        pos_code = pos_map.get(position, 'U')  # U for unknown
    
    if pos_code == 'F':  # Forward
        score += player_stats['goals.total'] * 3.0
        score += safe_ratio(player_stats['shots.on'], player_stats['shots.total']) * 1.5
        score += player_stats['goals.assists'] * 1.5
        score += safe_ratio(player_stats['dribbles.success'], player_stats['dribbles.attempts']) * 1.0
        score += player_stats['passes.accuracy'] * 0.5
        score += safe_ratio(player_stats['duels.won'], player_stats['duels.total']) * 0.5
        
    elif pos_code == 'M':  # Midfielder
        score += player_stats['passes.total'] * 0.01
        score += player_stats['passes.accuracy'] * 1.5
        score += player_stats['goals.assists'] * 2.0
        score += player_stats['goals.total'] * 1.0
        score += player_stats['tackles.total'] * 0.5
        score += player_stats['tackles.interceptions'] * 0.75
        score += safe_ratio(player_stats['dribbles.success'], player_stats['dribbles.attempts']) * 1.0
        score += safe_ratio(player_stats['duels.won'], player_stats['duels.total']) * 0.75
        
    elif pos_code == 'D':  # Defender
        score += player_stats['tackles.total'] * 1.5
        score += player_stats['tackles.blocks'] * 1.5
        score += player_stats['tackles.interceptions'] * 1.5
        score += safe_ratio(player_stats['duels.won'], player_stats['duels.total']) * 2.0
        score += player_stats['passes.accuracy'] * 1.0
        score += player_stats['goals.total'] * 1.0
        score += player_stats['goals.assists'] * 0.5
        
    elif pos_code == 'G':  # Goalkeeper
        score += player_stats['goals.saves'] * 2.5
        score += player_stats['passes.accuracy'] * 1.0
        score += player_stats['tackles.blocks'] * 1.0
        score += safe_ratio(player_stats['duels.won'], player_stats['duels.total']) * 0.5
        
        # If 'goals.conceded' is in the stats, add penalty for conceded goals
        if 'goals.conceded' in player_stats:
            score += player_stats['goals.conceded'] * -1
    
    return score

def rank_players_by_position(data_file_path, N_previous_games=10):
    """
    Ranks players based on predicted performance for each position.
    
    Args:
        data_file_path: Path to the JSON file with player data
        N_previous_games: Number of previous games to use for prediction
        
    Returns:
        Dictionary with ranked players by position
    """
    # Load and preprocess data
    with open(data_file_path, 'r') as file:
        matches = json.load(file)

    df = pd.json_normalize(matches)

    # Define columns to fill missing values
    fill_columns = [
        'goals.total', 'goals.assists', 'goals.saves',
        'shots.total', 'shots.on',
        'passes.total', 'passes.accuracy',
        'tackles.total', 'tackles.blocks', 'tackles.interceptions',
        'duels.total', 'duels.won',
        'dribbles.attempts', 'dribbles.success',
        'minutes'
    ]

    for col in fill_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    df['match_date'] = pd.to_datetime(df['match_date'])

    # Encode position if not already encoded
    if df['position'].dtype == 'object':
        position_mapping = {'G': 0, 'D': 1, 'M': 2, 'F': 3}
        df['position'] = df['position'].map(position_mapping)

    # Define target stats
    target_stats = [
        'goals.total', 'goals.assists', 'goals.saves',
        'shots.total', 'shots.on',
        'passes.total', 'passes.accuracy',
        'tackles.total', 'tackles.blocks', 'tackles.interceptions',
        'duels.total', 'duels.won',
        'dribbles.attempts', 'dribbles.success'
    ]

    # Group players and train models
    player_groups = df.groupby('player_id')
    player_models = {}
    player_features = defaultdict(list)
    player_targets = defaultdict(list)

    for player_id, group in player_groups:
        group = group.sort_values('match_date')
        
        if len(group) <= N_previous_games:
            continue  # Skip players without enough history

        for idx in range(N_previous_games, len(group)):
            past_games = group.iloc[idx-N_previous_games:idx]
            next_game = group.iloc[idx]
            
            # Features: average of past games
            features = past_games[target_stats].mean().values.tolist()
            features += [
                past_games['minutes'].mean(),
                past_games['position'].iloc[-1]
            ]
            
            # Target: actual next game stats
            targets = next_game[target_stats].values.tolist()
            
            player_features[player_id].append(features)
            player_targets[player_id].append(targets)

    # Train models for each player
    for player_id in player_features:
        X = np.array(player_features[player_id])
        y = np.array(player_targets[player_id])
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        player_models[player_id] = model

    # Generate predictions and calculate scores
    position_names = {0: 'G', 1: 'D', 2: 'M', 3: 'F'}
    position_players = {
        'G': [],  # Goalkeepers
        'D': [],  # Defenders
        'M': [],  # Midfielders
        'F': []   # Forwards
    }

    for player_id, model in player_models.items():
        player_row = df[df['player_id'] == player_id].iloc[-1]
        name = player_row['name']
        pos_code = player_row['position']
        
        
        
        
        if name.strip() == 'Lamine Yamal' or name.strip() == 'Raphinha':
            pos_code = 3
        
        pos_letter = position_names.get(pos_code)    

        #print("Player:", name, "Position:", pos_letter)
        
        # Get last N games for prediction
        last_games = df[df['player_id'] == player_id].sort_values('match_date').iloc[-N_previous_games:]
        
        features = last_games[target_stats].mean().values.tolist()
        features += [
            last_games['minutes'].mean(),
            last_games['position'].iloc[-1]
        ]
        
        features = np.array(features).reshape(1, -1)
        pred = model.predict(features)
        
        # Convert predictions to dictionary
        pred_dict = dict(zip(target_stats, pred.flatten()))
        
        # Add goals.conceded for goalkeepers if available
        if 'goals.conceded' in df.columns and pos_letter == 'G':
            avg_conceded = last_games['goals.conceded'].mean() if 'goals.conceded' in last_games.columns else 0
            pred_dict['goals.conceded'] = avg_conceded

        # Calculate score
        player_score = calculate_score(pred_dict, pos_letter)
        
        # Store player info
        player_info = {
            'id': player_id,
            'name': name,
            'position': pos_letter,
            'score': player_score,
            'predicted_stats': pred_dict
        }
        
        position_players[pos_letter].append(player_info)

    # Rank players in each position
    for pos in position_players:
        position_players[pos] = sorted(position_players[pos], key=lambda x: x['score'], reverse=True)
        
        # Add rank to each player
        for i, player in enumerate(position_players[pos]):
            player['rank'] = i + 1

    return position_players

def display_rankings(rankings):
    """
    Display the player rankings in a formatted way.
    """
    position_names = {
        'G': 'Goalkeepers',
        'D': 'Defenders',
        'M': 'Midfielders', 
        'F': 'Forwards'
    }
    
    for pos, players in rankings.items():
        print(f"\n{position_names[pos]} Rankings:")
        print("-" * 80)
        print(f"{'Rank':<5}{'Name':<25}{'Score':<10}{'Key Stats'}")
        print("-" * 80)
        
        for player in players:
            # Format key stats based on position
            if pos == 'G':
                key_stats = f"Saves: {player['predicted_stats']['goals.saves']:.2f}"
                if 'goals.conceded' in player['predicted_stats']:
                    key_stats += f", Conceded: {player['predicted_stats']['goals.conceded']:.2f}"
            elif pos == 'D':
                key_stats = f"Tackles: {player['predicted_stats']['tackles.total']:.2f}, Blocks: {player['predicted_stats']['tackles.blocks']:.2f}, Int: {player['predicted_stats']['tackles.interceptions']:.2f}"
            elif pos == 'M':
                key_stats = f"Passes: {player['predicted_stats']['passes.total']:.2f}, Acc: {player['predicted_stats']['passes.accuracy']:.2f}, Assists: {player['predicted_stats']['goals.assists']:.2f}"
            else:  # Forward
                key_stats = f"Goals: {player['predicted_stats']['goals.total']:.2f}, Shots: {player['predicted_stats']['shots.total']:.2f}, On Target: {player['predicted_stats']['shots.on']:.2f}"
            
            print(f"{player['rank']:<5}{player['name']:<25}{player['score']:.2f}{'':<10}{key_stats}")

def export_rankings_to_csv(rankings, output_file='player_rankings.csv'):
    """
    Export player rankings to a CSV file.
    """
    rows = []
    for pos, players in rankings.items():
        for player in players:
            # Extract main stats for each position
            if pos == 'G':
                key_stats = {
                    'saves': player['predicted_stats']['goals.saves'],
                    'conceded': player['predicted_stats'].get('goals.conceded', 0)
                }
            elif pos == 'D':
                key_stats = {
                    'tackles': player['predicted_stats']['tackles.total'],
                    'blocks': player['predicted_stats']['tackles.blocks'],
                    'interceptions': player['predicted_stats']['tackles.interceptions']
                }
            elif pos == 'M':
                key_stats = {
                    'passes': player['predicted_stats']['passes.total'],
                    'pass_accuracy': player['predicted_stats']['passes.accuracy'],
                    'assists': player['predicted_stats']['goals.assists']
                }
            else:  # Forward
                key_stats = {
                    'goals': player['predicted_stats']['goals.total'],
                    'shots': player['predicted_stats']['shots.total'],
                    'shots_on_target': player['predicted_stats']['shots.on']
                }
            
            row = {
                'player_id': player['id'],
                'name': player['name'],
                'position': pos,
                'rank': player['rank'],
                'score': player['score'],
                **key_stats
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"Rankings exported to {output_file}")

def create_bar_chart(rankings, save_file='position_rankings.png'):
    """
    Create horizontal bar charts showing player scores by position
    """
    position_names = {
        'G': 'Goalkeepers',
        'D': 'Defenders',
        'M': 'Midfielders', 
        'F': 'Forwards'
    }
    
    # Set up the figure
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Player Rankings by Position', fontsize=24, fontweight='bold', y=0.98)
    
    # Flatten axes for easier iteration
    axes = axes.flatten()
    
    # Plot each position
    for i, pos in enumerate(position_names.keys()):
        ax = axes[i]
        position_data = rankings[pos]
        
        # Sort by score if not already sorted
        position_data = sorted(position_data, key=lambda x: x['score'], reverse=True)
        
        # Get player names and scores
        names = [player['name'] for player in position_data]
        scores = [player['score'] for player in position_data]
        
        # Create color gradient based on score
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(scores)))
        
        # Create horizontal bar chart
        bars = ax.barh(names, scores, color=colors)
        
        # Add score values at the end of each bar
        for bar, score in zip(bars, scores):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2, 
                   f'{score:.2f}', va='center', fontsize=10)
        
        # Set title and labels
        ax.set_title(position_names[pos], fontsize=18)
        ax.set_xlabel('Score', fontsize=12)
        
        # Remove y-axis label and add gridlines
        ax.set_ylabel('')
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Adjust limits
        max_score = max(scores) if scores else 0
        ax.set_xlim(0, max_score * 1.15)
        
        # Invert y-axis so highest score is at top
        ax.invert_yaxis()
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_file

def create_soccer_field_viz(rankings, save_file='team_formation.png'):
    """
    Create a soccer field visualization showing top players in a formation
    """
    # Create an empty figure
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111)
    
    # Import Arc patch
    from matplotlib.patches import Arc
    
    # Draw the soccer field
    # Field dimensions (in meters, but we'll just use relative coordinates)
    field_length, field_width = 105, 68
    
    # Draw the field outline
    field_color = '#538032'
    line_color = 'white'
    
    # Draw the field
    rect = plt.Rectangle((0, 0), field_length, field_width, color=field_color)
    ax.add_patch(rect)
    
    # Draw halfway line
    plt.plot([field_length/2, field_length/2], [0, field_width], color=line_color, linewidth=2)
    
    # Draw center circle
    center_circle = plt.Circle((field_length/2, field_width/2), 9.15, color=line_color, fill=False, linewidth=2)
    ax.add_patch(center_circle)
    
    # Draw center spot
    center_spot = plt.Circle((field_length/2, field_width/2), 0.5, color=line_color, linewidth=2)
    ax.add_patch(center_spot)
    
    # Draw penalty areas
    left_penalty = plt.Rectangle((0, field_width/2 - 20.16), 16.5, 40.32, 
                                 ec=line_color, fc='none', linewidth=2)
    right_penalty = plt.Rectangle((field_length - 16.5, field_width/2 - 20.16), 16.5, 40.32, 
                                  ec=line_color, fc='none', linewidth=2)
    ax.add_patch(left_penalty)
    ax.add_patch(right_penalty)
    
    # Draw goal areas
    left_goal_area = plt.Rectangle((0, field_width/2 - 9.16), 5.5, 18.32, 
                                   ec=line_color, fc='none', linewidth=2)
    right_goal_area = plt.Rectangle((field_length - 5.5, field_width/2 - 9.16), 5.5, 18.32, 
                                    ec=line_color, fc='none', linewidth=2)
    ax.add_patch(left_goal_area)
    ax.add_patch(right_goal_area)
    
    # Draw goals
    goal_width = 7.32
    left_goal = plt.Rectangle((-2, field_width/2 - goal_width/2), 2, goal_width, 
                              ec=line_color, fc=line_color, alpha=0.3, linewidth=2)
    right_goal = plt.Rectangle((field_length, field_width/2 - goal_width/2), 2, goal_width, 
                               ec=line_color, fc=line_color, alpha=0.3, linewidth=2)
    ax.add_patch(left_goal)
    ax.add_patch(right_goal)
    
    # Draw corner arcs
    corner_radius = 1
    corners = [(0, 0), (0, field_width), (field_length, 0), (field_length, field_width)]
    for corner in corners:
        corner_arc = Arc(corner, 2*corner_radius, 2*corner_radius, 
                         theta1=0, theta2=90, angle=0, color=line_color, linewidth=2)
        ax.add_patch(corner_arc)
    
    # Draw penalty spots
    left_penalty_spot = plt.Circle((11, field_width/2), 0.5, color=line_color)
    right_penalty_spot = plt.Circle((field_length - 11, field_width/2), 0.5, color=line_color)
    ax.add_patch(left_penalty_spot)
    ax.add_patch(right_penalty_spot)
    
    # Draw penalty arcs
    left_penalty_arc = Arc((11, field_width/2), 18.3, 18.3, theta1=310, theta2=50, 
                          angle=0, color=line_color, linewidth=2)
    right_penalty_arc = Arc((field_length - 11, field_width/2), 18.3, 18.3, 
                            theta1=130, theta2=230, angle=0, color=line_color, linewidth=2)
    ax.add_patch(left_penalty_arc)
    ax.add_patch(right_penalty_arc)
    
    # Set up a 4-3-3 formation (common in modern football)
    player_positions = {
        'G': [(10, field_width/2)],  # Goalkeeper
        'D': [(25, field_width*0.2), (25, field_width*0.4), (25, field_width*0.6), (25, field_width*0.8)],  # Defenders
        'M': [(45, field_width*0.25), (45, field_width*0.5), (45, field_width*0.75)],  # Midfielders
        'F': [(70, field_width*0.25), (70, field_width*0.5), (70, field_width*0.75)]   # Forwards
    }
    
    # Colors by position
    position_colors = {
        'G': '#FFC75F',  # Gold for goalkeeper
        'D': '#FF9671',  # Orange for defenders
        'M': '#D65DB1',  # Purple for midfielders
        'F': '#FF6F91'   # Red for forwards
    }
    
    # Add player circles and names
    for pos, positions in player_positions.items():
        # Get top players for this position (limit to how many we need)
        top_players = rankings[pos][:len(positions)]
        
        for i, (x, y) in enumerate(positions):
            if i < len(top_players):
                player = top_players[i]
                
                # Draw player circle
                player_circle = plt.Circle((x, y), 3.5, color=position_colors[pos], 
                                          alpha=0.8, linewidth=2, edgecolor='white')
                ax.add_patch(player_circle)
                
                # Add player name with outline for better visibility
                name_text = ax.text(x, y, f"{player['name'].split()[-1]}", 
                                   ha='center', va='center', fontsize=10, 
                                   fontweight='bold', color='white')
                
                # Add white outline to text
                name_text.set_path_effects([
                    path_effects.Stroke(linewidth=3, foreground='black'),
                    path_effects.Normal()
                ])
                
                # Add player score below
                score_text = ax.text(x, y-4.5, f"({player['score']:.1f})", 
                                    ha='center', va='center', fontsize=8, 
                                    color='white', fontweight='bold')
                
                score_text.set_path_effects([
                    path_effects.Stroke(linewidth=2, foreground='black'),
                    path_effects.Normal()
                ])
    
    # Add title
    plt.title('Best XI Based on Position Rankings', fontsize=20, pad=20)
    
    # Remove axes
    plt.axis('off')
    plt.axis('equal')
    
    # Tight layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_file

def visualize_rankings(rankings):
    """
    Create multiple visualizations for the player rankings
    """
    # Generate all visualizations
    radar_chart = create_radar_chart(rankings)
    bar_chart = create_bar_chart(rankings)
    field_viz = create_soccer_field_viz(rankings)
    heatmap = create_heatmap(rankings)
    
    print(f"Generated visualizations: {radar_chart}, {bar_chart}, {field_viz}, {heatmap}")
    
    return {
        'radar_chart': radar_chart,
        'bar_chart': bar_chart,
        'field_viz': field_viz,
        'heatmap': heatmap
    }



def main():
    data_file_path = 'barcelona_next_30_matches_players.json'
    rankings = rank_players_by_position(data_file_path)
    display_rankings(rankings)
    export_rankings_to_csv(rankings)
    visualizations = visualize_rankings(rankings)
    return rankings, visualizations


if __name__ == "__main__":
    main()