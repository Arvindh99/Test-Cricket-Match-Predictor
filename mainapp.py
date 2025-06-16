import matplotlib.pyplot as plt
import gradio as gr
import sqlite3
import pandas as pd
import pickle
import joblib
import numpy as np


# === Load Data and Models ===
def load_database():
    conn = sqlite3.connect('data.db')
    match_data = pd.read_sql('SELECT * FROM train_data', conn)
    conn.close()
    return match_data

match_data = load_database()

result_model = joblib.load("draw_prediction.pkl")
winner_model = joblib.load("winner_prediction.pkl")

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# === Manual Encoding for Teams ===
encode = {
    'team1': {'Australia':1,'New Zealand':2,'West Indies':3,'Zimbabwe':4,'Bangladesh':5,'India':6,'England':7,'South Africa':8,'Pakistan':9,'Sri Lanka':10,'Ireland':11},
    'team2': {'Australia':1,'New Zealand':2,'West Indies':3,'Zimbabwe':4,'Bangladesh':5,'India':6,'England':7,'South Africa':8,'Pakistan':9,'Sri Lanka':10,'Ireland':11},
    'toss_winner': {'Australia':1,'New Zealand':2,'West Indies':3,'Zimbabwe':4,'Bangladesh':5,'India':6,'England':7,'South Africa':8,'Pakistan':9,'Sri Lanka':10,'Ireland':11}}

# === Update Toss Winner Options Dynamically ===
def update_toss_winner(team1, team2):
    if team1 and team2:
        return gr.update(choices=[team1, team2], value=None)
    return gr.update(choices=[], value=None)

# === Prediction Function ===
def predict_draw(toss_decision,toss_winner,venue, team1, team2):
    match_result_df = pd.DataFrame([{"toss_decision": toss_decision,"toss_winner": toss_winner,"venue": venue,"team1": team1,"team2": team2,}])

    for col in ['team1', 'team2', 'toss_winner']:
        match_result_df[col] = match_result_df[col].map(encode[col])
        
    for col in ['toss_decision', 'venue']:
        match_result_df[col] = encoders[col].transform(match_result_df[col])

    draw_prob = result_model.predict_proba(match_result_df)[0][1]
    not_draw_prob = 1 - draw_prob
    reverse_encode = {v: k for k, v in encode['team1'].items()}
    if draw_prob > 0.5:
        summary = f"""❓ Will it be a Draw?
                    ✅ Yes, likely ({draw_prob*100:.1f}% chance)"""
        probs = {"Draw": draw_prob * 100,
            team1: (not_draw_prob / 2) * 100,
            team2: (not_draw_prob / 2) * 100,}
    else:
        winner_probs = winner_model.predict_proba(match_result_df)[0]
        team1_encoded = encode['team1'][team1]
        team2_encoded = encode['team2'][team2]
        team1_prob = winner_probs[list(winner_model.classes_).index(team1_encoded)] * 100
        team2_prob = winner_probs[list(winner_model.classes_).index(team2_encoded)] * 100
        
        winner_team = team1 if team1_prob > team2_prob else team2

        summary = (
            f"❓ Will it be a Draw?\n No ({not_draw_prob*100:.1f}% chance)\n\n"
            f"🔮 Who has the upper hand?\n"
            f"- {team1}: {team1_prob:.1f}%\n"
            f"- {team2}: {team2_prob:.1f}%\n"
            f"🧠 Predicted winner: {winner_team}")
        
        probs = {"Draw": draw_prob * 100,team1: team1_prob,team2: team2_prob,}
        
        fig, ax = plt.subplots()
        ax.bar(probs.keys(), probs.values(), color=['gray', '#1f77b4', '#ff7f0e'])
        ax.set_ylabel('Probability (%)')
        ax.set_title('Match Outcome Probability Breakdown')
        ax.set_ylim(0, 100)
        
        for i, v in enumerate(probs.values()):
            ax.text(i, v + 1, f"{v:.1f}%", ha='center', fontweight='bold')
            
        plt.tight_layout()
        return summary, fig


# === Gradio UI ===
with gr.Blocks() as demo:
    gr.Markdown("## 🏏 Cricket Test Match Draw Predictor")

    with gr.Row():
        venue = gr.Dropdown(label="Select Venue", choices=match_data["venue"].unique().tolist(), interactive=True,value=None)
        team1 = gr.Dropdown(label="Select Team 1", choices=match_data["team1"].unique().tolist(), interactive=True,value=None)
        team2 = gr.Dropdown(label="Select Team 2", choices=match_data["team2"].unique().tolist(), interactive=True,value=None)
        toss_winner = gr.Dropdown(label="Select Toss Winner", choices=[team1,team2], interactive=True)
        toss_decision = gr.Dropdown(label="Select Toss Decision", choices=match_data["toss_decision"].unique().tolist(), interactive=True,value=None)

    team1.change(update_toss_winner, inputs=[team1, team2], outputs=toss_winner)
    team2.change(update_toss_winner, inputs=[team1, team2], outputs=toss_winner)
        
    with gr.Row():
        with gr.Column(scale=1):
            pass
        with gr.Column(scale=2):
            predict_button = gr.Button("Predict", variant="primary")            
        with gr.Column(scale=1):
            pass
    
    with gr.Row():
        with gr.Column(scale=1):
            output_text = gr.Textbox(label="Prediction Result",lines=7, interactive=False)            
        with gr.Column(scale=1):
            output_plot = gr.Plot(label="Match Outcome Breakdown")
    

    predict_button.click(fn=predict_draw,inputs=[toss_decision,toss_winner,venue, team1, team2],outputs=[output_text, output_plot])

demo.launch(inbrowser=True)
