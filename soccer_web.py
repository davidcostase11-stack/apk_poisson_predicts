import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
import os
import math
import json
from io import StringIO
from datetime import datetime

# --- SETTINGS ---
st.set_page_config(page_title="Soccer Predictions Pro", page_icon="⚽", layout="wide")

LEAGUES = [
    "austria", "austria2", "belgium", "germany", "germany2", "germany3", 
    "denmark", "denmark2", "england", "england2", "england3", "spain", 
    "spain2", "finland", "france", "france2", "greece", "southkorea", 
    "southkorea2", "netherlands", "netherlands2", "italy", "italy2", 
    "norway", "poland", "poland2", "portugal", "portugal2", 
    "scotland", "scotland2", "sweden", "sweden2", "turkey", "vietnam", 
    "brazil", "czechrepublic", "czechrepublic2", "switzerland", "hungary", 
    "slovakia", "slovenia", "croatia", "norway2", "saudiarabia"
]

CACHE_FILE = "soccer_stats_cache.json"
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

# --- UTILS ---
def is_genuine_score(info):
    """Robustly checks if a string is a final score or a kickoff time."""
    info = str(info).strip()
    if 'v' in info.lower(): return False
    
    # Match pattern X:Y or X-Y
    match = re.match(r'^(\d+)\s?[:\-]\s?(\d+)$', info)
    if not match: return False
    
    h_str, a_str = match.groups()
    
    # RULE 1: If it's formatted like HH:MM (e.g. 01:00), it's a TIME
    if len(a_str) == 2:
        # If it starts with '0' (like 01:00) and is 2 digits, it's a time
        if h_str.startswith('0') and len(h_str) == 2: return False
        # Kickoff times are usually multiples of 5 or 15
        if int(h_str) < 24 and int(a_str) % 5 == 0:
            # Special case: 0-0 is a score
            if h_str == "0" and a_str == "0": return True
            return False
            
    # RULE 2: Scores are usually small numbers. 
    # If both sides are small, it's likely a score.
    if int(h_str) < 15 and int(a_str) < 15:
        return True
        
    return False

def clean_team_name(name):
    name = str(name).lower()
    junk = [r'\bfc\b', r'\bafc\b', r'\bsc\b', r'\bud\b', r'\brc\b', r'\bsd\b', r'\bvfl\b', r'\bvfb\b', r'\bsv\b', r'\bas\b', 
            r'\bunited\b', r'\butd\b', r'\bcity\b', r'\btown\b', r'\brovers\b', r'\bwanderers\b', r'\bathletic\b', 
            r'\balbion\b', r'\bolympic\b', r'\breal\b', r'\bde\b', r'\bda\b', r'\bdo\b', r'\bst\b', r'\bfsv\b', 
            r'\bspvg\b', r'\bu21\b', r'\bu23\b', r'\bac\b']
    for pattern in junk: name = re.sub(pattern, '', name)
    name = re.sub(r'[^a-z0-9]', '', name)
    return name.strip()

def poisson_pmf(k, mu):
    try: return (math.pow(mu, k) * math.exp(-mu)) / math.factorial(k)
    except: return 0.0

def calculate_predictions(res_df, h_tab, a_tab):
    if h_tab is None or a_tab is None or res_df is None or res_df.empty: return res_df
    try:
        avg_h_gf = h_tab['GF'].sum() / h_tab['GP'].sum()
        avg_h_ga = h_tab['GA'].sum() / h_tab['GP'].sum()
        avg_a_gf = a_tab['GF'].sum() / a_tab['GP'].sum()
        avg_a_ga = a_tab['GA'].sum() / a_tab['GP'].sum()

        h_str = h_tab.set_index('Team')
        a_str = a_tab.set_index('Team')
        h_str['Atk'], h_str['Def'] = (h_str['GF']/h_str['GP'])/avg_h_gf, (h_str['GA']/h_str['GP'])/avg_h_ga
        a_str['Atk'], a_str['Def'] = (a_str['GF']/a_str['GP'])/avg_a_gf, (a_str['GA']/a_str['GP'])/avg_a_ga
        
        h_map = {clean_team_name(n): n for n in h_str.index}
        a_map = {clean_team_name(n): n for n in a_str.index}

        def predict(row):
            h_raw, a_raw = str(row['Home']).strip(), str(row['Away']).strip()
            h_clean, a_clean = clean_team_name(h_raw), clean_team_name(a_raw)
            ht = h_raw if h_raw in h_str.index else h_map.get(h_clean)
            at = a_raw if a_raw in a_str.index else a_map.get(a_clean)
            if not ht or not at: return pd.Series(["-", 0.0, "N/A", "N/A", 0.0, "N/A"])
            lh = h_str.loc[ht, 'Atk'] * a_str.loc[at, 'Def'] * avg_h_gf
            la = a_str.loc[at, 'Atk'] * h_str.loc[ht, 'Def'] * avg_a_gf
            m = np.outer([poisson_pmf(i, lh) for i in range(8)], [poisson_pmf(i, la) for i in range(8)])
            o25 = 1 - (m[0,0]+m[0,1]+m[0,2]+m[1,0]+m[1,1]+m[2,0])
            hw, d, aw = np.sum(np.tril(m, -1)), np.sum(np.diag(m)), np.sum(np.triu(m, 1))
            best_w = max({'Home Win': hw, 'Draw': d, 'Away Win': aw}, key=lambda k: {'Home Win': hw, 'Draw': d, 'Away Win': aw}[k])
            hs, aws = np.unravel_index(np.argmax(m), m.shape)
            return pd.Series([f"{hs}-{aws}", float(o25), f"{o25:.1%}", best_w, float(max(hw, d, aw)), f"{max(hw, d, aw):.1%}" ])

        new_cols = res_df.apply(predict, axis=1)
        res_df[['Score', 'O25_Raw', 'Prob O2.5', 'Winner', 'WProb_Raw', 'Win Prob']] = new_cols
    except: pass
    return res_df

def load_full_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f: return json.load(f)
        except: return {}
    return {}

def save_to_cache(league, data):
    cache = load_full_cache()
    cache[league] = data
    with open(CACHE_FILE, 'w') as f: json.dump(cache, f)

def fetch_data(league):
    try:
        r1 = requests.get(f"https://www.soccerstats.com/results.asp?league={league}&pmtype=bydate", headers=HEADERS, timeout=15)
        res_df = None
        for df in pd.read_html(StringIO(r1.text)):
            regex_game = r'(\d\s?[:\-]\s?\d|\b[vV]\b|\d{1,2}:\d{2})'
            idx = -1
            for i in range(len(df.columns)):
                if df.iloc[:, i].astype(str).str.contains(regex_game).sum() > 2:
                    idx = i; break
            if idx > 0:
                mask = df.iloc[:, idx].astype(str).str.contains(regex_game)
                df_f = df[mask].copy()
                if len(df_f) >= 2:
                    res_df = pd.DataFrame({'Date': df_f.iloc[:, 0].astype(str), 'Home': df_f.iloc[:, idx-1].astype(str), 'Info': df_f.iloc[:, idx].astype(str), 'Away': df_f.iloc[:, idx+1].astype(str)})
                    break
        
        r2 = requests.get(f"https://www.soccerstats.com/homeaway.asp?league={league}", headers=HEADERS, timeout=15)
        h_tab, a_tab = None, None
        for df in pd.read_html(StringIO(r2.text)):
            target_row = -1
            for i in range(min(5, len(df))):
                if 'GP' in [str(x).upper() for x in df.iloc[i].tolist()]:
                    target_row = i; break
            if target_row != -1:
                cols = [str(x).upper() for x in df.iloc[target_row].tolist()]
                d = df.iloc[target_row+1:].copy()
                stats = pd.DataFrame({'Team': d.iloc[:, 1], 'GP': pd.to_numeric(d.iloc[:, cols.index('GP')], errors='coerce'), 'GF': pd.to_numeric(d.iloc[:, cols.index('GF')], errors='coerce'), 'GA': pd.to_numeric(d.iloc[:, cols.index('GA')], errors='coerce')}).dropna()
                if h_tab is None: h_tab = stats
                else: a_tab = stats; break
        
        if h_tab is not None and a_tab is not None and res_df is not None:
            data = calculate_predictions(res_df, h_tab, a_tab).to_dict('records')
            save_to_cache(league, data)
            return data
    except: pass
    return load_full_cache().get(league, [])

# --- APP UI ---
st.markdown("""
<style>
.match-header { background-color: #1E1E1E; border-radius: 10px 10px 0 0; padding: 10px; border: 1px solid #333; display: flex; justify-content: space-between; align-items: center; }
.predict-body { background-color: #252525; border-radius: 0 0 10px 10px; padding: 15px; border: 1px solid #333; border-top: none; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
.team-text { color: #FFFFFF !important; font-weight: bold; font-size: 16px; }
.bet-box { text-align: center; border-right: 1px solid #444; }
.bet-box:last-child { border-right: none; }
.bet-label { color: #888; font-size: 11px; text-transform: uppercase; font-weight: 800; }
.bet-value { color: #FFF; font-size: 20px; font-weight: 900; margin-top: 5px; }
.bet-conf { font-size: 13px; font-weight: bold; }
.score-box { background: #007BFF; color: yellow !important; padding: 3px 10px; border-radius: 5px; font-weight: bold; font-size: 18px; border: 2px solid yellow; }
.upcoming-box { background: #444; color: #FFFFFF !important; padding: 3px 10px; border-radius: 5px; font-weight: bold; font-size: 16px; border: 1px solid #666; }
</style>
""", unsafe_allow_html=True)

st.title("⚽ Soccer Smart Predictor")
tab1, tab2 = st.tabs(["🔥 Top Predictions", "📈 Model Accuracy"])

with tab1:
    l_sel = st.selectbox("Select League", LEAGUES, index=LEAGUES.index("portugal") if "portugal" in LEAGUES else 0)
    if st.button("🔄 Sync Live Data"): st.cache_data.clear()
    
    data = fetch_data(l_sel)
    if not data: st.warning("Connecting to source...")
    else:
        for g in data:
            raw_info = str(g['Info']).strip()
            is_score = is_genuine_score(raw_info)
            
            score_html = f'<div class="score-box">{raw_info}</div>' if is_score else f'<div class="upcoming-box">🕒 {raw_info}</div>'
            st.markdown(f"""
            <div class="match-header">
                <div style="width: 38%; text-align: right;" class="team-text">{g['Home']}</div>
                <div style="width: 24%; text-align: center;">{score_html}</div>
                <div style="width: 38%; text-align: left;" class="team-text">{g['Away']}</div>
            </div>
            """, unsafe_allow_html=True)
            
            win_color = "#2ECC71" if g['Winner'] == 'Home Win' or g['Winner'] == 'Away Win' else "#F1C40F"
            if g['Winner'] == "N/A": win_color = "#888"
            o25_color = "#FF00FF" if float(g['O25_Raw']) > 0.6 else "#AAA"

            st.markdown(f"""
            <div class="predict-body">
                <div style="display: flex; justify-content: space-around; align-items: center;">
                    <div class="bet-box" style="width: 33%;">
                        <div class="bet-label">Outcome</div>
                        <div class="bet-value" style="color: {win_color}">{g['Winner']}</div>
                        <div class="bet-conf" style="color: {win_color}">{g['Win Prob']} Conf.</div>
                    </div>
                    <div class="bet-box" style="width: 33%;">
                        <div class="bet-label">Over 2.5</div>
                        <div class="bet-value" style="color: {o25_color}">{g['Prob O2.5']}</div>
                        <div class="bet-conf" style="color: {o25_color}">Probability</div>
                    </div>
                    <div class="bet-box" style="width: 33%;">
                        <div class="bet-label">Score</div>
                        <div class="bet-value" style="color: #007BFF">{g['Score']}</div>
                        <div class="bet-conf" style="color: #666">Predicted</div>
                    </div>
                </div>
                <div style="text-align: center; margin-top: 10px; font-size: 10px; color: #888;">
                    Match Date: {g['Date']}
                </div>
            </div>
            """, unsafe_allow_html=True)

with tab2:
    st.header("🏆 Model Performance Dashboard")
    st.write("This summary **only includes finished games** with real scores.")
    cache = load_full_cache()
    if not cache: st.info("Browse leagues in the first tab to generate performance data.")
    else:
        summary_rows = []
        for l, games in cache.items():
            wins, o25, total = 0, 0, 0
            for g in games:
                raw_i = str(g['Info'])
                # STRICT FILTER: Only games with confirmed final scores
                if is_genuine_score(raw_i) and g.get('WProb_Raw', 0) > 0:
                    h, a = map(int, re.findall(r'\d+', raw_i))
                    act_w = 'Home Win' if h > a else ('Draw' if h == a else 'Away Win')
                    if act_w == g['Winner']: wins += 1
                    if ((h + a) > 2.5) == (g.get('O25_Raw', 0) > 0.5): o25 += 1
                    total += 1
            
            if total > 0:
                summary_rows.append({
                    'League': l.upper(),
                    'Finished Games': total,
                    'Winner Accuracy': f"{(wins/total):.1%}",
                    'Over 2.5 Accuracy': f"{(o25/total):.1%}"
                })
        
        if summary_rows:
            df_summary = pd.DataFrame(summary_rows).sort_values(by='Winner Accuracy', ascending=False)
            st.table(df_summary)
            
            # Big summary metrics
            total_games = sum([r['Finished Games'] for r in summary_rows])
            st.subheader(f"Total Combined Stats ({total_games} Games)")
            # Add simple average calculation logic if needed
        else:
            st.warning("No finished games found in the currently cached leagues.")
