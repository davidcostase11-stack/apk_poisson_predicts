import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
import os
import math
import json
from io import StringIO

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

# --- PERSISTENCE ---
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

# --- MATH LOGIC ---
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
            prob_w = max(hw, d, aw)
            hs, aws = np.unravel_index(np.argmax(m), m.shape)
            return pd.Series([f"{hs}-{aws}", float(o25), f"{o25:.1%}", best_w, float(prob_w), f"{prob_w:.1%}" ])

        new_cols = res_df.apply(predict, axis=1)
        res_df[['Score', 'O25_Raw', 'Prob O2.5', 'Winner', 'WProb_Raw', 'Win Prob']] = new_cols
    except: pass
    return res_df

# --- DATA FETCH ---
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

# --- UI ---
st.markdown("""
<style>
.match-card { background-color: #1E1E1E; border-radius: 10px; padding: 15px; margin-bottom: 10px; border: 1px solid #333; }
.team-name { font-size: 18px; font-weight: bold; color: white; }
.result-box { background-color: #007BFF; color: white; padding: 5px 15px; border-radius: 5px; font-weight: bold; font-size: 20px; display: inline-block; }
.prediction-text { color: #AAA; font-size: 14px; }
</style>
""", unsafe_allow_html=True)

st.title("⚽ Soccer Results & Smart Predictions")
tab1, tab2 = st.tabs(["📊 Matches", "📈 Model Summary"])

with tab1:
    l_sel = st.selectbox("Choose League", LEAGUES, index=LEAGUES.index("portugal") if "portugal" in LEAGUES else 0)
    if st.button("🔄 Sync Live Data"): st.cache_data.clear()
    data = fetch_data(l_sel)
    if not data: st.warning("No matches found.")
    else:
        for g in data:
            st.markdown(f"""<div class="match-card"><div style="display: flex; justify-content: space-between; align-items: center;"><div style="width: 40%; text-align: right;" class="team-name">{g['Home']}</div><div style="width: 20%; text-align: center;"><span class="result-box">{g['Info']}</span></div><div style="width: 40%; text-align: left;" class="team-name">{g['Away']}</div></div><hr style="margin: 10px 0; border-color: #444;"><div style="display: flex; justify-content: space-between;"><div class="prediction-text">Pick: <b style="color:#00FF00">{g['Winner']}</b></div><div class="prediction-text">Conf: <b style="color:#00FF00">{g['Win Prob']}</b></div><div class="prediction-text">O2.5: <b style="color:#FF00FF">{g['Prob O2.5']}</b></div><div class="prediction-text">Score: <b>{g['Score']}</b></div></div><div style="font-size: 10px; color: #666; margin-top: 5px;">Date: {g['Date']}</div></div>""", unsafe_allow_html=True)

with tab2:
    st.header("Model Performance")
    cache = load_full_cache()
    if not cache: st.info("No cached data yet. Browse leagues to see accuracy summary.")
    else:
        summary_rows = []
        for l, games in cache.items():
            wins, o25, total = 0, 0, 0
            for g in games:
                m = re.search(r'(\d+)\s?[:\-]\s?(\d+)', str(g['Info']))
                if m and g.get('WProb_Raw', 0) > 0:
                    h, a = int(m.group(1)), int(m.group(2))
                    act_w = 'Home Win' if h > a else ('Draw' if h == a else 'Away Win')
                    if act_w == g['Winner']: wins += 1
                    if ((h + a) > 2.5) == (g.get('O25_Raw', 0) > 0.5): o25 += 1
                    total += 1
            if total > 0: summary_rows.append({'League': l.upper(), 'Games': total, 'Win Accuracy': f"{(wins/total):.1%}", 'O2.5 Accuracy': f"{(o25/total):.1%}"})
        
        if summary_rows: st.table(pd.DataFrame(summary_rows))
        else: st.write("Browse more leagues to see data here.")
