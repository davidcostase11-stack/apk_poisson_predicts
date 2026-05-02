import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
import os
import math
import json
import time
from io import StringIO
from datetime import datetime, timedelta

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

LEAGUE_TRIGGERS = {
    "germany": {"win": 0.65, "o25": 0.65},
    "netherlands": {"win": 0.65, "o25": 0.65},
    "netherlands2": {"win": 0.62, "o25": 0.60},
    "portugal": {"win": 0.75, "o25": 0.70},
    "spain": {"win": 0.72, "o25": 0.72},
    "greece": {"win": 0.70, "o25": 0.80},
    "england": {"win": 0.68, "o25": 0.70},
    "brazil": {"win": 0.65, "o25": 0.75},
    "austria": {"win": 0.65, "o25": 0.65},
    "norway": {"win": 0.65, "o25": 0.60},
}
DEFAULT_TRIGGERS = {"win": 0.68, "o25": 0.72}

CACHE_FILE = "soccer_stats_cache.json"
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

# --- UTILS ---
def parse_date(date_str):
    try:
        match = re.search(r'(\d{1,2})\s+([A-Za-z]+)', str(date_str))
        if match:
            day = int(match.group(1))
            month_str = match.group(2)[:3].lower()
            months = {"jan":1, "feb":2, "mar":3, "apr":4, "may":5, "jun":6, "jul":7, "aug":8, "sep":9, "oct":10, "nov":11, "dec":12}
            return datetime(datetime.now().year, months.get(month_str, 5), day)
        match_s = re.search(r'(\d{1,2})/(\d{1,2})', str(date_str))
        if match_s: return datetime(datetime.now().year, int(match_s.group(2)), int(match_s.group(1)))
        return None
    except: return None

def is_genuine_score(info):
    info = str(info).strip()
    if 'v' in info.lower(): return False
    match = re.match(r'^(\d+)\s?[:\-]\s?(\d+)$', info)
    if not match: return False
    h, a = match.groups()
    if len(a) == 2:
        if h.startswith('0') and len(h)==2: return False
        if int(h)<24 and int(a)%5==0:
            if h=="0" and a=="0": return True
            return False
    return int(h)<15 and int(a)<15

def clean_team_name(name):
    name = str(name).lower()
    junk = [r'\bfc\b', r'\bafc\b', r'\bsc\b', r'\bud\b', r'\brc\b', r'\bsd\b', r'\bvfl\b', r'\bvfb\b', r'\bsv\b', r'\bas\b', 
            r'\bunited\b', r'\butd\b', r'\bcity\b', r'\btown\b', r'\brovers\b', r'\bwanderers\b', r'\bathletic\b', 
            r'\balbion\b', r'\bolympic\b', r'\breal\b', r'\bde\b', r'\bda\b', r'\bdo\b', r'\bst\b', r'\bfsv\b', 
            r'\bspvg\b', r'\bu21\b', r'\bu23\b', r'\bac\b']
    for p in junk: name = re.sub(p, '', name)
    return re.sub(r'[^a-z0-9]', '', name).strip()

def poisson_pmf(k, mu):
    try: return (math.pow(mu, k) * math.exp(-mu)) / math.factorial(k)
    except: return 0.0

def calculate_predictions(res_df, h_tab, a_tab):
    if h_tab is None or a_tab is None or res_df is None or res_df.empty: return res_df
    try:
        avg_h_gf, avg_h_ga = h_tab['GF'].sum()/h_tab['GP'].sum(), h_tab['GA'].sum()/h_tab['GP'].sum()
        avg_a_gf, avg_a_ga = a_tab['GF'].sum()/a_tab['GP'].sum(), a_tab['GA'].sum()/a_tab['GP'].sum()
        
        # Keep GP info for the reliability check
        h_s, a_s = h_tab.set_index('Team'), a_tab.set_index('Team')
        
        h_s['Atk'], h_s['Def'] = (h_s['GF']/h_s['GP'])/avg_h_gf, (h_s['GA']/h_s['GP'])/avg_h_ga
        a_s['Atk'], a_s['Def'] = (a_s['GF']/a_s['GP'])/avg_a_gf, (a_s['GA']/a_s['GP'])/avg_a_ga
        
        h_m, a_m = {clean_team_name(n): n for n in h_s.index}, {clean_team_name(n): n for n in a_s.index}
        
        def pr(row):
            h_r, a_r = str(row['Home']).strip(), str(row['Away']).strip()
            ht = h_r if h_r in h_s.index else h_m.get(clean_team_name(h_r))
            at = a_r if a_r in a_s.index else a_m.get(clean_team_name(a_r))
            
            if not ht or not at: return pd.Series(["-", 0.0, "N/A", "N/A", 0.0, "N/A", 0, 0])
            
            # --- RELIABILITY DATA ---
            h_gp = int(h_s.loc[ht, 'GP'])
            a_gp = int(a_s.loc[at, 'GP'])
            
            lh, la = h_s.loc[ht, 'Atk'] * a_s.loc[at, 'Def'] * avg_h_gf, a_s.loc[at, 'Atk'] * h_s.loc[ht, 'Def'] * avg_a_gf
            m = np.outer([poisson_pmf(i, lh) for i in range(8)], [poisson_pmf(i, la) for i in range(8)])
            o25 = 1 - (m[0,0]+m[0,1]+m[0,2]+m[1,0]+m[1,1]+m[2,0])
            hw, d, aw = np.sum(np.tril(m,-1)), np.sum(np.diag(m)), np.sum(np.triu(m,1))
            best = max({'Home Win': hw, 'Draw': d, 'Away Win': aw}, key=lambda k: {'Home Win': hw, 'Draw': d, 'Away Win': aw}[k])
            hs, aws = np.unravel_index(np.argmax(m), m.shape)
            
            return pd.Series([f"{hs}-{aws}", float(o25), f"{o25:.1%}", best, float(max(hw,d,aw)), f"{max(hw,d,aw):.1%}", h_gp, a_gp])
            
        new = res_df.apply(pr, axis=1)
        res_df[['Score', 'O25_Raw', 'Prob O2.5', 'Winner', 'WProb_Raw', 'Win Prob', 'H_GP', 'A_GP']] = new
    except: pass
    return res_df

def load_full_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f: return json.load(f)
        except: return {}
    return {}

def save_to_cache(league, data):
    cache = load_full_cache(); cache[league] = data
    with open(CACHE_FILE, 'w') as f: json.dump(cache, f)

def fetch_data(league):
    try:
        r1 = requests.get(f"https://www.soccerstats.com/results.asp?league={league}&pmtype=bydate", headers=HEADERS, timeout=15)
        res_df = None
        for df in pd.read_html(StringIO(r1.text)):
            idx = -1
            for i in range(len(df.columns)):
                if df.iloc[:, i].astype(str).str.contains(r'(\d\s?[:\-]\s?\d|\b[vV]\b|\d{1,2}:\d{2})').sum() > 2: idx = i; break
            if idx > 0:
                mask = df.iloc[:, idx].astype(str).str.contains(r'(\d\s?[:\-]\s?\d|\b[vV]\b|\d{1,2}:\d{2})')
                df_f = df[mask].copy()
                if len(df_f) >= 2: res_df = pd.DataFrame({'Date': df_f.iloc[:, 0].astype(str), 'Home': df_f.iloc[:, idx-1].astype(str), 'Info': df_f.iloc[:, idx].astype(str), 'Away': df_f.iloc[:, idx+1].astype(str)}); break
        r2 = requests.get(f"https://www.soccerstats.com/homeaway.asp?league={league}", headers=HEADERS, timeout=15)
        h_t, a_t = None, None
        for df in pd.read_html(StringIO(r2.text)):
            tr = -1
            for i in range(min(5, len(df))):
                if 'GP' in [str(x).upper() for x in df.iloc[i].tolist()]: tr = i; break
            if tr != -1:
                cols = [str(x).upper() for x in df.iloc[tr].tolist()]
                d = df.iloc[tr+1:].copy()
                st_df = pd.DataFrame({'Team': d.iloc[:, 1], 'GP': pd.to_numeric(d.iloc[:, cols.index('GP')], errors='coerce'), 'GF': pd.to_numeric(d.iloc[:, cols.index('GF')], errors='coerce'), 'GA': pd.to_numeric(d.iloc[:, cols.index('GA')], errors='coerce')}).dropna()
                if h_t is None: h_t = st_df
                else: a_t = st_df; break
        if h_t is not None and a_t is not None and res_df is not None:
            data = calculate_predictions(res_df, h_t, a_t).to_dict('records')
            save_to_cache(league, data); return data
    except: pass
    return load_full_cache().get(league, [])

# --- UI ---
st.markdown("""<style>.match-card { background-color: #1E1E1E; border-radius: 12px; padding: 12px; margin-bottom: 15px; border: 1px solid #333; } .best-pick-badge { background-color: #2ECC71; color: #000; padding: 10px 20px; border-radius: 30px; font-weight: 900; font-size: 22px; display: inline-block; margin-bottom: 10px; } .bet-label { color: #888; font-size: 11px; text-transform: uppercase; font-weight: 800; } .team-text { color: #FFFFFF !important; font-weight: bold; font-size: 16px; } .score-box { background: #007BFF; color: yellow !important; padding: 3px 10px; border-radius: 5px; font-weight: bold; font-size: 18px; border: 2px solid yellow; } .upcoming-box { background: #444; color: #FFFFFF !important; padding: 3px 10px; border-radius: 5px; font-weight: bold; font-size: 16px; } .league-tag { color: #FFD700; font-weight: 800; font-size: 12px; margin-bottom: 5px; } .reliability-tag { color: #666; font-size: 10px; font-style: italic; }</style>""", unsafe_allow_html=True)
st.title("⚽ Soccer Smart Predictor")

if 'syncing' not in st.session_state: st.session_state.syncing = False
def global_sync():
    st.session_state.syncing = True
    progress_bar = st.progress(0); status_text = st.empty()
    for i, l in enumerate(LEAGUES):
        status_text.text(f"Syncing {l.upper()} ({i+1}/{len(LEAGUES)})...")
        fetch_data(l); progress_bar.progress((i + 1) / len(LEAGUES))
    status_text.text("✅ Global Sync Complete!"); time.sleep(2); status_text.empty(); progress_bar.empty(); st.session_state.syncing = False

tabs = st.tabs(["🚀 Best Picks", "📊 All Matches", "📈 Model Accuracy"])

with tabs[1]:
    col_a, col_b = st.columns([3, 1])
    with col_a: l_sel = st.selectbox("Select League", LEAGUES, index=LEAGUES.index("portugal") if "portugal" in LEAGUES else 0)
    with col_b: 
        if st.button("🔄 Sync This League"): st.cache_data.clear()
        if st.button("🌐 Sync ALL Leagues"): global_sync()
    data = fetch_data(l_sel)
    if not data: st.warning("Loading...")
    else:
        for g in data:
            is_s = is_genuine_score(g['Info'])
            s_h = f'<div class="score-box">{g["Info"]}</div>' if is_s else f'<div class="upcoming-box">🕒 {g["Info"]}</div>'
            st.markdown(f"""<div class="match-card" style="border-radius:10px 10px 0 0; margin-bottom:0;"><div style="display: flex; justify-content: space-between; align-items: center;"><div style="width: 38%; text-align: right;" class="team-text">{g['Home']}</div><div style="width: 24%; text-align: center;">{s_h}</div><div style="width: 38%; text-align: left;" class="team-text">{g['Away']}</div></div></div><div style="background-color: #252525; padding: 15px; border-radius: 0 0 10px 10px; border: 1px solid #333; border-top: none; margin-bottom: 20px;"><div style="display: flex; justify-content: space-around; align-items: center;"><div style="text-align:center; width:33%; border-right:1px solid #444;"><div class="bet-label">Outcome</div><div style="color: #2ECC71; font-size:18px; font-weight:900;">{g['Winner']}</div><div style="color: #2ECC71; font-size:12px; font-weight:bold;">{g['Win Prob']}</div></div><div style="text-align:center; width:33%; border-right:1px solid #444;"><div class="bet-label">Over 2.5</div><div style="color: #FF00FF; font-size:18px; font-weight:900;">{g['Prob O2.5']}</div><div style="color: #FF00FF; font-size:12px; font-weight:bold;">Prob.</div></div><div style="text-align:center; width:33%;"><div class="bet-label">Score</div><div style="color: #007BFF; font-size:18px; font-weight:900;">{g['Score']}</div><div style="color: #666; font-size:12px; font-weight:bold;">Forecast</div></div></div><div style="text-align:center; font-size:10px; color:#555; margin-top:5px;">Games Played (H/A): {g.get('H_GP',0)} / {g.get('A_GP',0)}</div></div>""", unsafe_allow_html=True)

with tabs[0]:
    st.header("💎 Top Picks (Next 5 Days)")
    st.info("⚠️ Only showing games where **both teams** have at least 5 games played (Home/Away).")
    if st.button("🚀 Find Global Picks Now"): global_sync()
    cache = load_full_cache()
    if not cache: st.info("Sync leagues to find picks.")
    else:
        best_p = []; now = datetime.now(); limit = now + timedelta(days=5)
        for l, games in cache.items():
            trig = LEAGUE_TRIGGERS.get(l, DEFAULT_TRIGGERS)
            for g in games:
                if not is_genuine_score(g['Info']):
                    # --- NEW RELIABILITY FILTER: MIN 5 GAMES ---
                    if int(g.get('H_GP', 0)) < 5 or int(g.get('A_GP', 0)) < 5: continue
                    
                    dt = parse_date(g['Date'])
                    if dt and not (now.date() <= dt.date() <= limit.date()): continue
                    w, o = float(g.get('WProb_Raw', 0)), float(g.get('O25_Raw', 0))
                    is_b, r, tag = False, "", ""
                    if w >= trig['win']: is_b, r, tag = True, g['Winner'], f"🔥 Confidence: {g['Win Prob']}"
                    elif o >= trig['o25']: is_b, r, tag = True, "OVER 2.5 GOALS", f"⚽ Probability: {g['Prob O2.5']}"
                    if is_b: g['L'], g['R'], g['T'] = l.upper(), r.upper(), tag; best_p.append(g)
        if not best_p: st.warning("No picks found meeting the 5-game criteria.")
        else:
            for g in sorted(best_p, key=lambda x: x['WProb_Raw'], reverse=True):
                st.markdown(f"""<div class="match-card" style="text-align: center; border: 2px solid #2ECC71;"><div class="league-tag">{g['L']} | {g['Date']}</div><div class="best-pick-badge">{g['R']}</div><div style="color: #2ECC71; font-weight: bold; margin-bottom: 15px;">{g['T']}</div><div style="display: flex; justify-content: space-between; align-items: center; background: #252525; padding: 12px; border-radius: 10px; margin-bottom: 10px;"><div style="width: 40%; text-align: right; color: white; font-weight: bold;">{g['Home']}</div><div style="width: 20%; color: #888;">VS</div><div style="width: 40%; text-align: left; color: white; font-weight: bold;">{g['Away']}</div></div><div style="display: flex; justify-content: space-around; font-size: 13px;"><span style="color: #666;">Score: <b style="color: #007BFF;">{g['Score']}</b></span><span style="color: #888;">GP (H/A): <b>{g['H_GP']}/{g['A_GP']}</b></span><span style="color: #666;">Time: <b style="color: #FFF;">{g['Info']}</b></span></div></div>""", unsafe_allow_html=True)

with tabs[2]:
    st.header("🏆 League Performance Ranking")
    cache = load_full_cache()
    if cache:
        rows = []
        for l, games in cache.items():
            wins, o25, tot = 0, 0, 0
            for g in games:
                if is_genuine_score(g['Info']):
                    h, a = map(int, re.findall(r'\d+', g['Info'])); tot += 1
                    if ('Home Win' if h>a else ('Draw' if h==a else 'Away Win')) == g['Winner']: wins += 1
                    if ((h+a)>2.5) == (g.get('O25_Raw', 0)>0.5): o25 += 1
            if tot > 0: rows.append({'League': l.upper(), 'Finished Games': tot, 'WinVal': wins/tot, 'Win Accuracy': f"{(wins/tot):.1%}", 'Over 2.5 Accuracy': f"{(o25/tot):.1%}"})
        if rows:
            df_acc = pd.DataFrame(rows).sort_values(by='WinVal', ascending=False)
            st.table(df_acc.set_index('League')[['Finished Games', 'Win Accuracy', 'Over 2.5 Accuracy']])
            st.success(f"🌟 **Top Performer**: {df_acc.iloc[0]['League']} ({df_acc.iloc[0]['Win Accuracy']})")
