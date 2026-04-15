import streamlit as st
import pandas as pd
from collections import Counter
import numpy as np

# Page Config
st.set_page_config(page_title="Super-AI Total Predictor", layout="wide")

st.title("🛡️ Super-AI: Total Combination Predictor")
st.write("सभी 6 तरीकों (Patterns, Sequences, Trends, Bar, Multi-Shift) का निचोड़।")

# 1. Master Configurations
master_patterns = [0, -18, -16, -26, -32, -1, -4, -11, -15, -10, -51, -50, 15, 5, -5, -55, 1, 10, 11, 51, 55, -40]
shifts = ['DS', 'FD', 'GD', 'GL', 'DB', 'SG']

# 2. File Upload + Column Detection
uploaded_file = st.sidebar.file_uploader("Upload Data File", type=['csv', 'xlsx'])

if uploaded_file:
    # Load data
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    st.sidebar.write(f"📊 Loaded {len(df)} rows")
    st.sidebar.dataframe(df.head(3))
    
    # Auto-detect shift columns
    available_shifts = []
    for col in df.columns:
        if col in shifts:
            available_shifts.append(col)
    
    if not available_shifts:
        st.error("❌ शिफ्ट कॉलम्स (`DS`, `FD`, `GD`, `GL`, `DB`, `SG`) नहीं मिले। कृपया सही कॉलम नाम चेक करें।")
        st.sidebar.write("आपके डेटा के कॉलम:", list(df.columns))
        st.stop()
    
    st.sidebar.success(f"✅ पाए गए शिफ्ट्स: {', '.join(available_shifts)}")
    
    # Convert to numeric
    for col in available_shifts:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove NaN rows for prediction
    df_clean = df[available_shifts].dropna()
    
    if len(df_clean) < 2:
        st.error("❌ कम से कम 2 दिन का डेटा चाहिए।")
        st.stop()

    # Get Latest Numbers (Today & Yesterday)
    today_row = df_clean.iloc[-1]
    yesterday_row = df_clean.iloc[-2] if len(df_clean) > 1 else None
    
    today_nums = {col: int(today_row[col]) for col in available_shifts}
    yesterday_nums = {col: int(yesterday_row[col]) for col in available_shifts} if yesterday_row is not None else {}

    st.success(f"🎯 आज के नंबर: {today_nums}")
    
    # --- THE SUPER SCORING ENGINE (FIXED) ---
    scores = np.zeros(100)  # 0 to 99 scores

    # A. Method 1: Master Pattern Score (Weight: 1 per hit)
    for val in today_nums.values():
        for p in master_patterns:
            res = int((val + p) % 100)
            scores[res] += 1

    # B. Method 2: History Transition (Weight: 3) - FIXED
    for s_name, s_val in today_nums.items():
        history_hits = df_clean[df_clean[s_name] == s_val].index
        next_vals = []
        for idx in history_hits:
            if idx + 1 < len(df_clean):
                v_next = df_clean.iloc[idx + 1][s_name]
                if not pd.isna(v_next):
                    next_vals.append(int(v_next))
        
        if next_vals:
            top_next = [n for n, c in Counter(next_vals).most_common(3)]
            for tn in top_next:
                scores[tn] += 3

    # C. Method 3: Weekly & Monthly Trend (Weight: 2) - SIMPLIFIED & FIXED
    recent_patterns = []
    for i in range(min(30, len(df_clean))):
        if i > 0:
            for col in available_shifts:
                curr_val = int(df_clean.iloc[i][col])
                prev_val = int(df_clean.iloc[i-1][col])
                pattern = int((curr_val - prev_val) % 100)
                recent_patterns.append(pattern)
    
    top_patterns = [p for p, c in Counter(recent_patterns).most_common(10)]
    for v in today_nums.values():
        for p in top_patterns[:5]:
            scores[int((v + p) % 100)] += 2

    # D. Method 4: Sequence Logic (Weight: 2) - FIXED
    for s_name, t_val in today_nums.items():
        if s_name in yesterday_nums:
            curr_p = int((t_val - yesterday_nums[s_name]) % 100)
            # Simple sequence rules
            sequence_rules = {
                84: 0,   # -16 -> 0
                89: 96,  # -11 -> -4
                0: 89,   # 0 -> -11
            }
            if curr_p in sequence_rules:
                next_num = int((t_val + sequence_rules[curr_p]) % 100)
                scores[next_num] += 2

    # --- DISPLAY FINAL RESULTS ---
    st.header("🎯 Combined Super Prediction")

    # Final Results Table
    final_results = []
    for num in range(100):
        if scores[num] > 0:
            final_results.append({
                "Number": f"{num:02d}",
                "Total Score": int(scores[num]),
                "Confidence": f"{min(95, scores[num]*6):.0f}%"
            })
    
    final_df = pd.DataFrame(final_results).sort_values(by="Total Score", ascending=False)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🏆 Top 10 Recommended Numbers")
        st.dataframe(final_df.head(10), use_container_width=True)
        
        # Top Prediction
        top_pred = final_df.iloc[0]
        st.markdown(f"""
        <div style='background: linear-gradient(45deg, #4CAF50, #45a049); 
                    color: white; padding: 20px; border-radius: 15px; text-align: center;'>
            <h2>🎯 #1 Prediction: {top_pred['Number']}</h2>
            <h3>Confidence: {top_pred['Confidence']}</h3>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.subheader("💡 Scoring Logic")
        st.markdown("""
        - **Pattern Hits (1 pt):** 22 master patterns  
        - **History Transition (3 pts):** आज के नंबर के बाद क्या आया  
        - **Trend Analysis (2 pts):** हाल के 30 दिन के hot patterns  
        - **Sequence Rules (2 pts):** Pattern-to-pattern logic  
        """)

    st.divider()
    st.header("🔍 Shift-wise Analysis")
    shift_results = []
    for s_name, s_val in today_nums.items():
        shift_scores = []
        for p in master_patterns:
            n = int((s_val + p) % 100)
            shift_scores.append({"Number": f"{n:02d}", "Score": scores[n]})
        best_shift = sorted(shift_scores, key=lambda x: x['Score'], reverse=True)[0]
        shift_results.append({
            "Shift": s_name,
            "Today": f"{s_val:02d}",
            "Best Prediction": best_shift['Number'],
            "Score": best_shift['Score']
        })
    
    st.dataframe(pd.DataFrame(shift_results), use_container_width=True)

else:
    st.info("👈 Sidebar में अपनी Excel/CSV फाइल अपलोड करें। यह सिस्टम आपके पुराने डेटा को खुद ही समझ लेगा।")
    
    st.markdown("""
    **📋 Expected File Format:**
    ```
    Date | DS | FD | GD | GL | DB | SG
    01/01| 23 | 45 | 67 | 89 | 12 | 34
    02/01| 34 | 56 | 78 | 90 | 23 | 45
    ```
    """)
