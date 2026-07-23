import streamlit as st
import pandas as pd
import json
import graphviz
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# --- CẤU HÌNH APPLE VIBE ---
st.set_page_config(page_title="CausClass Analytics", page_icon="📊", layout="wide", initial_sidebar_state="collapsed")

# --- HACK CSS ---
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 95%;
        font-family: -apple-system, BlinkMacSystemFont, sans-serif;
    }

    h1, h2, h3, h4 { font-weight: 600 !important; letter-spacing: -0.015em; color: #1D1D1F; }

    div[data-testid="stMetricValue"] { font-size: 2.2rem; font-weight: 700; color: #1D1D1F; }
    div[data-testid="stMetricLabel"] { font-size: 0.9rem; color: #86868B; text-transform: uppercase; }

    /* Card trắng cho các cột chứa kết quả */
    div[data-testid="column"] {
        background-color: #FFFFFF;
        border-radius: 16px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
        padding: 1.5rem 1rem;
        border: 1px solid rgba(0,0,0,0.03);
    }

    .stGraphVizChart > div > svg { border-radius: 16px; background: #FFFFFF; }

    button[data-testid="baseButton-primary"] {
        border-radius: 12px; font-weight: 600; background-color: #007AFF; transition: all 0.2s ease;
    }
    button[data-testid="baseButton-primary"]:hover {
        background-color: #0056b3; box-shadow: 0 4px 12px rgba(0, 122, 255, 0.3);
    }
</style>
""", unsafe_allow_html=True)

st.title("CausClass Observation")
st.markdown("<p style='color: #86868B; font-size: 1.2rem; margin-top: -15px; font-weight: 500;'>Causal Discovery & Temporal Dynamics</p>", unsafe_allow_html=True)
st.divider()

# --- HÀM ĐỌC FILE ---
@st.cache_data
def load_all_data():
    try:
        with open('stats.json', 'r', encoding='utf-8') as f:
            stats_data = json.load(f)
        with open('report.md', 'r', encoding='utf-8') as f:
            report_md = f.read()
        df_graph = pd.read_csv('graph.csv')
        df_ts = pd.read_csv('time_series.csv')
        return stats_data, report_md, df_graph, df_ts
    except FileNotFoundError as e:
        st.error(f"Lỗi: Không tìm thấy file dữ liệu! {e.filename}")
        return None, None, None, None

# --- UI ---
uploaded_file = st.file_uploader("Drop video file here or browse", type=['mp4', 'mov'])

if uploaded_file is not None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("File Name", uploaded_file.name)
    c2.metric("Duration", "18m 20s")
    c3.metric("Resolution", "1080p HD")
    c4.metric("Time Bins", "276 (4s/bin)")

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Analyze Video Dynamics", type="primary", use_container_width=True):

        stats_data, report_md, df_graph, df_ts = load_all_data()

        if stats_data is not None:
            st.divider()
            # Đổi tỷ lệ cột: Cột trái (0.9), Cột giữa rộng nhất (1.7), Cột phải (1.4)
            col_left, col_mid, col_right = st.columns([0.9, 1.7, 1.4])

            # --- 1. CỘT TRÁI: THỐNG KÊ & GRAPH ---
            with col_left:
                st.subheader("Descriptive Stats")
                ranked_behaviors = stats_data.get("behaviors_ranked_by_mean_detected_share", [])
                df_stats = pd.DataFrame({
                    "Behavior": [item["behavior"] for item in ranked_behaviors],
                    "Mean (%)": [round(item["mean_detected_behavior_share_pct"], 2) for item in ranked_behaviors],
                    "Max (%)": [round(item["max_detected_behavior_share_pct"], 2) for item in ranked_behaviors],
                    "Std": [round(item["std_detected_behavior_share_pct"], 2) for item in ranked_behaviors],
                    "Active Bins": [f"{round(item['active_bins_pct'], 1)}%" for item in ranked_behaviors]
                })
                st.dataframe(df_stats, hide_index=True, use_container_width=True)
                st.caption(f"* Note: {stats_data.get('important_measurement_warning', '')}")

                st.markdown("<br>", unsafe_allow_html=True)
                st.subheader("Temporal Graph")

                graph = graphviz.Digraph(engine='dot')
                graph.attr(bgcolor='transparent', rankdir='LR')
                graph.attr('node', shape='box', style='rounded,filled', fillcolor='white', fontname='Helvetica-Bold', color='#E5E5EA', penwidth='2')
                graph.attr('edge', fontname='Helvetica', fontsize='10')

                nodes = set(df_graph['source']).union(set(df_graph['target']))
                for node in nodes: graph.node(node)

                for _, row in df_graph.iterrows():
                    weight = round(row['weight'], 3)
                    sign = row['sign']
                    if sign == '+':
                        color, label, style = '#34C759', f" {sign}{weight}\n(Positive)", 'solid'
                    else:
                        color, label, style = '#FF3B30', f" {weight}\n(Negative)", 'dashed'
                    graph.edge(row['source'], row['target'], label=label, color=color, fontcolor=color, penwidth='2' if abs(weight) > 0.1 else '1', style=style)
                st.graphviz_chart(graph, use_container_width=True)

            # --- 2. CỘT GIỮA: REPORT (FULL CHIỀU CAO, KHÔNG SCROLL TRONG) ---
            with col_mid:
                st.subheader("Observation Report")
                # Bỏ tham số height, container sẽ tự động giãn theo content
                with st.container(border=True):
                    st.markdown(report_md)

            # --- 3. CỘT PHẢI: CHART (RỘNG RÃI, KHÔNG CHỒNG CHÉO) ---
            with col_right:
                st.subheader("Temporal Dynamics")

                behavior_cols = [col for col in df_ts.columns if col != 'time_bin_sec']

                # Tăng vertical_spacing để các đồ thị cách xa nhau ra
                fig = make_subplots(
                    rows=len(behavior_cols), cols=1,
                    subplot_titles=behavior_cols,
                    vertical_spacing=0.08
                )

                colors = ['#FF3B30', '#FF9500', '#FFCC00', '#34C759', '#007AFF', '#5856D6']

                time_secs = df_ts['time_bin_sec'].values
                tick_vals = time_secs[::15] if len(time_secs) > 15 else time_secs
                tick_text = [f"{int(x//60):02d}:{int(x%60):02d}" for x in tick_vals]

                for i, b_name in enumerate(behavior_cols):
                    fig.add_trace(
                        go.Scatter(
                            x=time_secs,
                            y=df_ts[b_name],
                            mode='lines',
                            line=dict(color=colors[i % len(colors)], width=2),
                            fill='tozeroy',
                            name=b_name
                        ),
                        row=i+1, col=1
                    )

                # Kéo height lên 1600px để 6 biểu đồ có không gian thở
                fig.update_layout(
                    height=1600, margin=dict(l=0, r=0, t=30, b=30),
                    showlegend=False, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
                )

                fig.update_xaxes(
                    tickmode='array', tickvals=tick_vals, ticktext=tick_text,
                    showgrid=False, zeroline=False
                )
                fig.update_yaxes(showgrid=True, gridcolor='rgba(200, 200, 200, 0.2)', zeroline=False)

                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
