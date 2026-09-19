"""
dashboard/components/theme.py — Nautical Command Center Styling & Theme.

Provides custom CSS injection and styled HTML UI components for the
NEMO (Nautical Eco-Monitoring Ocean system) dashboard.
"""
import streamlit as st


def inject_custom_css():
    """Inject high-tech oceanic command-center CSS."""
    st.markdown(
        """
        <style>
        /* ========================================================= */
        /* NEMO TACTICAL OCEANOGRAPHIC STYLING                       */
        /* ========================================================= */

        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

        :root {
            --bg-primary: #080D1A;
            --bg-secondary: #0E1626;
            --bg-card: #121E36;
            --border-glow: rgba(0, 240, 255, 0.25);
            --border-subtle: rgba(255, 255, 255, 0.08);
            --cyan-glow: #00F0FF;
            --cyan-dim: #008B99;
            --emerald-alert: #10B981;
            --amber-warn: #F59E0B;
            --rose-danger: #F43F5E;
            --text-main: #E2E8F0;
            --text-muted: #94A3B8;
        }

        /* General page adjustments */
        .stApp {
            background-color: #070B14;
            color: #E2E8F0;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: #0B1120;
            border-right: 1px solid rgba(0, 240, 255, 0.12);
        }

        section[data-testid="stSidebar"] hr {
            border-color: rgba(255, 255, 255, 0.07);
        }

        /* Header bar banner */
        .nemo-header-container {
            background: linear-gradient(135deg, rgba(14, 26, 50, 0.9) 0%, rgba(10, 18, 36, 0.95) 100%);
            border: 1px solid rgba(0, 240, 255, 0.25);
            border-radius: 12px;
            padding: 20px 24px;
            margin-bottom: 22px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4), inset 0 0 15px rgba(0, 240, 255, 0.05);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .nemo-header-title {
            font-size: 26px;
            font-weight: 700;
            color: #FFFFFF;
            letter-spacing: -0.5px;
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .nemo-header-subtitle {
            font-size: 13px;
            color: #00F0FF;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            font-weight: 600;
            margin-top: 4px;
            font-family: 'JetBrains Mono', monospace;
        }

        /* Metric cards */
        .nemo-metric-card {
            background: #0E172A;
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 10px;
            padding: 16px 18px;
            transition: all 0.2s ease-in-out;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.25);
        }

        .nemo-metric-card:hover {
            border-color: rgba(0, 240, 255, 0.4);
            box-shadow: 0 4px 18px rgba(0, 240, 255, 0.15);
            transform: translateY(-1px);
        }

        .nemo-metric-title {
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #94A3B8;
            font-family: 'JetBrains Mono', monospace;
        }

        .nemo-metric-value {
            font-size: 28px;
            font-weight: 700;
            color: #F8FAFC;
            margin: 6px 0 2px 0;
            font-family: 'Inter', sans-serif;
        }

        .nemo-metric-delta {
            font-size: 12px;
            color: #10B981;
            font-weight: 500;
        }

        /* Status Badges */
        .status-badge {
            display: inline-flex;
            align-items: center;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            font-family: 'JetBrains Mono', monospace;
        }

        .status-verified {
            background: rgba(16, 185, 129, 0.15);
            color: #34D399;
            border: 1px solid rgba(16, 185, 129, 0.4);
        }

        .status-rejected {
            background: rgba(244, 63, 94, 0.15);
            color: #FB7185;
            border: 1px solid rgba(244, 63, 94, 0.4);
        }

        .status-candidate {
            background: rgba(245, 158, 11, 0.15);
            color: #FBBF24;
            border: 1px solid rgba(245, 158, 11, 0.4);
        }

        /* Custom Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: #0B1324;
            padding: 6px;
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.06);
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            padding: 8px 18px;
            color: #94A3B8;
            font-weight: 600;
            font-size: 13px;
        }

        .stTabs [aria-selected="true"] {
            background-color: #1E293B !important;
            color: #00F0FF !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        }

        /* Telemetry chip */
        .telemetry-chip {
            background: rgba(0, 240, 255, 0.07);
            border: 1px solid rgba(0, 240, 255, 0.2);
            padding: 4px 10px;
            border-radius: 6px;
            color: #00F0FF;
            font-size: 12px;
            font-family: 'JetBrains Mono', monospace;
            display: inline-block;
            margin-right: 6px;
        }

        /* Pulse dot for live feed indicator */
        .pulse-dot {
            width: 8px;
            height: 8px;
            background-color: #10B981;
            border-radius: 50%;
            display: inline-block;
            box-shadow: 0 0 10px #10B981;
            animation: pulse-animation 2s infinite;
        }

        @keyframes pulse-animation {
            0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
            70% { transform: scale(1.1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
            100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(mission_id: str = "NEMO-ALPHA", status: str = "MISSION ARCHIVE"):
    """Render the master mission operations header banner."""
    st.markdown(
        f"""
        <div class="nemo-header-container">
            <div>
                <div class="nemo-header-title">
                    <span style="color: #00F0FF; font-size: 28px;">◈</span>
                    <span>NEMO <strong>COMMAND DECK</strong></span>
                    <span class="telemetry-chip">{mission_id}</span>
                </div>
                <div class="nemo-header-subtitle">
                    AUV Acoustic Payload Telemetry • Dual-Stage AI Marine Debris Analysis
                </div>
            </div>
            <div style="text-align: right;">
                <div style="display: flex; align-items: center; gap: 8px; justify-content: flex-end;">
                    <span class="pulse-dot"></span>
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #10B981; font-weight: 600;">
                        {status}
                    </span>
                </div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #64748B; margin-top: 4px;">
                    ACOUSTIC ENGINE: RCDI-YOLO 1C + MOBILENET-V3
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(title: str, value: str, delta: str = None, delta_color: str = "#10B981"):
    """Render a styled high-tech KPI card."""
    delta_html = f'<div class="nemo-metric-delta" style="color: {delta_color};">{delta}</div>' if delta else ''
    return f"""
    <div class="nemo-metric-card">
        <div class="nemo-metric-title">{title}</div>
        <div class="nemo-metric-value">{value}</div>
        {delta_html}
    </div>
    """


def get_status_badge(status_str: str) -> str:
    """Return an HTML badge for a detection status."""
    st_clean = str(status_str).lower()
    if "verified" in st_clean:
        return '<span class="status-badge status-verified">● VERIFIED DEBRIS</span>'
    elif "rejected" in st_clean:
        return '<span class="status-badge status-rejected">✕ NATURAL STRUCTURE</span>'
    else:
        return '<span class="status-badge status-candidate">▲ CANDIDATE</span>'
