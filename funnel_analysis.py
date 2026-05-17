"""
Marketing & Lead Funnel Analysis
=================================
Simulates a realistic B2B SaaS marketing funnel across channels,
computes conversion rates, drop-offs, and exports visualizations as PNGs.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

# ── Reproducibility ──────────────────────────────────────────────────────────
np.random.seed(42)

# ══════════════════════════════════════════════════════════════════════════════
# 1.  SYNTHETIC DATA GENERATION
# ══════════════════════════════════════════════════════════════════════════════

channels = ["Organic Search", "Paid Search", "Social Media", "Email", "Referral", "Direct"]
months   = pd.date_range("2024-01-01", periods=12, freq="MS").strftime("%b %Y").tolist()

# Base visitors per channel per month
base_visitors = {
    "Organic Search": 12000,
    "Paid Search":     8000,
    "Social Media":    6000,
    "Email":           4000,
    "Referral":        3000,
    "Direct":          5000,
}

# Conversion rates at each funnel stage (channel-specific)
conv_rates = {
    #                  visitor→lead  lead→MQL   MQL→SQL    SQL→customer
    "Organic Search": [0.042, 0.55, 0.42, 0.28],
    "Paid Search":    [0.065, 0.48, 0.38, 0.22],
    "Social Media":   [0.031, 0.40, 0.30, 0.18],
    "Email":          [0.085, 0.62, 0.50, 0.35],
    "Referral":       [0.075, 0.68, 0.55, 0.40],
    "Direct":         [0.055, 0.52, 0.44, 0.30],
}

rows = []
for m_idx, month in enumerate(months):
    growth = 1 + m_idx * 0.015          # 1.5 % MoM growth
    for ch in channels:
        visitors  = int(base_visitors[ch] * growth * np.random.uniform(0.92, 1.08))
        cr        = conv_rates[ch]
        leads     = int(visitors   * cr[0] * np.random.uniform(0.90, 1.10))
        mqls      = int(leads      * cr[1] * np.random.uniform(0.90, 1.10))
        sqls      = int(mqls       * cr[2] * np.random.uniform(0.90, 1.10))
        customers = int(sqls       * cr[3] * np.random.uniform(0.90, 1.10))
        acv       = np.random.randint(800, 4000)          # avg contract value
        revenue   = customers * acv
        rows.append(dict(
            month=month, channel=ch,
            visitors=visitors, leads=leads, mqls=mqls,
            sqls=sqls, customers=customers,
            avg_contract_value=acv, revenue=revenue
        ))

df = pd.DataFrame(rows)

# Derived rates
df["v2l_rate"]  = df["leads"]     / df["visitors"]
df["l2m_rate"]  = df["mqls"]      / df["leads"]
df["m2s_rate"]  = df["sqls"]      / df["mqls"]
df["s2c_rate"]  = df["customers"] / df["sqls"]
df["v2c_rate"]  = df["customers"] / df["visitors"]   # end-to-end

df.to_csv("/home/claude/funnel_data.csv", index=False)
print("✅  Dataset saved →  funnel_data.csv")
print(df.head())

# ── Aggregate helpers ─────────────────────────────────────────────────────────
agg_ch = df.groupby("channel")[
    ["visitors","leads","mqls","sqls","customers","revenue"]
].sum().reset_index()
agg_ch["v2l"]  = agg_ch["leads"]     / agg_ch["visitors"]
agg_ch["l2m"]  = agg_ch["mqls"]      / agg_ch["leads"]
agg_ch["m2s"]  = agg_ch["sqls"]      / agg_ch["mqls"]
agg_ch["s2c"]  = agg_ch["customers"] / agg_ch["sqls"]
agg_ch["e2e"]  = agg_ch["customers"] / agg_ch["visitors"]

agg_mo = df.groupby("month")[
    ["visitors","leads","mqls","sqls","customers","revenue"]
].sum().reset_index()
# keep chronological order
month_order = months
agg_mo["month"] = pd.Categorical(agg_mo["month"], categories=month_order, ordered=True)
agg_mo = agg_mo.sort_values("month")

# ══════════════════════════════════════════════════════════════════════════════
# 2.  STYLE SETUP
# ══════════════════════════════════════════════════════════════════════════════

DARK   = "#0d1117"
CARD   = "#161b22"
ACCENT = "#58a6ff"
GREEN  = "#3fb950"
ORANGE = "#f78166"
PURPLE = "#bc8cff"
YELLOW = "#e3b341"
TEAL   = "#39d353"
PINK   = "#ff7b72"
WHITE  = "#e6edf3"
GRAY   = "#8b949e"

CH_COLORS = {
    "Organic Search": ACCENT,
    "Paid Search":    ORANGE,
    "Social Media":   PURPLE,
    "Email":          GREEN,
    "Referral":       YELLOW,
    "Direct":         TEAL,
}

plt.rcParams.update({
    "figure.facecolor":  DARK,
    "axes.facecolor":    CARD,
    "axes.edgecolor":    "#30363d",
    "axes.labelcolor":   WHITE,
    "axes.titlecolor":   WHITE,
    "xtick.color":       GRAY,
    "ytick.color":       GRAY,
    "text.color":        WHITE,
    "grid.color":        "#21262d",
    "grid.linestyle":    "--",
    "grid.alpha":        0.6,
    "font.family":       "DejaVu Sans",
    "legend.facecolor":  CARD,
    "legend.edgecolor":  "#30363d",
})

def save(fig, name):
    path = f"/home/claude/{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close(fig)
    print(f"  📊  Saved {path}")

# ══════════════════════════════════════════════════════════════════════════════
# 3.  CHART 1 — OVERALL FUNNEL (horizontal waterfall)
# ══════════════════════════════════════════════════════════════════════════════

stages  = ["Visitors", "Leads", "MQLs", "SQLs", "Customers"]
totals  = [df[c].sum() for c in ["visitors","leads","mqls","sqls","customers"]]
drop_pct = [100.0] + [totals[i]/totals[0]*100 for i in range(1, len(totals))]
colors  = [ACCENT, PURPLE, ORANGE, YELLOW, GREEN]

fig, ax = plt.subplots(figsize=(12, 5))
fig.patch.set_facecolor(DARK)
ax.set_facecolor(CARD)

bars = ax.barh(stages[::-1], [t/1000 for t in totals[::-1]],
               color=colors[::-1], height=0.55, edgecolor="none")

for bar, tot, pct in zip(bars, totals[::-1], drop_pct[::-1]):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f"{tot:,.0f}  ({pct:.1f}%)", va="center", fontsize=11, color=WHITE)

ax.set_xlabel("Volume (thousands)", fontsize=11)
ax.set_title("Full-Year Funnel: Visitors → Customers", fontsize=15, fontweight="bold",
             pad=15, color=WHITE)
ax.set_xlim(0, max(t/1000 for t in totals)*1.28)
ax.grid(axis="x")
ax.spines[["top","right","left","bottom"]].set_visible(False)
plt.tight_layout()
save(fig, "chart1_overall_funnel")

# ══════════════════════════════════════════════════════════════════════════════
# 4.  CHART 2 — CONVERSION RATES BY CHANNEL (grouped bar)
# ══════════════════════════════════════════════════════════════════════════════

rate_cols  = ["v2l","l2m","m2s","s2c"]
rate_labels= ["Visitor→Lead","Lead→MQL","MQL→SQL","SQL→Customer"]
x          = np.arange(len(agg_ch))
w          = 0.18
rate_colors= [ACCENT, PURPLE, ORANGE, GREEN]

fig, ax = plt.subplots(figsize=(13, 6))
for i, (col, lbl, col_c) in enumerate(zip(rate_cols, rate_labels, rate_colors)):
    offsets = x + (i - 1.5) * w
    rects   = ax.bar(offsets, agg_ch[col]*100, w, label=lbl, color=col_c, alpha=0.88)

ax.set_xticks(x)
ax.set_xticklabels(agg_ch["channel"], rotation=20, ha="right", fontsize=10)
ax.set_ylabel("Conversion Rate (%)", fontsize=11)
ax.set_title("Conversion Rates by Channel at Each Funnel Stage", fontsize=14,
             fontweight="bold", pad=14)
ax.legend(fontsize=9)
ax.grid(axis="y")
ax.spines[["top","right","left","bottom"]].set_visible(False)
plt.tight_layout()
save(fig, "chart2_conversion_by_channel")

# ══════════════════════════════════════════════════════════════════════════════
# 5.  CHART 3 — DROP-OFF HEATMAP
# ══════════════════════════════════════════════════════════════════════════════

heat_data = agg_ch.set_index("channel")[rate_cols].copy() * 100

fig, ax = plt.subplots(figsize=(9, 5))
im = ax.imshow(heat_data.values, cmap="YlOrRd", aspect="auto", vmin=15, vmax=90)

ax.set_xticks(range(len(rate_labels)))
ax.set_xticklabels(rate_labels, fontsize=11)
ax.set_yticks(range(len(heat_data)))
ax.set_yticklabels(heat_data.index, fontsize=11)

for i in range(len(heat_data)):
    for j in range(len(rate_cols)):
        val = heat_data.values[i, j]
        ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                fontsize=10, color="black" if val > 50 else "white", fontweight="bold")

cb = fig.colorbar(im, ax=ax, shrink=0.85)
cb.ax.yaxis.set_tick_params(color=WHITE)
plt.setp(cb.ax.yaxis.get_ticklabels(), color=WHITE)
cb.set_label("Conv. Rate (%)", color=WHITE)

ax.set_title("Conversion Rate Heatmap — Channel × Stage", fontsize=14,
             fontweight="bold", pad=14)
ax.spines[["top","right","left","bottom"]].set_visible(False)
fig.patch.set_facecolor(DARK)
ax.set_facecolor(DARK)
plt.tight_layout()
save(fig, "chart3_dropoff_heatmap")

# ══════════════════════════════════════════════════════════════════════════════
# 6.  CHART 4 — MONTHLY TRENDS (multi-line)
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Monthly Funnel Trends", fontsize=15, fontweight="bold", color=WHITE, y=1.01)

# Left: volume
ax = axes[0]
for col, color, lbl in zip(
    ["visitors","leads","mqls","sqls","customers"],
    [ACCENT, PURPLE, ORANGE, YELLOW, GREEN],
    ["Visitors","Leads","MQLs","SQLs","Customers"]
):
    ax.plot(range(12), agg_mo[col]/1000, color=color, linewidth=2.2,
            marker="o", markersize=4, label=lbl)
ax.set_xticks(range(12))
ax.set_xticklabels([m[:3] for m in month_order], fontsize=9)
ax.set_ylabel("Volume (thousands)")
ax.set_title("Funnel Volume by Month", fontsize=12)
ax.legend(fontsize=9)
ax.grid(True)
ax.spines[["top","right","left","bottom"]].set_visible(False)

# Right: revenue
ax = axes[1]
ax.bar(range(12), agg_mo["revenue"]/1000, color=ACCENT, alpha=0.75, edgecolor="none")
ax.plot(range(12), agg_mo["revenue"]/1000, color=GREEN, linewidth=2, marker="o", markersize=4)
ax.set_xticks(range(12))
ax.set_xticklabels([m[:3] for m in month_order], fontsize=9)
ax.set_ylabel("Revenue ($ thousands)")
ax.set_title("Monthly Revenue", fontsize=12)
ax.grid(axis="y")
ax.spines[["top","right","left","bottom"]].set_visible(False)

plt.tight_layout()
save(fig, "chart4_monthly_trends")

# ══════════════════════════════════════════════════════════════════════════════
# 7.  CHART 5 — CHANNEL REVENUE & END-TO-END CONV (bubble chart)
# ══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(10, 6))

for _, row in agg_ch.iterrows():
    color = CH_COLORS[row["channel"]]
    size  = row["revenue"] / 5000
    ax.scatter(row["v2l"]*100, row["e2e"]*100, s=size, color=color,
               alpha=0.85, edgecolors="white", linewidths=0.6, zorder=5)
    ax.annotate(row["channel"], (row["v2l"]*100, row["e2e"]*100),
                textcoords="offset points", xytext=(8, 4),
                fontsize=9.5, color=color)

ax.set_xlabel("Visitor → Lead Rate (%)", fontsize=11)
ax.set_ylabel("End-to-End Conversion Rate (%)", fontsize=11)
ax.set_title("Channel Quality: Lead Rate vs E2E Conversion\n(bubble size = total revenue)",
             fontsize=13, fontweight="bold", pad=12)
ax.grid(True)
ax.spines[["top","right","left","bottom"]].set_visible(False)
plt.tight_layout()
save(fig, "chart5_channel_bubble")

# ══════════════════════════════════════════════════════════════════════════════
# 8.  CHART 6 — SUMMARY KPI DASHBOARD (text-heavy infographic)
# ══════════════════════════════════════════════════════════════════════════════

total_visitors  = df["visitors"].sum()
total_leads     = df["leads"].sum()
total_mqls      = df["mqls"].sum()
total_sqls      = df["sqls"].sum()
total_customers = df["customers"].sum()
total_revenue   = df["revenue"].sum()
best_ch         = agg_ch.loc[agg_ch["e2e"].idxmax(), "channel"]
worst_ch        = agg_ch.loc[agg_ch["e2e"].idxmin(), "channel"]
biggest_drop    = "Visitor → Lead"   # widest drop universally

fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor(DARK)
ax.set_facecolor(DARK)
ax.axis("off")

kpis = [
    ("Total Visitors",   f"{total_visitors:,.0f}",   ACCENT),
    ("Total Leads",      f"{total_leads:,.0f}",      PURPLE),
    ("Total MQLs",       f"{total_mqls:,.0f}",       ORANGE),
    ("Total SQLs",       f"{total_sqls:,.0f}",       YELLOW),
    ("Total Customers",  f"{total_customers:,.0f}",  GREEN),
    ("Total Revenue",    f"${total_revenue/1e6:.2f}M", TEAL),
]

cols  = 3
rows_ = 2
cell_w = 1/cols
cell_h = 0.38

for idx, (label, value, color) in enumerate(kpis):
    c = idx % cols
    r = idx // cols
    cx = c * cell_w + cell_w/2
    cy = 0.97 - r * cell_h

    ax.text(cx, cy, value, transform=ax.transAxes,
            ha="center", va="top", fontsize=28, fontweight="bold", color=color)
    ax.text(cx, cy - 0.09, label, transform=ax.transAxes,
            ha="center", va="top", fontsize=11, color=GRAY)

# Divider line
ax.plot([0, 1], [0.20, 0.20], color="#30363d", linewidth=1, transform=ax.transAxes)

insights = [
    f"📌  Biggest drop-off: {biggest_drop} stage — invest in landing page CRO.",
    f"🏆  Best channel (E2E conv.): {best_ch}  |  Lowest: {worst_ch} — reallocate budget.",
    f"💡  Email & Referral show highest lead quality (SQL→Customer > 35%).",
    f"📈  Paid Search generates volume but converts poorly — review ad targeting.",
    f"🚀  End-to-end conversion: {total_customers/total_visitors*100:.2f}%  |  Revenue / Lead: ${total_revenue/total_leads:,.0f}",
]

for i, txt in enumerate(insights):
    ax.text(0.03, 0.17 - i*0.035, txt, transform=ax.transAxes,
            ha="left", va="top", fontsize=10.5, color=WHITE)

ax.set_title("Funnel KPI Summary — Full Year 2024", fontsize=16,
             fontweight="bold", color=WHITE, pad=18, loc="left")
plt.tight_layout()
save(fig, "chart6_kpi_summary")

print("\n✅  All 6 charts saved successfully.")
print(f"\n{'='*55}")
print("KEY INSIGHTS")
print(f"{'='*55}")
print(f"  Total Visitors  : {total_visitors:>12,.0f}")
print(f"  Total Leads     : {total_leads:>12,.0f}  ({total_leads/total_visitors*100:.1f}%)")
print(f"  Total MQLs      : {total_mqls:>12,.0f}  ({total_mqls/total_leads*100:.1f}% of leads)")
print(f"  Total SQLs      : {total_sqls:>12,.0f}  ({total_sqls/total_mqls*100:.1f}% of MQLs)")
print(f"  Total Customers : {total_customers:>12,.0f}  ({total_customers/total_sqls*100:.1f}% of SQLs)")
print(f"  Total Revenue   : ${total_revenue/1e6:>10.2f}M")
print(f"  Best Channel    : {best_ch}")
print(f"  Weakest Channel : {worst_ch}")
print(f"{'='*55}")
