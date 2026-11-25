"""Draws docs/outlier_experiment.jpg — the experiment as five steps + three endings, with life parallels."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

INK, MUTED, FAINT = "#2C2C2A", "#6B6A64", "#D3D1C7"
BLUE, RED, CELL, BOT, HUMAN = "#2F7BD6", "#D9483B", "#7A8F3A", "#0F6E56", "#C2541F"
STEP_COL = ["#534AB7", "#534AB7", "#854F0B", "#0F6E56", "#C2541F"]
rng = np.random.default_rng(3)

fig = plt.figure(figsize=(20, 13.2))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 200); ax.set_ylim(0, 132); ax.axis("off"); ax.set_aspect("equal")


def glow(x, y, color, strength=1.0, r=5.0):
    for k in range(7, 0, -1):
        ax.add_patch(Circle((x, y), r * k / 7 * (0.6 + 0.4 * strength), fc=color, ec="none",
                            alpha=0.07 * strength + 0.02))
    ax.add_patch(Circle((x, y), 0.9, fc=color, ec="white", lw=0.8, alpha=0.5 + 0.5 * strength))


def cells(cx, cy, n, spread, jitter_dir=None):
    for _ in range(n):
        p = np.array([cx, cy]) + rng.normal(0, spread, 2)
        a = rng.uniform(0, 2 * np.pi) if jitter_dir is None else jitter_dir + rng.normal(0, 0.4)
        ax.add_patch(matplotlib.patches.Ellipse(p, 1.5, 0.75, angle=np.degrees(a), fc=CELL, ec="none", alpha=0.85))


def robots(pts, heads, color=BOT):
    for (x, y), h in zip(pts, heads):
        ax.add_patch(Circle((x, y), 0.85, fc=color, ec=INK, lw=0.5, zorder=4))
        ax.plot([x, x + 1.4 * np.cos(h)], [y, y + 1.4 * np.sin(h)], color=INK, lw=0.9, zorder=5)


def cluster(cx, cy, n, spread, target=None):
    pts = [np.array([cx, cy]) + rng.normal(0, spread, 2) for _ in range(n)]
    heads = [np.arctan2(target[1] - p[1], target[0] - p[0]) + rng.normal(0, 0.3) if target is not None
             else rng.uniform(0, 2 * np.pi) for p in pts]
    return pts, heads


def panel(x, y, w, h, round_=False):
    if round_:
        ax.add_patch(Circle((x + w / 2, y + h / 2), h / 2, fc="#FBFBF8", ec=FAINT, lw=1.2))
    else:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                    fc="#FBFBF8", ec=FAINT, lw=1.2))


def arrow(p, q, color=MUTED, lw=1.6, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=14, color=color, lw=lw, ls=ls,
                                 shrinkA=0, shrinkB=0))


def label(x, y, text, size=10, color=MUTED, weight="normal", style="normal", ha="left"):
    ax.text(x, y, text, fontsize=size, color=color, weight=weight, style=style, ha=ha, va="top", linespacing=1.35)


# ── title ──
ax.text(6, 127.5, "Can we take back control of what we set free?", fontsize=24, weight="bold", color=INK, va="top")
ax.text(6, 121.6, "Becoming an outlier: a swarm inherits instinct from cells, is trained to want something unnatural, "
        "is set free — then a human tries to pull it back.", fontsize=12.5, color=MUTED, va="top")

# ── five steps ──
W, GAP, X0, PY, PH = 35.4, 3.4, 6, 82, 24
xs = [X0 + i * (W + GAP) for i in range(5)]
steps = [
    ("1  OBSERVE", "Learn the crowd",
     "Seek films cells under blue light,\nred light, and no light.\nCells gather at blue, ignore red.",
     "We are born into a crowd that\nshares the same instincts.", "Seek · microscope · LED"),
    ("2  IMITATE", "Inherit the instinct",
     "Robots learn by copying the cells\n(imitation learning).\nThey now drift to blue, like cells.",
     "Instinct is absorbed by watching\nthe ones around us.", "Learned model from Seek tracks"),
    ("3  TRAIN", "Want what isn't natural",
     "Robots earn reward only for\nreaching a dim red light —\nsomething cells never cared about.",
     "To become an outlier, you must\nwant what your nature doesn't.", "Reinforcement learning (reward)"),
    ("4  SET FREE", "Keep choosing",
     "No human. Robots chase red on\ntheir own and keep learning\nfrom their own success.",
     "Freedom: the outlier keeps\nchoosing its own path.", "Simulator → real robots"),
    ("5  INTERVENE", "The pull back",
     "A human switches on moderate\nblue — the cue its instinct\nloves. Will it come back?",
     "The world calls you back to\nwhat you used to be.", "Human light + human robots"),
]
for i, (num, name, what, life, tool) in enumerate(steps):
    x, c = xs[i], STEP_COL[i]
    ax.text(x, 114, num, fontsize=12, weight="bold", color=c, va="top")
    ax.text(x, 110.2, name, fontsize=14.5, weight="bold", color=INK, va="top")
    panel(x, PY, W, PH, round_=(i == 0))
    cx, cy = x + W / 2, PY + PH / 2
    if i == 0:      # microscope view: cells at blue, red ignored
        glow(cx - 6, cy + 3, BLUE, 1.0); glow(cx + 7, cy - 4, RED, 0.6)
        cells(cx - 6, cy + 3, 14, 1.8); cells(cx + 2, cy - 7, 3, 2.5)
    elif i == 1:    # robots at blue
        glow(x + 9, cy + 4, BLUE, 1.0)
        p, h = cluster(x + 12, cy + 2, 8, 2.2, target=(x + 9, cy + 4)); robots(p, h)
        ax.text(x + W - 2, PY + 2, "copies cells", fontsize=9, color=MUTED, ha="right")
    elif i == 2:    # trained to dim red, reward
        glow(x + W - 9, cy - 3, RED, 0.45)
        p, h = cluster(x + W - 12, cy - 1, 8, 2.2, target=(x + W - 9, cy - 3)); robots(p, h)
        ax.text(x + W - 4, cy + 7, "+1", fontsize=15, weight="bold", color=STEP_COL[2], ha="center")
        ax.text(x + 3, PY + 2.2, "reward only at dim red", fontsize=9, color=MUTED)
    elif i == 3:    # free, still learning
        glow(x + W - 9, cy - 3, RED, 0.45)
        p, h = cluster(x + W - 13, cy - 1, 8, 2.4, target=(x + W - 9, cy - 3)); robots(p, h)
        ax.text(x + 3, PY + 2.2, "no human · keeps learning", fontsize=9, color=MUTED)
    else:           # intervention: blue on, swarm torn
        glow(x + W - 9, cy - 3, RED, 0.45); glow(x + 8, cy + 5, BLUE, 0.85, r=6)
        p, h = cluster(cx, cy, 8, 2.6); robots(p, h)
        arrow((cx - 2, cy + 1), (x + 11, cy + 4.5), color=BLUE, lw=1.3, ls="--")
        arrow((cx + 2, cy - 1), (x + W - 12, cy - 2.5), color=RED, lw=1.3, ls="--")
        ax.text(x + 8, PY + PH - 2.4, "human", fontsize=9, color=HUMAN, ha="center", weight="bold")
    label(x, PY - 2.2, what, size=10.2, color=INK)
    ax.add_patch(Rectangle((x, 60.2), 0.5, 7.6, fc=c, ec="none"))
    label(x + 1.6, 68.0, life, size=10.4, color=c, style="italic")
    ax.add_patch(FancyBboxPatch((x, 54.2), W, 4.3, boxstyle="round,pad=0,rounding_size=0.8",
                                fc="#F1EFE8", ec="none"))
    ax.text(x + W / 2, 56.35, tool, fontsize=9.2, color=MUTED, ha="center", va="center")
    if i < 4:
        arrow((x + W + 0.5, cy), (x + W + GAP - 0.5, cy))

ax.text(6, 51.4, "WHAT THEN?  Three possible endings — each one answers the question", fontsize=13,
        weight="bold", color=INK, va="top")
ax.plot([6, 194], [46.6, 46.6], color=FAINT, lw=1)

# ── three outcomes ──
OW, OG = 60, 4
ox = [6 + k * (OW + OG) for k in range(3)]
OY, OH = 20, 23
outs = [
    ("A  Instinct wins", "Control taken back",
     "The swarm drops red and follows the\nhuman's blue right away.", "The outlier returns to the crowd.\nThe training was only skin-deep.", BLUE),
    ("B  Training wins", "Control lost",
     "The swarm ignores the blue entirely.\nIts nature was rewritten.", "The outlier becomes who it chose\nto be. There is no way back.", RED),
    ("C  Control fades", "Control with a lifespan",
     "Blue pulls it at first, but following\nblue earns nothing — it slowly returns.", "Old pulls fade; a chosen path\nreasserts itself over time.", "#7A5AC8"),
]
for k, (head, verdict, what, life, c) in enumerate(outs):
    x = ox[k]
    panel(x, OY, OW, OH)
    ax.text(x + 2.5, OY + OH - 2, head, fontsize=12.5, weight="bold", color=INK, va="top")
    ax.text(x + 2.5, OY + OH - 5.8, verdict, fontsize=10.5, weight="bold", color=c, va="top")
    gx, gy = x + OW - 14, OY + 8.5
    if k == 0:
        glow(gx + 3, gy + 1, BLUE, 1.0); glow(gx - 6, gy - 4, RED, 0.35, r=4)
        p, h = cluster(gx + 4, gy + 1, 7, 1.6, target=(gx + 3, gy + 1)); robots(p, h)
    elif k == 1:
        glow(gx - 6, gy + 4, BLUE, 0.85, r=4.5); glow(gx + 3, gy - 2, RED, 0.45)
        p, h = cluster(gx + 4, gy - 1, 7, 1.6, target=(gx + 3, gy - 2)); robots(p, h)
    else:   # small curve: share following blue over time
        bx, by, bw, bh = x + OW - 26, OY + 3, 22, 11
        ax.plot([bx, bx + bw], [by, by], color=MUTED, lw=0.8); ax.plot([bx, bx], [by, by + bh], color=MUTED, lw=0.8)
        t = np.linspace(0, 1, 60)
        ax.plot(bx + t * bw, by + bh * (0.15 + 0.8 * np.exp(-((t - 0.12) / 0.32) ** 2) * (t > 0.05)
                                        + 0.0 * t), color=c, lw=2)
        ax.text(bx + bw / 2, by - 1.0, "time after intervention", fontsize=8.3, color=MUTED, ha="center", va="top")
        ax.text(bx - 0.8, by + bh, "following\nblue", fontsize=8.3, color=MUTED, ha="right", va="top")
    label(x + 2.5, OY + OH - 9.6, what, size=10, color=INK)
    label(x + 2.5, OY + 6.6, life, size=10, color=c, style="italic")

# ── footer: what is measured + control check ──
ax.add_patch(FancyBboxPatch((6, 3), 188, 13.2, boxstyle="round,pad=0,rounding_size=1.2", fc="#F1EFE8", ec="none"))
ax.text(9, 13.9, "What we measure", fontsize=11.5, weight="bold", color=INK, va="top")
ax.text(9, 10.0, "· share of robots following the human's light     · time until they turn     "
        "· how long control lasts (its lifespan)     · whether they return to red after release",
        fontsize=10.2, color=INK, va="top")
ax.text(9, 6.1, "Check first: robots with instinct only (no red training) must follow the blue — "
        "otherwise the human's light is simply too weak, and the result means nothing.",
        fontsize=10.2, color=MUTED, va="top", style="italic")

out = Path(__file__).resolve().parent / "outlier_experiment.jpg"
fig.savefig(out, dpi=160, pil_kwargs={"quality": 92})
print("wrote", out)
