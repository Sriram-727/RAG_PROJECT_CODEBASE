"""Builds the paper figures: code-faithful diagrams, the DBV ablation plot (from results.json) and
screenshot composites (cropped from the figures extracted from the project report)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")
OUT = os.path.join(HERE, "..", "paper_figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 8})

INK, GREY, LIGHT = "#1f2933", "#6b7280", "#eef1f5"
BLUE, ORANGE, GREEN, RED, PURPLE = "#dbe7f5", "#fbe3cf", "#dff0dc", "#f6d7d7", "#e6def2"


def box(ax, x, y, w, h, text, fc=LIGHT, ec=INK, fs=7.5, bold=False, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.0, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=INK, linespacing=1.25)


def arrow(ax, p, q, text=None, color=INK, rad=0.0, tx=None, ty=None, ha="center"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, lw=1.0, color=color,
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text(tx if tx is not None else (p[0] + q[0]) / 2, ty if ty is not None else (p[1] + q[1]) / 2 + 0.08,
                text, fontsize=6.5, ha=ha, color=GREY, style="italic")


# ----------------------------------------------------------------- Figure 1: architecture
fig, ax = plt.subplots(figsize=(7.4, 6.6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 9.4)
ax.axis("off")
ax.add_patch(FancyBboxPatch((0.1, 0.1), 5.2, 9.2, boxstyle="round,pad=0.02,rounding_size=0.1",
                            fc="#f7f9fc", ec="#2b4c7e", lw=1.4, ls="--"))
ax.add_patch(FancyBboxPatch((6.2, 0.1), 3.7, 9.2, boxstyle="round,pad=0.02,rounding_size=0.1",
                            fc="#fff8f2", ec="#b4560f", lw=1.4, ls="--"))
ax.text(2.7, 9.05, "Trusted client service (FastAPI)", ha="center", fontsize=9, fontweight="bold", color="#2b4c7e")
ax.text(8.05, 9.05, "Untrusted cloud (Qdrant)", ha="center", fontsize=9, fontweight="bold", color="#b4560f")
ax.text(2.7, 8.65, "PHASE A – ingestion (per document)", ha="center", fontsize=7.5, color=GREY, fontweight="bold")
box(ax, 0.5, 7.7, 4.4, 0.7, "A1  Chunking\n500-character windows, 50 overlap, section-aware", BLUE)
box(ax, 0.5, 6.6, 4.4, 0.8, "A2  Embedding + cell assignment\nall-MiniLM-L6-v2 (384-d, L2-normalised)\nnearest IVF centroid → cell id", BLUE)
box(ax, 0.5, 5.5, 4.4, 0.8, "A3  Seal\nSHA-256 chunk hash;  AES-256-GCM\n(fresh 96-bit nonce) over the chunk text", BLUE)
box(ax, 0.5, 4.4, 4.4, 0.8, "A4  Two-tier MMR commitment\nchunk leaves → user MMR → organisation MMR\nproof path + root attached to each chunk", PURPLE)
for y0, y1 in ((7.7, 7.4), (6.6, 6.3), (5.5, 5.2)):
    arrow(ax, (2.7, y0), (2.7, y1))
box(ax, 6.5, 5.35, 3.1, 3.0,
    "Per-tenant collection\n(one per organisation)\n\nstored per point:\nvector (384-d, cosine)\nAES-GCM ciphertext + nonce\nchunk hash, cell id, ACL\nleaf hash + MMR proof path\nHNSW index (m=16, ef=100)", ORANGE, fs=7)
arrow(ax, (4.9, 5.9), (6.5, 6.6), "upsert", tx=5.7, ty=6.4)
arrow(ax, (4.9, 4.8), (6.5, 5.6), None, color="#5b3f8c", rad=0.0)
ax.text(5.9, 4.95, "proofs", fontsize=6.5, color="#5b3f8c", style="italic", ha="center")

ax.text(2.7, 3.85, "PHASE B – query, retrieval and verification", ha="center", fontsize=7.5, color=GREY, fontweight="bold")
box(ax, 0.5, 3.05, 4.4, 0.6, "B1  PII redaction → query embedding\nrequired IVF cells from local centroids", BLUE)
box(ax, 0.5, 1.75, 4.4, 1.0, "B2  Five checks on the response\n1 RBAC / tenant   2 MMR proof   3 DBV cells\n4 AES-GCM tag   5 CRAG grounding", GREEN, bold=False)
box(ax, 0.5, 0.55, 4.4, 0.8, "B3  Hybrid re-ranking + LLM synthesis\nanswer, grounding score, Self-RAG token", GREEN)
arrow(ax, (2.7, 3.05), (2.7, 2.75))
arrow(ax, (2.7, 1.75), (2.7, 1.35))
box(ax, 6.5, 3.05, 3.1, 0.6, "search (vector + filters)", ORANGE, fs=7)
box(ax, 6.5, 1.75, 3.1, 1.0, "returns points: ciphertext,\nproof path, cell ids, scores", ORANGE, fs=7)
arrow(ax, (4.9, 3.35), (6.5, 3.35), "query vector", tx=5.7, ty=3.45)
arrow(ax, (8.05, 3.05), (8.05, 2.75))
arrow(ax, (6.5, 2.25), (4.9, 2.25), "response", tx=5.7, ty=2.35)
fig.savefig(os.path.join(OUT, "fig1_architecture.png"), dpi=220, bbox_inches="tight")
plt.close(fig)

# ----------------------------------------------------------------- Figure 2: two-tier MMR
fig, ax = plt.subplots(figsize=(7.4, 4.6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6.2)
ax.axis("off")
box(ax, 3.1, 5.35, 3.8, 0.7, "Organisation root  R_org\nSHA-256(0x04 | org | U | N | bag of org peaks)", GREEN, bold=True)
box(ax, 2.3, 4.15, 5.4, 0.7, "Organisation MMR (Tier 2): leaves are user roots,\nusers in alphabetical order, peaks bagged right-to-left", PURPLE)
arrow(ax, (5.0, 4.15), (5.0, 4.05 + 0.0))
arrow(ax, (5.0, 4.85), (5.0, 5.35))
for i, (x, name) in enumerate(((0.3, "user A"), (3.6, "user B"), (6.9, "user C"))):
    box(ax, x, 2.8, 2.8, 0.7, f"R_user ({name})\nSHA-256(0x03 | user | n | bag)", BLUE, fs=7)
    arrow(ax, (x + 1.4, 3.5), (5.0 + (x + 1.4 - 5.0) * 0.35, 4.15))
    box(ax, x, 1.55, 2.8, 0.75, f"User MMR (Tier 1)\nperfect subtrees ('mountains')\nof chunk leaves", BLUE, fs=7)
    arrow(ax, (x + 1.4, 2.3), (x + 1.4, 2.8))
    for j in range(4):
        box(ax, x + 0.1 + j * 0.68, 0.45, 0.6, 0.55, "leaf", "#ffffff", fs=6.5)
        ax.plot([x + 0.4 + j * 0.68, x + 1.4], [1.0, 1.55], color=GREY, lw=0.6)
ax.text(5.0, 0.12, "leaf = SHA-256(0x00 | org | user | doc | chunk | index | SHA-256(plaintext)),   node = SHA-256(0x01 | left | right)",
        ha="center", fontsize=6.8, color=INK)
fig.savefig(os.path.join(OUT, "fig2_two_tier_mmr.png"), dpi=220, bbox_inches="tight")
plt.close(fig)

# ----------------------------------------------------------------- Figure 3: five checks
fig, ax = plt.subplots(figsize=(7.4, 5.2))
ax.set_xlim(-1.6, 10)
ax.set_ylim(0, 8.6)
ax.axis("off")
box(ax, 2.2, 7.8, 5.6, 0.6, "Cloud response: ciphertext, proof path, cell ids", ORANGE, bold=True, fs=7.5)
gates = [
    ("1  RBAC and tenant isolation", "user registered, approved, and in the target organisation", "cross-tenant access"),
    ("2  Two-tier MMR inclusion proof", "leaf and both tiers recomputed; peaks bagged; root compared", "forged hash / proof"),
    ("3  Dynamic Boundary Validation", "required cells ⊆ reported cells", "lazy cell omission"),
    ("4  AES-256-GCM authenticated decryption", "128-bit tag must verify before plaintext is released", "ciphertext modification"),
    ("5  CRAG / Self-RAG grounding", "retrieval grade, grounding score ≥ 0.35, support ratio", "ungrounded answer"),
]
y = 6.7
for k, (t, d, thr) in enumerate(gates):
    box(ax, 2.2, y, 5.6, 0.85, f"Gate {t}\n{d}", [PURPLE, BLUE, "#d5eeee", ORANGE, "#eeeeee"][k], fs=7)
    ax.text(-1.5, y + 0.42, thr, fontsize=6.8, va="center", color=GREY, style="italic")
    ax.plot([0.55, 2.2], [y + 0.42, y + 0.42], color=GREY, lw=0.7, ls=":")
    arrow(ax, (7.8, y + 0.42), (8.6, y + 0.42), None, color="#a11")
    ax.text(8.0, y + 0.5, "fail", fontsize=6.5, color="#a11", ha="center")
    if k < 4:
        arrow(ax, (5.0, y), (5.0, y - 0.35 + 0.0), "pass", tx=5.35, ty=y - 0.27, ha="left")
    y -= 1.2
box(ax, 8.65, 1.1, 1.3, 5.9, "BLOCKED\n\nno context\nreaches the\nLLM", RED, fs=7, bold=True)
box(ax, 2.2, 0.15, 5.6, 0.55, "VERIFIED → grounded LLM synthesis", GREEN, bold=True)
arrow(ax, (5.0, 2.0), (5.0, 0.72), "pass", tx=5.35, ty=1.2, ha="left")
fig.savefig(os.path.join(OUT, "fig3_five_checks.png"), dpi=220, bbox_inches="tight")
plt.close(fig)

# ----------------------------------------------------------------- Figure 4: ablation
res = json.load(open(os.path.join(HERE, "results.json")))
abl = res["dbv_ablation"]
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
styles = {6: ("o", "-", "#1f4e79"), 16: ("s", "--", "#b4560f")}
for n in (6, 16):
    rows = [a for a in abl if a["n_lists"] == n]
    xs = [r["margin_factor"] for r in rows]
    m, ls, c = styles[n]
    axes[0].plot(range(len(xs)), [r["recall_at_5"] for r in rows], marker=m, ls=ls, color=c, label=f"{n} cells")
    axes[1].plot(range(len(xs)), [r["mean_fraction_scanned"] for r in rows], marker=m, ls=ls, color=c, label=f"{n} cells")
for a_, yl in zip(axes, ("Recall@5 inside required cells", "Fraction of corpus in required cells")):
    a_.set_xticks(range(len(xs)))
    a_.set_xticklabels([str(x) for x in xs])
    a_.set_xlabel("margin factor λ")
    a_.set_ylabel(yl)
    a_.set_ylim(0, 1.05)
    a_.grid(alpha=0.3)
    a_.axvline(xs.index(1.5), color=GREY, lw=0.8, ls=":")
axes[0].legend(frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig4_ablation.png"), dpi=220)
plt.close(fig)


# ----------------------------------------------------------------- screenshots
def crop(name, box_):
    return Image.open(os.path.join(FIG, name)).crop(box_)


def composite(panels, fname, ncols, figsize, titles):
    fig, axs = plt.subplots(len(panels) // ncols, ncols, figsize=figsize,
                            gridspec_kw={"width_ratios": [2, 1]} if (ncols == 2) else None)
    axs = axs.flatten() if hasattr(axs, "flatten") else [axs]
    for a_, im, t in zip(axs, panels, titles):
        a_.imshow(im)
        a_.axis("off")
        a_.set_title(t, fontsize=8, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=220)
    plt.close(fig)


composite([crop("fig6_1.png", (0, 145, 1917, 1078))], "fig5_collections.png", 1, (7.4, 3.9), [""])
composite([crop("fig6_2.png", (300, 140, 1912, 937)), crop("fig6_5.png", (1110, 130, 1912, 920))],
          "fig6_payload.png", 2, (7.4, 3.3), ["(a) stored payload: ciphertext, nonce, hash, ACL", "(b) leaf hash and two-tier proof path"])
composite([crop("fig6_7.png", (0, 145, 1917, 740)), crop("fig6_8.png", (0, 145, 1917, 740))],
          "fig7_ui.png", 1, (7.4, 5.0), ["(a) authorised query: 22 chunks verified, 137.9 ms", "(b) cross-tenant query blocked, 0 chunks released, 137.64 ms"])
print(os.listdir(OUT))
