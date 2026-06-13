from __future__ import annotations

from html import escape
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "assets" / "figures"

BLUE = "#d7e8ff"
BLUE_EDGE = "#2f65b0"
GREEN = "#dff3df"
GREEN_EDGE = "#438a43"
ORANGE = "#ffe6c9"
ORANGE_EDGE = "#bf6b21"
PURPLE = "#eadfff"
PURPLE_EDGE = "#7453b6"
RED = "#ffd7d7"
RED_EDGE = "#b83a3a"
GRAY = "#f1f3f5"
GRAY_EDGE = "#70757a"
DARK = "#1f2933"


class SVG:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.items: list[str] = []

    def rect(self, x: float, y: float, w: float, h: float, text: str, *, fill=BLUE, stroke=BLUE_EDGE, size=15) -> None:
        self.items.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="2" rx="4" />'
        )
        self.text(x + w / 2, y + h / 2, text, size=size, anchor="middle", valign="middle")

    def text(self, x: float, y: float, text: str, *, size=15, anchor="start", valign="start", color=DARK) -> None:
        lines = text.split("\n")
        if valign == "middle":
            y = y - (len(lines) - 1) * size * 0.35
        for i, line in enumerate(lines):
            dy = i * size * 1.15
            self.items.append(
                f'<text x="{x}" y="{y + dy}" font-family="Arial, sans-serif" '
                f'font-size="{size}" fill="{color}" text-anchor="{anchor}" '
                f'dominant-baseline="middle">{escape(line)}</text>'
            )

    def line(self, x1: float, y1: float, x2: float, y2: float, *, color=DARK, width=2) -> None:
        self.items.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{color}" stroke-width="{width}" marker-end="url(#arrow)" />'
        )

    def path(self, d: str, *, color=DARK, width=2) -> None:
        self.items.append(
            f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" marker-end="url(#arrow)" />'
        )

    def matrix(self, x: float, y: float, rows: int, cols: int, *, cell=32, label="", highlights=None) -> None:
        highlights = highlights or {}
        for r in range(rows):
            for c in range(cols):
                fill = highlights.get((r, c), GRAY)
                self.items.append(
                    f'<rect x="{x + c * cell}" y="{y + r * cell}" width="{cell}" height="{cell}" '
                    f'fill="{fill}" stroke="{GRAY_EDGE}" stroke-width="1" />'
                )
        if label:
            self.text(x + cols * cell / 2, y + rows * cell + 24, label, size=14, anchor="middle")

    def save(self, name: str) -> None:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        body = "\n".join(self.items)
        svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" viewBox="0 0 {self.width} {self.height}">
<defs>
  <marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
    <path d="M0,0 L0,6 L9,3 z" fill="{DARK}" />
  </marker>
</defs>
<rect x="0" y="0" width="{self.width}" height="{self.height}" fill="white" />
{body}
</svg>
'''
        (FIG_DIR / name).write_text(svg, encoding="utf-8")


def pipeline_overview() -> None:
    s = SVG(1180, 380)
    stages = [
        ("raw text", ORANGE, ORANGE_EDGE),
        ("tokenizer", BLUE, BLUE_EDGE),
        ("token ids\n(B, N)", BLUE, BLUE_EDGE),
        ("embeddings\n+ positions", GREEN, GREEN_EDGE),
        ("transformer\nblocks x L", PURPLE, PURPLE_EDGE),
        ("logits\n(B, N, V)", GREEN, GREEN_EDGE),
        ("loss or\nsample", RED, RED_EDGE),
    ]
    x0, y, w, h, gap = 35, 125, 135, 70, 32
    for i, (label, fill, stroke) in enumerate(stages):
        x = x0 + i * (w + gap)
        s.rect(x, y, w, h, label, fill=fill, stroke=stroke, size=14)
        if i < len(stages) - 1:
            s.line(x + w, y + h / 2, x + w + gap, y + h / 2)
    s.text(40, 255, "training: logits + shifted targets -> cross entropy -> backward -> optimizer step", size=17)
    s.text(40, 300, "generation: last-position logits -> temperature/top-k -> sample -> append token -> repeat", size=17)
    s.save("pipeline_overview.svg")


def matrix_multiplication() -> None:
    s = SVG(840, 390)
    s.matrix(70, 90, 3, 4, highlights={(1, 0): ORANGE, (1, 1): ORANGE, (1, 2): ORANGE, (1, 3): ORANGE}, label="X: (B*N, D_in)")
    s.text(260, 145, "@", size=32, anchor="middle")
    s.matrix(310, 72, 4, 2, highlights={(0, 1): BLUE, (1, 1): BLUE, (2, 1): BLUE, (3, 1): BLUE}, label="W: (D_in, D_out)")
    s.text(445, 145, "=", size=32, anchor="middle")
    s.matrix(500, 90, 3, 2, highlights={(1, 1): GREEN}, label="Y: (B*N, D_out)")
    s.path("M 130 70 C 210 25, 470 25, 548 118", color=ORANGE_EDGE)
    s.path("M 360 55 C 430 10, 565 35, 580 118", color=BLUE_EDGE)
    s.text(70, 330, "Highlighted output cell = dot(row of X, column of W)", size=17)
    s.save("matrix_multiplication.svg")


def linear_backward() -> None:
    s = SVG(950, 390)
    s.rect(40, 70, 120, 55, "X", fill=ORANGE, stroke=ORANGE_EDGE)
    s.rect(40, 180, 120, 55, "W, b", fill=BLUE, stroke=BLUE_EDGE)
    s.rect(230, 115, 145, 80, "Linear\nY = XW + b", fill=GREEN, stroke=GREEN_EDGE, size=14)
    s.rect(445, 128, 110, 55, "Y", fill=GRAY, stroke=GRAY_EDGE)
    s.rect(635, 128, 110, 55, "dY", fill=RED, stroke=RED_EDGE)
    s.rect(790, 55, 130, 45, "dW = X^T dY", fill=PURPLE, stroke=PURPLE_EDGE, size=13)
    s.rect(790, 138, 130, 45, "db = sum dY", fill=PURPLE, stroke=PURPLE_EDGE, size=13)
    s.rect(790, 221, 130, 45, "dX = dY W^T", fill=PURPLE, stroke=PURPLE_EDGE, size=13)
    s.line(160, 98, 230, 140)
    s.line(160, 208, 230, 170)
    s.line(375, 155, 445, 155)
    s.line(555, 155, 635, 155)
    s.line(745, 150, 790, 78)
    s.line(745, 155, 790, 160)
    s.line(745, 160, 790, 244)
    s.text(45, 330, "Backward receives dY, fills parameter gradients, and returns dX for the previous layer.", size=16)
    s.save("linear_backward.svg")


def embedding_shifted_targets() -> None:
    s = SVG(900, 360)
    tokens = ["Once", "upon", "a", "time", "."]
    x0 = 90
    for i, token in enumerate(tokens[:-1]):
        x = x0 + i * 145
        s.rect(x, 90, 105, 48, token, fill=BLUE, stroke=BLUE_EDGE, size=14)
        s.rect(x, 210, 105, 48, tokens[i + 1], fill=GREEN, stroke=GREEN_EDGE, size=14)
        s.line(x + 52, 145, x + 52, 202)
    s.text(80, 45, "input x: tokens[t : t + N]", size=17)
    s.text(80, 300, "target y: tokens[t + 1 : t + N + 1]", size=17)
    s.text(80, 335, "Each position predicts the next token, so targets are shifted one step to the right.", size=15)
    s.save("embedding_shifted_targets.svg")


def attention_qk_softmax_v() -> None:
    s = SVG(1060, 430)
    s.text(40, 45, "Attention is a soft lookup table: queries score keys, softmax creates weights, weights mix values.", size=17)
    s.rect(55, 110, 80, 55, "Q", fill=BLUE, stroke=BLUE_EDGE)
    s.rect(55, 200, 80, 55, "K", fill=BLUE, stroke=BLUE_EDGE)
    s.rect(55, 290, 80, 55, "V", fill=BLUE, stroke=BLUE_EDGE)
    s.rect(205, 155, 150, 70, "scores\nQK^T / sqrt(d)", fill=ORANGE, stroke=ORANGE_EDGE, size=13)
    s.rect(410, 155, 120, 70, "causal\nmask", fill=RED, stroke=RED_EDGE, size=13)
    s.rect(585, 155, 140, 70, "row\nsoftmax", fill=GREEN, stroke=GREEN_EDGE, size=13)
    s.rect(780, 155, 90, 70, "A", fill=GREEN, stroke=GREEN_EDGE)
    s.rect(920, 245, 105, 70, "Y = A V", fill=PURPLE, stroke=PURPLE_EDGE)
    s.line(135, 138, 205, 175)
    s.line(135, 228, 205, 205)
    s.line(355, 190, 410, 190)
    s.line(530, 190, 585, 190)
    s.line(725, 190, 780, 190)
    s.line(870, 190, 920, 270)
    s.path("M 135 318 C 350 365, 700 370, 920 292")
    s.text(360, 385, "Mask before softmax so future tokens receive probability zero.", size=16)
    s.save("attention_qk_softmax_v.svg")


def multihead_attention() -> None:
    s = SVG(980, 420)
    s.text(45, 45, "Multi-head attention runs several smaller attentions in parallel, then recombines them.", size=17)
    s.rect(55, 175, 110, 70, "X\n(B,N,D)", fill=BLUE, stroke=BLUE_EDGE, size=13)
    head_y = [90, 175, 260]
    for i, y in enumerate(head_y):
        label = "head H" if i == 2 else f"head {i + 1}"
        s.rect(250, y, 110, 50, label, fill=GREEN, stroke=GREEN_EDGE, size=13)
        s.rect(435, y, 125, 50, "attention", fill=ORANGE, stroke=ORANGE_EDGE, size=13)
        s.line(165, 210, 250, y + 25)
        s.line(360, y + 25, 435, y + 25)
        s.line(560, y + 25, 675, 210)
    s.rect(675, 175, 120, 70, "concat\n(B,N,D)", fill=PURPLE, stroke=PURPLE_EDGE, size=13)
    s.rect(855, 175, 90, 70, "W_o", fill=BLUE, stroke=BLUE_EDGE)
    s.line(795, 210, 855, 210)
    s.text(245, 365, "Each head has dimension D/H, so concatenating H heads returns to D.", size=16)
    s.save("multihead_attention.svg")


def transformer_block() -> None:
    s = SVG(1050, 420)
    s.text(40, 45, "Pre-norm transformer block: normalize before each sublayer, then add the residual stream.", size=17)
    s.rect(45, 185, 80, 55, "x", fill=BLUE, stroke=BLUE_EDGE)
    s.rect(190, 105, 100, 50, "norm1", fill=GRAY, stroke=GRAY_EDGE, size=13)
    s.rect(340, 105, 125, 50, "attention", fill=ORANGE, stroke=ORANGE_EDGE, size=13)
    s.rect(540, 105, 70, 50, "+", fill=GREEN, stroke=GREEN_EDGE)
    s.rect(680, 185, 100, 50, "norm2", fill=GRAY, stroke=GRAY_EDGE, size=13)
    s.rect(825, 185, 135, 50, "FFN or MoE", fill=PURPLE, stroke=PURPLE_EDGE, size=13)
    s.rect(915, 290, 70, 50, "+", fill=GREEN, stroke=GREEN_EDGE)
    s.line(125, 212, 190, 130)
    s.line(290, 130, 340, 130)
    s.line(465, 130, 540, 130)
    s.path("M 125 212 C 270 260, 425 260, 540 150")
    s.line(610, 130, 680, 210)
    s.line(780, 210, 825, 210)
    s.line(960, 210, 950, 290)
    s.path("M 610 150 C 720 330, 825 345, 915 315")
    s.line(985, 315, 1030, 315)
    s.text(640, 380, "MoE swaps only this FFN box; the residual scaffold stays the same.", size=16)
    s.save("transformer_block.svg")


def moe_routing() -> None:
    s = SVG(980, 420)
    s.text(45, 45, "MoE routing: every token receives expert probabilities, then only top-k experts run.", size=17)
    token_y = [100, 165, 230, 295]
    for i, y in enumerate(token_y):
        s.rect(70, y, 100, 40, f"token {i}", fill=BLUE, stroke=BLUE_EDGE, size=12)
        s.line(170, y + 20, 300, 210)
    s.rect(300, 170, 125, 80, "router\nsoftmax", fill=ORANGE, stroke=ORANGE_EDGE, size=13)
    expert_y = [90, 155, 220, 285]
    for i, y in enumerate(expert_y):
        s.rect(585, y, 110, 42, f"expert {i}", fill=GREEN, stroke=GREEN_EDGE, size=12)
        s.line(425, 210, 585, y + 21)
    s.text(735, 210, "top-k chooses a small\nset of experts per token", size=16, valign="middle")
    s.save("moe_routing.svg")


def moe_dispatch_combine() -> None:
    s = SVG(1060, 420)
    s.text(45, 45, "Dispatch/combine preserves token order: flatten for routing, then reshape back for the residual path.", size=17)
    s.rect(55, 160, 105, 60, "x\n(B,N,D)", fill=BLUE, stroke=BLUE_EDGE, size=13)
    s.rect(220, 160, 120, 60, "flatten\n(T,D)", fill=BLUE, stroke=BLUE_EDGE, size=13)
    s.rect(400, 160, 125, 60, "dispatch\nby expert", fill=ORANGE, stroke=ORANGE_EDGE, size=13)
    expert_y = [85, 160, 235]
    for i, y in enumerate(expert_y):
        label = "expert E" if i == 2 else f"expert {i}"
        s.rect(600, y, 110, 42, label, fill=GREEN, stroke=GREEN_EDGE, size=12)
        s.line(525, 190, 600, y + 21)
        s.line(710, y + 21, 790, 190)
    s.rect(790, 160, 125, 60, "weighted\ncombine", fill=PURPLE, stroke=PURPLE_EDGE, size=13)
    s.rect(965, 160, 80, 60, "y\n(B,N,D)", fill=BLUE, stroke=BLUE_EDGE, size=13)
    s.line(160, 190, 220, 190)
    s.line(340, 190, 400, 190)
    s.line(915, 190, 965, 190)
    s.text(230, 340, "index_add_ sums contributions when top-k sends one token to multiple experts.", size=16)
    s.save("moe_dispatch_combine.svg")


def main() -> None:
    pipeline_overview()
    matrix_multiplication()
    linear_backward()
    embedding_shifted_targets()
    attention_qk_softmax_v()
    multihead_attention()
    transformer_block()
    moe_routing()
    moe_dispatch_combine()
    print(f"wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
