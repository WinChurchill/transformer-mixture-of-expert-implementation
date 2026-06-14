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

    def plain_line(self, x1: float, y1: float, x2: float, y2: float, *, color=DARK, width=2) -> None:
        self.items.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{color}" stroke-width="{width}" />'
        )

    def path(self, d: str, *, color=DARK, width=2) -> None:
        self.items.append(
            f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" marker-end="url(#arrow)" />'
        )

    def circle(self, x: float, y: float, r: float, text: str, *, fill=GRAY, stroke=GRAY_EDGE, size=15) -> None:
        self.items.append(
            f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="2" />'
        )
        self.text(x, y, text, size=size, anchor="middle", valign="middle")

    def labeled_arrow(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        label: str = "",
        *,
        color=DARK,
        size=13,
    ) -> None:
        self.line(x1, y1, x2, y2, color=color)
        if label:
            self.text((x1 + x2) / 2, (y1 + y2) / 2 - 14, label, size=size, anchor="middle")

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


def decoder_shape_ladder() -> None:
    s = SVG(1120, 430)
    s.text(45, 45, "Decoder-only shape ladder: only the last dimension changes until the vocabulary projection.", size=17)
    stages = [
        ("token ids\nidx\n(B,T)", ORANGE, ORANGE_EDGE),
        ("embedding\nlookup\n(B,T,C)", BLUE, BLUE_EDGE),
        ("position\ninformation\n(B,T,C)", BLUE, BLUE_EDGE),
        ("blocks x L\n(B,T,C)", PURPLE, PURPLE_EDGE),
        ("lm_head\n(B,T,V)", GREEN, GREEN_EDGE),
        ("cross entropy\n(B*T,V) + (B*T)", RED, RED_EDGE),
    ]
    x0, y, w, h, gap = 45, 150, 130, 80, 42
    for i, (label, fill, stroke) in enumerate(stages):
        x = x0 + i * (w + gap)
        s.rect(x, y, w, h, label, fill=fill, stroke=stroke, size=13)
        if i < len(stages) - 1:
            s.labeled_arrow(x + w, y + h / 2, x + w + gap, y + h / 2)
    s.text(65, 300, "Training uses logits at every position. Generation reads only logits[:, -1, :].", size=16)
    s.text(65, 335, "Targets stay integer token IDs; they do not get a vocabulary dimension.", size=16)
    s.save("decoder_shape_ladder.svg")


def training_vs_generation_paths() -> None:
    s = SVG(1120, 460)
    s.text(45, 45, "The same model forward supports two workflows: parallel training and autoregressive generation.", size=17)
    train = [
        ("text shard", ORANGE, ORANGE_EDGE),
        ("sample\nx,y", BLUE, BLUE_EDGE),
        ("model\nforward", PURPLE, PURPLE_EDGE),
        ("logits + y\nCE loss", RED, RED_EDGE),
        ("backward\nAdamW", GREEN, GREEN_EDGE),
        ("checkpoint", GRAY, GRAY_EDGE),
    ]
    gen = [
        ("prompt", ORANGE, ORANGE_EDGE),
        ("tokenize", BLUE, BLUE_EDGE),
        ("model\nforward", PURPLE, PURPLE_EDGE),
        ("last logits", GREEN, GREEN_EDGE),
        ("filter +\nsample", RED, RED_EDGE),
        ("append\nrepeat", GRAY, GRAY_EDGE),
    ]
    for row, stages in enumerate([train, gen]):
        y = 120 + row * 160
        s.text(45, y + 35, "training" if row == 0 else "generation", size=16, anchor="start")
        x0, w, h, gap = 160, 120, 60, 38
        for i, (label, fill, stroke) in enumerate(stages):
            x = x0 + i * (w + gap)
            s.rect(x, y, w, h, label, fill=fill, stroke=stroke, size=13)
            if i < len(stages) - 1:
                s.line(x + w, y + h / 2, x + w + gap, y + h / 2)
    s.text(160, 405, "Training consumes shifted targets; generation feeds each sampled token back into the next step.", size=16)
    s.save("training_vs_generation_paths.svg")


def causal_mask_matrix() -> None:
    s = SVG(700, 430)
    s.text(45, 40, "Causal mask: each target position can read only itself and earlier source positions.", size=17)
    rows = cols = 6
    cell = 42
    x0, y0 = 190, 90
    for r in range(rows):
        s.text(x0 - 28, y0 + r * cell + cell / 2, f"t{r}", size=13, anchor="middle", valign="middle")
        s.text(x0 + r * cell + cell / 2, y0 - 24, f"s{r}", size=13, anchor="middle", valign="middle")
        for c in range(cols):
            fill = GREEN if c <= r else RED
            stroke = GREEN_EDGE if c <= r else RED_EDGE
            label = "ok" if c <= r else "x"
            s.items.append(
                f'<rect x="{x0 + c * cell}" y="{y0 + r * cell}" width="{cell}" height="{cell}" '
                f'fill="{fill}" stroke="{stroke}" stroke-width="1" />'
            )
            s.text(x0 + c * cell + cell / 2, y0 + r * cell + cell / 2, label, size=12, anchor="middle", valign="middle")
    s.text(90, 380, "Green cells survive softmax. Red future cells are set to -inf before softmax.", size=16)
    s.save("causal_mask_matrix.svg")


def attention_memory_scaling() -> None:
    s = SVG(980, 430)
    s.text(45, 45, "Attention memory changes shape between full prefill and one-token cached decode.", size=17)
    s.matrix(100, 120, 5, 5, cell=38, label="prefill scores: (T,T)")
    s.text(120, 330, "B * H * T * T", size=18, anchor="start")
    s.rect(440, 150, 110, 70, "vs", fill=GRAY, stroke=GRAY_EDGE, size=24)
    highlights = {(0, c): GREEN for c in range(7)}
    s.matrix(650, 160, 1, 7, cell=38, highlights=highlights, label="decode scores: (1,T_cache)")
    s.text(670, 330, "B * H * 1 * T_cache", size=18, anchor="start")
    s.text(70, 385, "KV cache removes repeated K/V computation; the newest query still reads the cached sequence.", size=16)
    s.save("attention_memory_scaling.svg")


def rope_qk_rotation() -> None:
    s = SVG(1080, 430)
    s.text(45, 45, "RoPE injects position by rotating Q and K after projection and head split; V is unchanged.", size=17)
    s.rect(60, 185, 100, 60, "x\n(B,T,C)", fill=BLUE, stroke=BLUE_EDGE, size=13)
    ys = [90, 185, 280]
    labels = ["Q", "K", "V"]
    for y, label in zip(ys, labels):
        s.rect(250, y, 95, 48, f"{label} proj", fill=ORANGE, stroke=ORANGE_EDGE, size=13)
        s.rect(405, y, 120, 48, f"{label}\n(B,H,T,Dh)", fill=BLUE, stroke=BLUE_EDGE, size=12)
        s.line(160, 215, 250, y + 24)
        s.line(345, y + 24, 405, y + 24)
    s.rect(610, 90, 130, 48, "rotate by\nposition", fill=PURPLE, stroke=PURPLE_EDGE, size=12)
    s.rect(610, 185, 130, 48, "rotate by\nposition", fill=PURPLE, stroke=PURPLE_EDGE, size=12)
    s.rect(610, 280, 130, 48, "no rotation", fill=GRAY, stroke=GRAY_EDGE, size=12)
    s.line(525, 114, 610, 114)
    s.line(525, 209, 610, 209)
    s.line(525, 304, 610, 304)
    s.rect(830, 140, 140, 70, "scores\nQ_rot K_rot^T", fill=GREEN, stroke=GREEN_EDGE, size=13)
    s.rect(830, 260, 140, 55, "values V", fill=BLUE, stroke=BLUE_EDGE, size=13)
    s.line(740, 114, 830, 165)
    s.line(740, 209, 830, 185)
    s.line(740, 304, 830, 288)
    s.text(90, 375, "Relative position enters through the Q/K dot product; values remain the content being mixed.", size=16)
    s.save("rope_qk_rotation.svg")


def rmsnorm_vs_layernorm() -> None:
    s = SVG(980, 410)
    s.text(45, 45, "LayerNorm and RMSNorm both rescale token vectors, but RMSNorm skips mean subtraction.", size=17)
    s.rect(85, 115, 170, 70, "x\none token vector", fill=BLUE, stroke=BLUE_EDGE, size=13)
    s.rect(345, 90, 190, 70, "LayerNorm\nsubtract mean\nthen divide by std", fill=GREEN, stroke=GREEN_EDGE, size=13)
    s.rect(345, 220, 190, 70, "RMSNorm\ndivide by RMS\nno mean subtraction", fill=PURPLE, stroke=PURPLE_EDGE, size=13)
    s.line(255, 150, 345, 125)
    s.line(255, 150, 345, 255)
    s.rect(660, 90, 205, 70, "gamma * normalized\n(+ beta usually)", fill=GREEN, stroke=GREEN_EDGE, size=13)
    s.rect(660, 220, 205, 70, "weight * x / rms\n(no bias usually)", fill=PURPLE, stroke=PURPLE_EDGE, size=13)
    s.line(535, 125, 660, 125)
    s.line(535, 255, 660, 255)
    s.text(95, 355, "Both preserve shape (B,T,C). RMSNorm is simpler and common in modern decoder-only blocks.", size=16)
    s.save("rmsnorm_vs_layernorm.svg")


def swiglu_mlp() -> None:
    s = SVG(1040, 420)
    s.text(45, 45, "SwiGLU uses a value path and a gate path before projecting back to the residual width.", size=17)
    s.rect(65, 180, 100, 58, "x\n(B,T,C)", fill=BLUE, stroke=BLUE_EDGE, size=13)
    s.rect(250, 115, 140, 55, "value linear\nW_v x", fill=ORANGE, stroke=ORANGE_EDGE, size=13)
    s.rect(250, 255, 140, 55, "gate linear\nW_g x", fill=ORANGE, stroke=ORANGE_EDGE, size=13)
    s.rect(455, 255, 100, 55, "SiLU", fill=GREEN, stroke=GREEN_EDGE, size=14)
    s.circle(640, 210, 35, "*", fill=PURPLE, stroke=PURPLE_EDGE, size=22)
    s.rect(760, 180, 125, 58, "out linear\nW_o", fill=BLUE, stroke=BLUE_EDGE, size=13)
    s.rect(935, 180, 85, 58, "y\n(B,T,C)", fill=GREEN, stroke=GREEN_EDGE, size=13)
    s.line(165, 209, 250, 142)
    s.line(165, 209, 250, 282)
    s.line(390, 142, 640, 190)
    s.line(390, 282, 455, 282)
    s.line(555, 282, 640, 230)
    s.line(675, 210, 760, 209)
    s.line(885, 209, 935, 209)
    s.text(75, 360, "The external contract is unchanged: a token-wise MLP maps (B,T,C) back to (B,T,C).", size=16)
    s.save("swiglu_mlp.svg")


def sampling_filters() -> None:
    s = SVG(1120, 430)
    s.text(45, 45, "Sampling turns last-position logits into one next token; filters change the candidate set.", size=17)
    stages = [
        ("last logits\n(B,V)", BLUE, BLUE_EDGE),
        ("temperature\nscale", ORANGE, ORANGE_EDGE),
        ("top-k\nfixed count", GREEN, GREEN_EDGE),
        ("top-p\nprob mass", GREEN, GREEN_EDGE),
        ("softmax\nrenormalize", PURPLE, PURPLE_EDGE),
        ("sample\nnext token", RED, RED_EDGE),
    ]
    x0, y, w, h, gap = 55, 160, 130, 62, 42
    for i, (label, fill, stroke) in enumerate(stages):
        x = x0 + i * (w + gap)
        s.rect(x, y, w, h, label, fill=fill, stroke=stroke, size=13)
        if i < len(stages) - 1:
            s.line(x + w, y + h / 2, x + w + gap, y + h / 2)
    s.text(80, 300, "top-k keeps a fixed number of tokens; top-p keeps enough tokens to reach cumulative probability p.", size=16)
    s.text(80, 335, "This is vocabulary filtering, not MoE expert routing.", size=16)
    s.save("sampling_filters.svg")


def prefill_decode_timeline() -> None:
    s = SVG(1080, 420)
    s.text(45, 45, "Inference has two phases: one prompt prefill, then repeated one-token decode steps.", size=17)
    s.rect(80, 140, 300, 90, "prefill\nprocess prompt tokens\nbuild initial KV cache", fill=ORANGE, stroke=ORANGE_EDGE, size=14)
    decode_x = [470, 610, 750, 890]
    for i, x in enumerate(decode_x):
        s.rect(x, 150, 95, 70, f"decode\nstep {i+1}", fill=BLUE, stroke=BLUE_EDGE, size=12)
        if i < len(decode_x) - 1:
            s.line(x + 95, 185, decode_x[i + 1], 185)
    s.line(380, 185, 470, 185)
    s.rect(470, 285, 515, 55, "cache grows: T_prompt -> T_prompt+1 -> T_prompt+2 -> ...", fill=GREEN, stroke=GREEN_EDGE, size=14)
    for x in decode_x:
        s.line(x + 48, 220, x + 48, 285)
    s.text(85, 375, "Prefill is prompt-length dependent; decode repeats once per generated token.", size=16)
    s.save("prefill_decode_timeline.svg")


def kv_cache_layout() -> None:
    s = SVG(1060, 460)
    s.text(45, 45, "KV cache layout: every transformer layer stores keys and values for previous tokens.", size=17)
    y0 = 105
    for layer in range(3):
        y = y0 + layer * 100
        s.rect(70, y, 105, 58, f"layer {layer}", fill=GRAY, stroke=GRAY_EDGE, size=13)
        s.rect(245, y - 10, 210, 35, "K cache\n(B,H,T_cache,Dh)", fill=BLUE, stroke=BLUE_EDGE, size=12)
        s.rect(245, y + 35, 210, 35, "V cache\n(B,H,T_cache,Dh)", fill=GREEN, stroke=GREEN_EDGE, size=12)
        s.line(175, y + 29, 245, y + 8)
        s.line(175, y + 29, 245, y + 52)
    s.rect(610, 125, 150, 55, "new K,V\n(B,H,1,Dh)", fill=ORANGE, stroke=ORANGE_EDGE, size=13)
    s.rect(835, 125, 160, 55, "append on\nsequence axis", fill=PURPLE, stroke=PURPLE_EDGE, size=13)
    s.line(455, 135, 610, 152)
    s.line(760, 152, 835, 152)
    s.text(610, 250, "memory = layers * 2 * B * H * T_cache * Dh * dtype_bytes", size=16)
    s.text(610, 300, "The factor 2 is for K and V. The cache grows linearly with sequence length.", size=16)
    s.save("kv_cache_layout.svg")


def cached_vs_uncached_decode() -> None:
    s = SVG(1100, 440)
    s.text(45, 45, "Cached decode should match uncached final-token logits while avoiding repeated K/V work.", size=17)
    s.text(70, 105, "uncached generation step", size=16)
    s.rect(70, 130, 190, 60, "full sequence\nall tokens", fill=ORANGE, stroke=ORANGE_EDGE, size=13)
    s.rect(340, 130, 165, 60, "full forward\nrecompute K,V", fill=RED, stroke=RED_EDGE, size=13)
    s.rect(585, 130, 150, 60, "final logits", fill=GREEN, stroke=GREEN_EDGE, size=13)
    s.line(260, 160, 340, 160)
    s.line(505, 160, 585, 160)
    s.text(70, 250, "cached decode step", size=16)
    s.rect(70, 275, 150, 60, "newest\ntoken only", fill=ORANGE, stroke=ORANGE_EDGE, size=13)
    s.rect(290, 275, 145, 60, "past K,V\ncache", fill=BLUE, stroke=BLUE_EDGE, size=13)
    s.rect(505, 275, 165, 60, "decode forward\nreuse K,V", fill=PURPLE, stroke=PURPLE_EDGE, size=13)
    s.rect(750, 275, 150, 60, "final logits", fill=GREEN, stroke=GREEN_EDGE, size=13)
    s.line(220, 305, 505, 305)
    s.line(435, 305, 505, 305)
    s.line(670, 305, 750, 305)
    s.text(70, 390, "Correctness compares logits, not sampled text. Use eval mode so dropout cannot hide cache bugs.", size=16)
    s.save("cached_vs_uncached_decode.svg")


def benchmark_metrics_dashboard() -> None:
    s = SVG(1080, 430)
    s.text(45, 45, "Inference benchmarks need workload shape, speed, memory, and a cached-vs-uncached comparison.", size=17)
    cards = [
        ("prefill\nlatency", "prompt cost"),
        ("decode\nlatency", "per token"),
        ("tokens/sec", "throughput"),
        ("peak\nmemory", "capacity"),
        ("speedup", "cached / uncached"),
    ]
    x0, y, w, h, gap = 70, 120, 150, 95, 35
    colors = [(ORANGE, ORANGE_EDGE), (BLUE, BLUE_EDGE), (GREEN, GREEN_EDGE), (PURPLE, PURPLE_EDGE), (RED, RED_EDGE)]
    for i, ((title, subtitle), (fill, stroke)) in enumerate(zip(cards, colors)):
        x = x0 + i * (w + gap)
        s.rect(x, y, w, h, title, fill=fill, stroke=stroke, size=16)
        s.text(x + w / 2, y + h + 32, subtitle, size=14, anchor="middle")
    s.rect(230, 300, 625, 55, "always record: model config, device, dtype, prompt length, new tokens, seed", fill=GRAY, stroke=GRAY_EDGE, size=15)
    s.text(95, 395, "A tokens/sec number without prompt length and cache mode is not interpretable.", size=16)
    s.save("benchmark_metrics_dashboard.svg")


def training_step_flow() -> None:
    s = SVG(1120, 430)
    s.text(45, 45, "One training step: shifted targets create a supervised signal at every position.", size=17)
    stages = [
        ("tokens", ORANGE, ORANGE_EDGE),
        ("sample\nx,y", BLUE, BLUE_EDGE),
        ("model\nforward", PURPLE, PURPLE_EDGE),
        ("CE loss\n+ MoE aux", RED, RED_EDGE),
        ("backward", RED, RED_EDGE),
        ("clip grad\nAdamW", GREEN, GREEN_EDGE),
        ("checkpoint\nperiodically", GRAY, GRAY_EDGE),
    ]
    x0, y, w, h, gap = 45, 160, 120, 62, 30
    for i, (label, fill, stroke) in enumerate(stages):
        x = x0 + i * (w + gap)
        s.rect(x, y, w, h, label, fill=fill, stroke=stroke, size=13)
        if i < len(stages) - 1:
            s.line(x + w, y + h / 2, x + w + gap, y + h / 2)
    s.text(70, 310, "Dense and MoE models share the same language-model objective; MoE adds an auxiliary router loss.", size=16)
    s.save("training_step_flow.svg")


def generation_loop() -> None:
    s = SVG(1040, 430)
    s.text(45, 45, "Autoregressive generation loops because each sampled token becomes part of the next input.", size=17)
    stages = [
        ("current ids", BLUE, BLUE_EDGE),
        ("crop\ncontext", GRAY, GRAY_EDGE),
        ("model\nforward", PURPLE, PURPLE_EDGE),
        ("last logits", GREEN, GREEN_EDGE),
        ("filter\nsample", RED, RED_EDGE),
        ("append\ntoken", ORANGE, ORANGE_EDGE),
    ]
    x0, y, w, h, gap = 60, 160, 120, 62, 32
    centers = []
    for i, (label, fill, stroke) in enumerate(stages):
        x = x0 + i * (w + gap)
        centers.append((x + w / 2, y + h / 2))
        s.rect(x, y, w, h, label, fill=fill, stroke=stroke, size=13)
        if i < len(stages) - 1:
            s.line(x + w, y + h / 2, x + w + gap, y + h / 2)
    s.path(f"M {x0 + 5 * (w + gap) + w/2} {y + h} C 820 330, 160 330, {x0 + w/2} {y + h}")
    s.text(90, 365, "The model outputs logits for all visible positions, but generation consumes only the final row.", size=16)
    s.save("generation_loop.svg")


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
    decoder_shape_ladder()
    training_vs_generation_paths()
    causal_mask_matrix()
    attention_memory_scaling()
    rope_qk_rotation()
    rmsnorm_vs_layernorm()
    swiglu_mlp()
    sampling_filters()
    prefill_decode_timeline()
    kv_cache_layout()
    cached_vs_uncached_decode()
    benchmark_metrics_dashboard()
    training_step_flow()
    generation_loop()
    print(f"wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
