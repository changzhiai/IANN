#!/usr/bin/env python
"""Generate the IANN logo asset set from a single geometry spec.

The icon is concept A: atom i, its six neighbours, and the cutoff radius r_c.
The wordmark is converted to outlines so the SVG renders identically on a machine
with no particular font installed -- SVG <text> would silently substitute.

    python make_logo.py          # writes docs/source/_static/logo/
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Circle
from matplotlib.path import Path
from matplotlib.textpath import TextPath

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "source", "_static", "logo")

# ---------------------------------------------------------------- palette
INK = "#16324F"     # navy, from the paper's figure palette
ACCENT = "#E07A2F"  # orange, atom i
INK_LIGHT = "#FFFFFF"

# ---------------------------------------------------------------- geometry
# One spec, used for both the SVG and the PNG, so they cannot drift apart.
# Canvas is 256x256; the hexagon of neighbours sits inside the cutoff ring.
CX = CY = 128.0
R_CUT = 96.0        # cutoff ring
R_HEX = 68.0        # neighbour shell
R_I = 22.0          # atom i
R_J = 13.0          # neighbour atoms
W_EDGE = 8.0

HEX = [  # six neighbours at 60 degree steps
    (196.0, 128.0),
    (162.0, 187.0),
    (94.0, 187.0),
    (60.0, 128.0),
    (94.0, 69.0),
    (162.0, 69.0),
]

# Small-size variant: solid ring, four neighbours, fatter centre. Below ~24 px the
# dashed ring of the primary mark turns to mush. It is also drawn larger in the
# frame -- a favicon has no room for the primary mark's breathing space, and the
# ring needs more opacity to survive being 1 px wide.
CROSS = [(128.0, 44.0), (128.0, 212.0), (44.0, 128.0), (212.0, 128.0)]
R_CUT_SMALL = 108.0


def icon_body(ink, accent, dashed=True):
    """The icon's SVG elements, minus the <svg> wrapper."""
    if dashed:
        ring = (f'  <circle cx="{CX:g}" cy="{CY:g}" r="{R_CUT:g}" fill="none" '
                f'stroke="{ink}" stroke-opacity="0.30" stroke-width="5" '
                f'stroke-dasharray="10 9"/>\n')
        nodes, r_i, r_j, w = HEX, R_I, R_J, W_EDGE
    else:
        ring = (f'  <circle cx="{CX:g}" cy="{CY:g}" r="{R_CUT_SMALL:g}" fill="none" '
                f'stroke="{ink}" stroke-opacity="0.45" stroke-width="13"/>\n')
        nodes, r_i, r_j, w = CROSS, 32.0, 21.0, 15.0

    out = [ring]
    out.append(f'  <g stroke="{ink}" stroke-width="{w:g}" stroke-linecap="round" fill="none">\n')
    for x, y in nodes:
        out.append(f'    <path d="M{CX:g} {CY:g} L{x:g} {y:g}"/>\n')
    out.append("  </g>\n")
    out.append(f'  <g fill="{ink}">\n')
    for x, y in nodes:
        out.append(f'    <circle cx="{x:g}" cy="{y:g}" r="{r_j:g}"/>\n')
    out.append("  </g>\n")
    out.append(f'  <circle cx="{CX:g}" cy="{CY:g}" r="{r_i:g}" fill="{accent}"/>\n')
    return "".join(out)


def write_icon(path, ink=INK, accent=ACCENT, dashed=True, label="IANN"):
    # The small variant is drawn larger, so crop the viewBox to match or it would
    # render with a margin a favicon cannot spare.
    box = "0 0 256 256" if dashed else "12 12 232 232"
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{box}" '
        f'width="256" height="256" role="img" aria-label="{label}">\n'
        f"  <title>IANN &#8212; atom i, its neighbours, and the cutoff radius</title>\n"
        f"{icon_body(ink, accent, dashed)}"
        f"</svg>\n"
    )
    with open(path, "w") as fh:
        fh.write(svg)


# ---------------------------------------------------------------- text outlines
def text_to_svg_path(s, size, weight="normal", family="DejaVu Sans"):
    """Return (path_d, width, height, y_min) with y flipped into SVG's frame."""
    tp = TextPath((0, 0), s, size=size,
                  prop=FontProperties(family=family, weight=weight))
    d = []
    for verts, code in tp.iter_segments():
        if code == Path.MOVETO:
            d.append(f"M{verts[0]:.2f} {-verts[1]:.2f}")
        elif code == Path.LINETO:
            d.append(f"L{verts[0]:.2f} {-verts[1]:.2f}")
        elif code == Path.CURVE3:
            d.append(f"Q{verts[0]:.2f} {-verts[1]:.2f} {verts[2]:.2f} {-verts[3]:.2f}")
        elif code == Path.CURVE4:
            d.append(f"C{verts[0]:.2f} {-verts[1]:.2f} {verts[2]:.2f} {-verts[3]:.2f} "
                     f"{verts[4]:.2f} {-verts[5]:.2f}")
        elif code == Path.CLOSEPOLY:
            d.append("Z")
    ext = tp.get_extents()
    return " ".join(d), ext.width, ext.height, ext.y0


def write_lockup(path, ink=INK, accent=ACCENT, tagline=True):
    """Icon on the left, outlined wordmark on the right."""
    icon_box = 160.0                      # rendered icon size
    scale = icon_box / 256.0
    pad = 20.0
    gap = 24.0

    word_d, word_w, word_h, _ = text_to_svg_path("IANN", 84, weight="bold")
    tag_d, tag_w, tag_h, _ = text_to_svg_path(
        "InterAtomic Neural Network", 25, weight="normal")

    text_x = pad + icon_box + gap
    # Centre the text block on the icon's centre line rather than guessing a
    # baseline: cap height for the wordmark, plus the tagline if present.
    block_h = word_h + (38.0 if tagline else 0.0)
    icon_mid = pad + icon_box / 2.0
    word_y = icon_mid - block_h / 2.0 + word_h
    tag_y = word_y + 38.0

    width = text_x + max(word_w, tag_w if tagline else 0) + pad * 1.6
    height = pad * 2 + icon_box

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0f} {height:.0f}" '
        f'width="{width:.0f}" height="{height:.0f}" role="img" '
        f'aria-label="IANN &#8212; InterAtomic Neural Network">\n',
        "  <title>IANN &#8212; InterAtomic Neural Network</title>\n",
        f'  <g transform="translate({pad:g} {pad:g}) scale({scale:.6f})">\n',
        icon_body(ink, accent),
        "  </g>\n",
        f'  <path transform="translate({text_x:.2f} {word_y:.2f})" fill="{ink}" d="{word_d}"/>\n',
    ]
    if tagline:
        parts.append(
            f'  <path transform="translate({text_x + 2:.2f} {tag_y:.2f})" fill="{ink}" '
            f'fill-opacity="0.65" d="{tag_d}"/>\n')
    parts.append("</svg>\n")
    with open(path, "w") as fh:
        fh.write("".join(parts))


# ---------------------------------------------------------------- raster
def write_png(path, px, dashed=True, ink=INK, accent=ACCENT, bg=None):
    """Rasterise the icon from the same geometry spec."""
    fig = plt.figure(figsize=(px / 100.0, px / 100.0), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 256)
    ax.set_ylim(256, 0)          # flip so the spec's y matches the SVG
    ax.axis("off")
    if bg:
        fig.patch.set_facecolor(bg)
    else:
        fig.patch.set_alpha(0.0)

    # Crop to the art so a favicon is not mostly empty canvas.
    half = (R_CUT if dashed else R_CUT_SMALL) + (12 if dashed else 6)
    ax.set_xlim(CX - half, CX + half)
    ax.set_ylim(CY + half, CY - half)

    nodes, r_i, r_j, w = ((HEX, R_I, R_J, W_EDGE) if dashed
                          else (CROSS, 32.0, 21.0, 15.0))
    lw = lambda pts: pts * px / (2 * half) * 72.0 / 100.0   # spec units -> points

    if dashed:
        ax.add_patch(Circle((CX, CY), R_CUT, fill=False, ec=ink, alpha=0.30,
                            lw=lw(5), ls=(0, (2.0, 1.8)), zorder=1))
    else:
        ax.add_patch(Circle((CX, CY), R_CUT_SMALL, fill=False, ec=ink, alpha=0.45,
                            lw=lw(13), zorder=1))
    for x, y in nodes:
        ax.plot([CX, x], [CY, y], color=ink, lw=lw(w), solid_capstyle="round", zorder=2)
    for x, y in nodes:
        ax.add_patch(Circle((x, y), r_j, color=ink, zorder=3))
    ax.add_patch(Circle((CX, CY), r_i, color=accent, zorder=4))

    fig.savefig(path, dpi=100, transparent=bg is None,
                facecolor=fig.get_facecolor() if bg else "none")
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    write_icon(os.path.join(OUT, "iann-icon.svg"))
    write_icon(os.path.join(OUT, "iann-icon-dark.svg"), ink=INK_LIGHT)
    write_icon(os.path.join(OUT, "iann-favicon.svg"), dashed=False)
    write_lockup(os.path.join(OUT, "iann-logo.svg"))
    write_lockup(os.path.join(OUT, "iann-logo-dark.svg"), ink=INK_LIGHT)
    write_lockup(os.path.join(OUT, "iann-logo-notagline.svg"), tagline=False)
    # For the docs sidebar: white ink on the dark nav header, and no tagline,
    # which would be ~4 px tall at the sidebar's width.
    write_lockup(os.path.join(OUT, "iann-logo-notagline-dark.svg"),
                 ink=INK_LIGHT, tagline=False)
    for px in (512, 256, 128, 64, 32, 16):
        write_png(os.path.join(OUT, f"iann-icon-{px}.png"), px,
                  dashed=(px >= 64))
    print("wrote:")
    for f in sorted(os.listdir(OUT)):
        print("  ", f, os.path.getsize(os.path.join(OUT, f)), "bytes")


if __name__ == "__main__":
    main()
