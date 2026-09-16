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


def icon_body(ink, accent, dashed=True, ring_opacity=0.30):
    """The icon's SVG elements, minus the <svg> wrapper."""
    if dashed:
        ring = (f'  <circle cx="{CX:g}" cy="{CY:g}" r="{R_CUT:g}" fill="none" '
                f'stroke="{ink}" stroke-opacity="{ring_opacity:g}" stroke-width="5" '
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
    """Return (path_d, width, height, x_min) with y flipped into SVG's frame.

    x_min is the glyph run's left side bearing: the path's own coordinates do not
    start at 0, so a caller that wants the ink to begin at a chosen x must
    subtract it. Without that the lockup ends up with unequal side margins.
    """
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
    return " ".join(d), ext.width, ext.height, ext.x0


DEFAULT_TAGLINE = ["InterAtomic Neural Network"]


DEFAULT_LABEL = "IANN &#8212; InterAtomic Neural Network"


def write_lockup(path, ink=INK, accent=ACCENT, tagline=True, ring_opacity=0.30,
                 pad=20.0, tag_size=25.0, tag_opacity=0.65, line_gap=1.52,
                 label=DEFAULT_LABEL):
    """Icon on the left, outlined wordmark on the right.

    ``tagline`` is ``True`` for the default one-liner, ``False`` for none, or a
    list of strings to set explicitly -- one per line.

    Wrapping the tagline is not cosmetic. The tagline is usually the widest
    element, so it sets the whole viewBox width; and because a consumer such as
    the docs sidebar renders the ``<img>`` at a *fixed pixel width*, a wider
    viewBox scales everything down. A long single-line tagline therefore shrinks
    the icon and wordmark and still renders itself at ~8 px, too small to read.
    Split across lines so that no line exceeds the wordmark's width and the
    viewBox stays as narrow as the no-tagline variant, which buys about 50% more
    rendered tagline size for free. Measured for the 219 px sidebar: one line at
    size 25 renders 8.1 px, two lines at size 22 render 11.9 px.

    ``pad`` is transparent margin on all four sides, in the same units as the
    viewBox. It gives the mark breathing room when it sits in a page next to
    other content, but it is dead space when something else already provides
    the margin -- the docs sidebar has its own padding, so the sidebar variant
    is generated with ``pad=0`` and fills its box.

    ``line_gap`` defaults to 1.52 because 1.52 * 25 = 38, the baseline step the
    single-line variants were built with; keeping it makes their output
    byte-identical to before this parameter existed.
    """
    # The icon is a sparse ring of small nodes; the wordmark is solid, so
    # matching their bounding boxes can still look left-heavy. That is fixed by
    # raising the ring's opacity (see ring_opacity), NOT by enlarging the icon:
    # a bigger icon crowds the wordmark and leaves too little air between them.
    icon_box = 160.0                      # rendered size of the icon's artwork
    gap = 24.0

    # The icon is composed in a 256-unit box, but its artwork only spans the
    # cutoff ring -- radius R_CUT plus half its 5-unit stroke -- so roughly 23%
    # of that box is empty. Embedding the box whole turns that emptiness into
    # extra padding on the left of the mark, which makes the lockup look
    # left-weighted and defeats the sidebar's centred <img>. Scale and offset by
    # the artwork's true extent so it starts exactly at `pad`.
    art = 2.0 * (R_CUT + 2.5)
    scale = icon_box / art
    icon_x = pad - (CX - art / 2.0) * scale

    if tagline is True:
        lines = list(DEFAULT_TAGLINE)
    elif not tagline:
        lines = []
    elif isinstance(tagline, str):
        lines = [tagline]
    else:
        lines = list(tagline)

    word_d, word_w, word_h, word_x0 = text_to_svg_path("IANN", 84, weight="bold")
    tags = [text_to_svg_path(t, tag_size, weight="normal") for t in lines]

    text_x = pad + icon_box + gap
    # Equal padding on both sides, so the artwork is centred in the canvas.
    width = text_x + max([word_w] + [w for _, w, _, _ in tags]) + pad
    height = pad * 2 + icon_box
    icon_y = (height - icon_box) / 2.0 - (CY - art / 2.0) * scale

    # Centre the text block on the icon's centre line rather than guessing a
    # baseline: cap height for the wordmark, plus one step per tagline line.
    step = tag_size * line_gap
    block_h = word_h + step * len(tags)
    word_y = height / 2.0 - block_h / 2.0 + word_h

    # The accessible name is deliberately independent of what is drawn: the
    # no-tagline variants are the same product, and a screen-reader user gains
    # nothing from a purely visual omission.
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0f} {height:.0f}" '
        f'width="{width:.0f}" height="{height:.0f}" role="img" '
        f'aria-label="{label}">\n',
        f"  <title>{label}</title>\n",
        f'  <g transform="translate({icon_x:.3f} {icon_y:.3f}) scale({scale:.6f})">\n',
        icon_body(ink, accent, ring_opacity=ring_opacity),
        "  </g>\n",
        f'  <path transform="translate({text_x - word_x0:.2f} {word_y:.2f})" '
        f'fill="{ink}" d="{word_d}"/>\n',
    ]
    # The +2 is an optical nudge: the tagline's lighter weight makes a flush
    # left edge read as overhanging the wordmark above it.
    for i, (tag_d, _tw, _th, tag_x0) in enumerate(tags):
        parts.append(
            f'  <path transform="translate({text_x - tag_x0 + 2:.2f} '
            f'{word_y + step * (i + 1):.2f})" fill="{ink}" '
            f'fill-opacity="{tag_opacity}" d="{tag_d}"/>\n')
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
    # The one-line-tagline lockups are gone -- iann-logo.svg, its dark variant
    # and the padded no-tagline light variant. The two-line lockups below
    # replaced them, so regenerating those names would only resurrect files no
    # longer referenced anywhere.
    #
    # Flush -- pad=0, artwork to the edges. For a host that
    # supplies its own spacing and sizes the image itself, such as the README,
    # where the built-in margin only shrinks the artwork inside the width= it
    # is given.
    write_lockup(os.path.join(OUT, "iann-logo-notagline-flush.svg"),
                 tagline=False, pad=0.0)
    write_lockup(os.path.join(OUT, "iann-logo-notagline-dark.svg"),
                 ink=INK_LIGHT, tagline=False, ring_opacity=0.55)
    # The two-line lockups, flush and with the full "... Framework" tagline.
    # Every dimension is chosen for the docs sidebar, the narrowest place either
    # one appears (219 px):
    #   * pad=0 -- the sidebar header supplies its own padding, so built-in
    #     margin would only shrink the artwork inside a fixed-width <img>.
    #   * the tagline is split over two lines, neither wider than "IANN", so
    #     the viewBox stays 405x160 and the tagline renders at 11.9 px instead
    #     of the 8.1 px a single line would give.
    #
    # -dark is white ink for a dark ground, and raises the ring to 0.55 and the
    # tagline to 0.75 because white at the default opacities washes out on the
    # mid-blue nav header. -flush is navy for a light ground such as the README,
    # where those same raised values would read as too heavy, so it keeps the
    # defaults.
    two_line = dict(pad=0.0, tag_size=22.0,
                    tagline=["InterAtomic Neural", "Network Framework"],
                    label="IANN &#8212; InterAtomic Neural Network Framework")
    write_lockup(os.path.join(OUT, "iann-logo-dark.svg"), ink=INK_LIGHT,
                 ring_opacity=0.55, tag_opacity=0.75, **two_line)
    write_lockup(os.path.join(OUT, "iann-logo-flush.svg"), ink=INK, **two_line)
    for px in (512, 256, 128, 64, 32, 16):
        write_png(os.path.join(OUT, f"iann-icon-{px}.png"), px,
                  dashed=(px >= 64))
    print("wrote:")
    for f in sorted(os.listdir(OUT)):
        print("  ", f, os.path.getsize(os.path.join(OUT, f)), "bytes")


if __name__ == "__main__":
    main()
