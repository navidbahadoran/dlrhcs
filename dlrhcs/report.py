"""Empirical figure-snippet generators: turn a saved ``run_ar2`` JSON record into the
paper's pgfplots/TikZ figures.  Numbers are emitted by the code, never transcribed by
hand.  The simulation tables and figure coordinates are built separately by the
self-contained ``scripts/sim_report.py``; this module is the empirical counterpart,
driven by ``scripts/emp_report.py``.
"""
from __future__ import annotations

import os


def empirical_irf_figure(z: dict) -> str:
    """Empirical impulse-response path psi_h vs horizon with 95% bars
    (cor:irf_body applied to the lag means).  Reads ``z['derived']['irf']``."""
    irf = z["derived"]["irf"]
    hs = sorted(int(h) for h in irf)
    pts = " ".join(f"({h},{irf[str(h)]['est']:.4f}) +- (0,{1.96*irf[str(h)]['se']:.4f})"
                   for h in hs)
    L = [r"\begin{tikzpicture}",
         r"\begin{axis}[width=0.8\textwidth,height=0.5\textwidth,",
         r"  xlabel={Horizon $h$}, ylabel={IRF $\psi_h$}, grid=both]",
         r"\addplot+[mark=*,thick,error bars/.cd,y dir=both,y explicit] coordinates {"
         + pts + "};",
         r"\end{axis}", r"\end{tikzpicture}"]
    return "\n".join(L)


def empirical_coefpath_figure(z: dict, series_key: str, ylabel: str, ref: float,
                              ymin: float, ymax: float) -> str:
    """Per-period coefficient path (fig:emp_coefpath): cross-sectional average of the
    estimated persistence in each period, with a dashed full-sample headline line and a
    dotted stationarity boundary at one.  ``series_key`` is ``cum_t`` (housing AR(2),
    a_t+b_t) or ``a_t`` (unemployment AR(1)).  ``ref`` is the headline value."""
    cp = z["derived"]["coef_path"][series_key]
    months = z["months"]

    def _yr(m):
        y, mm = str(m).split("-")[:2]
        return int(y) + (int(mm) - 1) / 12.0

    xs = [_yr(m) for m in months]
    co = " ".join(f"({x:.3f},{v:.4f})" for x, v in zip(xs, cp))
    x0, x1 = int(min(xs)), int(max(xs)) + 1
    color = "red!70!black" if series_key == "a_t" else "blue!70!black"
    L = [r"\begin{tikzpicture}",
         r"\begin{axis}[width=0.8\textwidth,height=0.40\textwidth,",
         f"  xlabel={{Year}}, ylabel={{{ylabel}}}, grid=major,",
         f"  ymin={ymin}, ymax={ymax}, xmin={x0}, xmax={x1}]",
         f"\\addplot[mark=none,thick,{color}] coordinates {{{co}}};",
         f"\\draw[dashed] (axis cs:{x0},{ref:.3f}) -- (axis cs:{x1},{ref:.3f});",
         f"\\draw[densely dotted] (axis cs:{x0},1) -- (axis cs:{x1},1);",
         r"\end{axis}", r"\end{tikzpicture}"]
    return "\n".join(L)


def companion_root_figure(z: dict) -> str:
    """Estimated companion eigenvalues against the unit circle (companion of the
    global lag means): the visual form of ``lag-1 may exceed one yet the process is
    stationary'' -- a stable AR(2) has both roots inside the circle."""
    import numpy as _np
    a = z["targets"]["lag1_mean"]["est"]
    b = z["targets"].get("lag2_mean", {}).get("est", 0.0)
    ev = _np.linalg.eigvals(_np.array([[a, b], [1.0, 0.0]]))
    pts = " ".join(f"({e.real:.4f},{e.imag:.4f})" for e in ev)
    rad = z.get("derived", {}).get("companion_radius", float(_np.max(_np.abs(ev))))
    L = [r"\begin{tikzpicture}",
         r"\begin{axis}[width=0.55\textwidth,axis equal image,",
         r"  xlabel={Re}, ylabel={Im}, xmin=-1.25,xmax=1.25,ymin=-1.25,ymax=1.25,",
         r"  grid=major, axis lines=middle, enlargelimits=false]",
         r"\addplot[domain=0:360,samples=120,thick] ({cos(x)},{sin(x)});",
         r"\addplot[only marks,mark=*,mark size=2.6pt] coordinates {" + pts + "};",
         f"\\node[anchor=north east] at (rel axis cs:0.98,0.98) "
         f"{{$\\widehat\\rho_{{\\rm pt}}={rad:.3f}$}};",
         r"\end{axis}", r"\end{tikzpicture}"]
    return "\n".join(L)


def coef_hist_figure(z: dict) -> str:
    """Histogram of the per-cell estimated lag-1 coefficient surface (its
    heterogeneity), split by group when available.  Reads
    ``z['derived']['coef_hist']`` written by ``run_ar2``."""
    h = z.get("derived", {}).get("coef_hist")
    if not h:
        return "% coef_hist not present in JSON (re-run the empirical stage)\n"
    edges = h["edges"]

    def _series(counts):
        return " ".join(f"({edges[i]:.4f},{counts[i]})" for i in range(len(counts))) \
               + f" ({edges[-1]:.4f},0)"
    L = [r"\begin{tikzpicture}",
         r"\begin{axis}[width=0.8\textwidth,height=0.45\textwidth,",
         r"  xlabel={Per-cell lag-1 coefficient $\hat a_{ti}$}, ylabel={count},",
         r"  ymin=0, grid=major, legend pos=north east, area legend]"]
    if "counts_g1" in h:
        lab = h.get("labels", ["group 0", "group 1"])
        L.append(r"\addplot+[ybar interval,fill opacity=0.45,draw opacity=0.7] "
                 r"coordinates {" + _series(h["counts_g0"]) + "};")
        L.append(r"\addplot+[ybar interval,fill opacity=0.45,draw opacity=0.7] "
                 r"coordinates {" + _series(h["counts_g1"]) + "};")
        L.append(f"\\legend{{{lab[0].replace('_', chr(92)+'_')}, "
                 f"{lab[1].replace('_', chr(92)+'_')}}}")
    else:
        L.append(r"\addplot+[ybar interval,fill opacity=0.5] coordinates {"
                 + _series(h["counts_all"]) + "};")
    L.append(f"\\draw[dashed,thick] ({{axis cs:{h['mean']:.4f},0}}|-{{rel axis cs:0,0}}) "
             f"-- ({{axis cs:{h['mean']:.4f},0}}|-{{rel axis cs:0,1}});")
    L += [r"\end{axis}", r"\end{tikzpicture}"]
    return "\n".join(L)


def write_tex(text, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text + "\n")
    return path
