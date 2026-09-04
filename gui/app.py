"""
Local launcher UI for the autocallable pricer.

Deliberately a THIN wrapper. It does three things -- edit an input file, run a
command, look at the results -- and it does them by shelling out to the
existing scripts rather than importing them. Consequences worth stating:

  * Nothing in the pricer had to change to support this. main.py,
    run_scenarios.py and model_quality.py are untouched and remain the real
    interface; this is a convenience layer over them.
  * What you see in the output pane is exactly what the terminal would show,
    because it IS the terminal output, streamed live.
  * A run here and a run from the shell are the same run. No behaviour can
    drift between the two, because there is only one code path.

Configs are edited as raw YAML rather than through generated form widgets.
config.yaml is ~300 lines of heavily commented settings; a form would strip
those comments, and would need updating every time a field is added. The
editor validates on save instead.

Run with:
    streamlit run gui/app.py
"""
from __future__ import annotations

import glob
import os
import re
import subprocess
import sys
import time
from datetime import datetime

import pandas as pd
import streamlit as st
import yaml

try:
    import altair as alt
except ImportError:                                  # pragma: no cover
    alt = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = os.path.join(ROOT, ".venv", "bin", "python")
if not os.path.exists(PYTHON):
    PYTHON = sys.executable

MAX_OUTPUT_LINES = 600      # keep the DOM from growing without bound on long runs

st.set_page_config(page_title="Autocall Pricer", page_icon="🎲", layout="wide")


# ── helpers ───────────────────────────────────────────────────────────────────

def config_files() -> list[str]:
    """Every YAML input the pricer accepts, repo-relative."""
    found = []
    if os.path.exists(os.path.join(ROOT, "config.yaml")):
        found.append("config.yaml")
    for p in sorted(glob.glob(os.path.join(ROOT, "configs", "*.yaml"))):
        found.append(os.path.relpath(p, ROOT))
    for p in sorted(glob.glob(os.path.join(ROOT, "scenarios", "*.yaml"))):
        found.append(os.path.relpath(p, ROOT))
    return found


def result_files() -> list[str]:
    found = []
    for pattern in ("results/*.csv", "scenarios/*.csv"):
        for p in sorted(glob.glob(os.path.join(ROOT, pattern)), reverse=True):
            found.append(os.path.relpath(p, ROOT))
    return found


def detect_models(df: pd.DataFrame) -> list[str]:
    """
    Model names present in a results CSV.

    Detected from the '<model>_se' columns rather than hardcoded, because the
    archived files in results/ predate LSV and carry only Local Vol and Heston.
    A hardcoded list would silently drop or crash on those.
    """
    return [c[:-3] for c in df.columns if c.endswith("_se") and c[:-3] in df.columns]


def run_streaming(cmd: list[str], out_area) -> int:
    """
    Run `cmd` and stream its output into `out_area` as it arrives.

    PYTHONUNBUFFERED is set and -u passed because Python block-buffers stdout
    when it is a pipe rather than a tty -- without it nothing appears until the
    process exits, which on a 7-minute scenario sweep is indistinguishable
    from a hang.
    """
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    started = time.time()
    lines: list[str] = []

    proc = subprocess.Popen(
        cmd, cwd=ROOT, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    st.session_state["proc_pid"] = proc.pid

    assert proc.stdout is not None
    for line in proc.stdout:
        lines.append(line.rstrip("\n"))
        if len(lines) > MAX_OUTPUT_LINES:
            trimmed = len(lines) - MAX_OUTPUT_LINES
            shown = [f"… {trimmed} earlier lines trimmed …"] + lines[-MAX_OUTPUT_LINES:]
        else:
            shown = lines
        out_area.code("\n".join(shown), language="text")

    code = proc.wait()
    elapsed = time.time() - started
    lines.append("")
    lines.append(f"[exit {code}]  {elapsed:.1f}s")
    out_area.code("\n".join(lines[-(MAX_OUTPUT_LINES + 2):]), language="text")

    st.session_state["last_run"] = {
        "cmd": " ".join(cmd), "code": code,
        "elapsed": elapsed, "when": datetime.now().strftime("%H:%M:%S"),
    }
    return code


# ── categorical palette ───────────────────────────────────────────────────────
# Validated slots (dataviz validate_palette.js, --pairs all): light CVD dE 9.2 /
# normal-vision 24.0; dark 9.4 / 20.9. Both modes pass every hard gate.
#
# Colour is bound to the MODEL NAME, never to column position. An archived CSV
# holding only Local Vol and Heston therefore paints them the same hues as a
# current file that also has LSV -- a reader who learned "Heston is orange" is
# not misled by opening a different file.
_SLOTS_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
_SLOTS_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"]
_MODEL_SLOT = {
    "Local Vol": 0, "Basket Local Vol": 0,
    "Heston": 1, "Basket Heston": 1,
    "LSV": 2, "Basket LSV": 2,
    "SABR": 3,
}


def _is_dark() -> bool:
    try:
        return getattr(st.context.theme, "type", "light") == "dark"
    except Exception:
        return False


def model_colors(models: list[str]) -> tuple[list[str], list[str]]:
    """(domain, range) for an Altair colour scale, keyed on model name."""
    slots = _SLOTS_DARK if _is_dark() else _SLOTS_LIGHT
    used: set[int] = set()
    out: list[str] = []
    for m in models:
        i = _MODEL_SLOT.get(m)
        if i is None or i in used:
            i = next(k for k in range(len(slots)) if k not in used)
        used.add(i)
        out.append(slots[i % len(slots)])
    return list(models), out



MANUAL_PATH = os.path.join(ROOT, "USER_MANUAL.md")


def manual_sections() -> list[tuple[str, str]]:
    """
    Split USER_MANUAL.md into (heading, body) pairs on its top-level `## `
    headings.

    The manual is ~1300 lines. Rendering it whole makes it unreadable and slow,
    and Streamlit has no in-page anchors, so the document's own table of
    contents links would not work anyway. Splitting lets the reader jump
    straight to a section, which is what the TOC was for.
    """
    if not os.path.exists(MANUAL_PATH):
        return []
    lines = open(MANUAL_PATH, encoding="utf-8").read().splitlines(keepends=True)
    sections: list[tuple[str, list[str]]] = []
    preamble: list[str] = []
    fenced = False
    for line in lines:
        # Never treat a "## " inside a fenced code block as a heading.
        if line.lstrip().startswith("```"):
            fenced = not fenced
        if not fenced and line.startswith("## "):
            sections.append((line[3:].strip(), [line]))
        elif sections:
            sections[-1][1].append(line)
        else:
            preamble.append(line)
    out = [(h, "".join(b)) for h, b in sections]
    if preamble and "".join(preamble).strip():
        out.insert(0, ("(front matter)", "".join(preamble)))
    return out


# ── header ────────────────────────────────────────────────────────────────────

st.title("Autocall Pricer")
st.caption(
    f"Launcher for the scripts in `{os.path.basename(ROOT)}` — "
    "edit an input, run a command, read the results. "
    "Everything runs through the existing CLI, unchanged."
)

tab_edit, tab_run, tab_results, tab_manual = st.tabs(
    ["Inputs", "Run", "Results", "Manual"])


# ── 1. edit inputs ────────────────────────────────────────────────────────────

with tab_edit:
    files = config_files()
    if not files:
        st.error("No YAML inputs found under the project root.")
        st.stop()

    left, right = st.columns([1, 2.4])

    with left:
        chosen = st.selectbox("File", files, key="edit_file")
        path = os.path.join(ROOT, chosen)
        on_disk = open(path).read()

        st.caption(f"`{chosen}` — {len(on_disk.splitlines())} lines")

        # Reload from disk when the selection changes, but do NOT clobber
        # unsaved edits on every rerun.
        if st.session_state.get("_loaded_file") != chosen:
            st.session_state["editor"] = on_disk
            st.session_state["_loaded_file"] = chosen
            # Point Save-As at the folder this file came from. Streamlit
            # widget state is sticky once a key exists, so the radio's `index`
            # is honoured only on its first render -- the default has to be
            # pushed through session_state instead, or the folder silently
            # keeps whatever was chosen last.
            _d = os.path.dirname(chosen)
            st.session_state["saveas_dir"] = (
                _d if _d in ("configs", "scenarios") else "configs")
            st.session_state.pop("saveas_name", None)

        if st.button("Reload from disk", width="stretch"):
            st.session_state["editor"] = on_disk
            st.rerun()

    with right:
        text = st.text_area(
            "YAML", key="editor", height=560,
            label_visibility="collapsed",
        )

    # Validate continuously; saving invalid YAML would only fail later, inside
    # a run, where the traceback is far less obvious.
    err = None
    try:
        parsed = yaml.safe_load(text)
        if not isinstance(parsed, dict):
            err = "Top level of the document is not a mapping."
    except yaml.YAMLError as e:
        err = str(e)

    dirty = text != on_disk

    c1, c2, c3, c4 = st.columns([1, 1, 2, 2])
    with c1:
        if st.button("Save", type="primary", disabled=bool(err) or not dirty,
                     width="stretch"):
            open(path, "w").write(text)
            st.success(f"Saved {chosen}")
            st.rerun()
    with c2:
        save_as = st.popover("Save as…", width="stretch")
        with save_as:
            # The destination has to be selectable, not fixed. The Run tab
            # reads --config from configs/ and --scenarios from scenarios/,
            # so hardcoding one of them makes the other kind of file
            # impossible to create here.
            folders = ["configs", "scenarios"]
            # No index= here: session_state drives this widget (set when the
            # opened file changes, above). Passing both makes Streamlit warn
            # that a default and a session-state value are competing.
            st.session_state.setdefault("saveas_dir", "configs")
            dest_dir = st.radio(
                "Folder", folders,
                horizontal=True,
                captions=["deal configs (--config)", "scenario specs (--scenarios)"],
                key="saveas_dir",
            )
            default_name = ("sce_new.yaml" if dest_dir == "scenarios"
                            else "config_new.yaml")
            if st.session_state.get("_saveas_dir_prev") != dest_dir:
                st.session_state["_saveas_dir_prev"] = dest_dir
                st.session_state.pop("saveas_name", None)
            new_name = st.text_input("New file name", value=default_name,
                                     key="saveas_name")
            if not new_name.endswith((".yaml", ".yml")):
                new_name = new_name + ".yaml"
            rel = f"{dest_dir}/{new_name}"
            st.caption(f"→ `{rel}`")

            # Non-blocking shape hint. A scenario spec needs base_config and
            # scenarios; a deal config needs product. Getting this wrong is
            # easy and the failure would otherwise surface much later, as a
            # KeyError in the middle of a run.
            if not err and isinstance(parsed, dict):
                if dest_dir == "scenarios" and not (
                        "scenarios" in parsed and "base_config" in parsed):
                    st.warning(
                        "This doesn't look like a scenario spec — those need "
                        "`base_config:` and `scenarios:` at the top level."
                    )
                elif dest_dir == "configs" and "product" not in parsed \
                        and "assets" not in parsed:
                    st.warning(
                        "This doesn't look like a deal config — those need "
                        "`product:` (or `assets:` for a basket)."
                    )

            exists = os.path.exists(os.path.join(ROOT, rel))
            if exists:
                st.error(f"`{rel}` already exists — pick another name.")
            if st.button(f"Create {rel}", disabled=bool(err) or exists,
                         type="primary"):
                dest = os.path.join(ROOT, rel)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                open(dest, "w").write(text)
                st.success(f"Created {rel}")
                st.rerun()
    with c3:
        if err:
            st.error("Invalid YAML — cannot save")
        elif dirty:
            st.warning("Unsaved changes")
        else:
            st.caption("In sync with disk")

    if err:
        st.code(err, language="text")


# ── 2. run ────────────────────────────────────────────────────────────────────

with tab_run:
    files = config_files()
    yaml_configs = [f for f in files if not f.startswith("scenarios/")]
    scenario_specs = [f for f in files if f.startswith("scenarios/")]

    script = st.radio(
        "Command",
        ["main.py", "scenarios/run_scenarios.py", "diagnostics/model_quality.py"],
        horizontal=True,
        captions=["Price one note", "Sweep all scenarios", "Vol surface fit check"],
    )

    cmd = [PYTHON, "-u", script]
    col_a, col_b = st.columns(2)

    if script == "main.py":
        with col_a:
            cfg = st.selectbox("--config", yaml_configs, key="run_cfg_main")
            cmd += ["--config", cfg]
        with col_b:
            fit = st.radio("Fit check", ["default", "--no-fit-check", "--full-diagnostic"],
                           key="run_fit")
            if fit != "default":
                cmd.append(fit)

    elif script == "scenarios/run_scenarios.py":
        with col_a:
            spec = st.selectbox("--scenarios", scenario_specs or ["scenarios/scenarios.yaml"],
                                key="run_spec")
            cmd += ["--scenarios", spec]
            n_paths = st.number_input("--n-paths", 1_000, 500_000, 20_000, step=1_000)
            cmd += ["--n-paths", str(int(n_paths))]
            out_csv = st.text_input("--output", "scenarios/results.csv")
            cmd += ["--output", out_csv]
        with col_b:
            verbose = st.checkbox("--verbose", value=True)
            if verbose:
                cmd.append("--verbose")
            greeks = st.checkbox("--greeks", value=False,
                                 help="Adds ~10 bumped repricings per model. Slow.")
            if greeks:
                cmd.append("--greeks")
                n_g = st.number_input("--n-paths-greeks", 1_000, 200_000, 10_000, step=1_000)
                cmd += ["--n-paths-greeks", str(int(n_g))]
                st.caption(
                    "At 12 scenarios this is a multi-minute run — "
                    "roughly 7 minutes at these settings with LSV enabled."
                )

    else:  # model_quality.py
        with col_a:
            cfg = st.selectbox("--config", yaml_configs, key="run_cfg_diag")
            cmd += ["--config", cfg]
        with col_b:
            n_paths = st.number_input("--n-paths", 1_000, 200_000, 30_000, step=1_000,
                                      key="diag_paths")
            cmd += ["--n-paths", str(int(n_paths))]

    st.markdown("**Command**")
    st.code(" ".join([os.path.basename(cmd[0])] + cmd[1:]), language="bash")

    go = st.button("Run", type="primary")
    st.caption(
        "The page blocks while the command runs — same as waiting at a terminal. "
        "Output streams live below. Ctrl+C in the terminal that launched Streamlit "
        "stops a run you want to abandon."
    )

    last = st.session_state.get("last_run")
    if last and not go:
        icon = "✅" if last["code"] == 0 else "❌"
        st.caption(f"{icon} last run {last['when']} · {last['elapsed']:.1f}s · exit {last['code']}")

    out_area = st.empty()
    if go:
        out_area.code("starting…", language="text")
        code = run_streaming(cmd, out_area)
        if code == 0:
            st.success("Finished — see the Results tab.")
        else:
            st.error(f"Exited with code {code}.")


# ── 3. results ────────────────────────────────────────────────────────────────

with tab_results:
    csvs = result_files()
    if not csvs:
        st.info("No result CSVs yet. Run a scenario sweep from the Run tab.")
        st.stop()

    pick = st.selectbox("Result file", csvs, key="res_file")
    df = pd.read_csv(os.path.join(ROOT, pick))
    models = detect_models(df)

    mtime = datetime.fromtimestamp(os.path.getmtime(os.path.join(ROOT, pick)))
    st.caption(
        f"{len(df)} scenarios · models: {', '.join(models) or 'none detected'} · "
        f"modified {mtime:%Y-%m-%d %H:%M}"
    )
    if models and "LSV" not in models:
        st.info(
            "This file predates the LSV model. It also predates the Dupire "
            "correction, so its prices are a few bp below what the current code "
            "produces — see USER_MANUAL §12."
        )

    if not models:
        st.dataframe(df, width="stretch")
        st.stop()

    # -- prices ---------------------------------------------------------------
    st.subheader("Prices")
    price_df = df[["scenario"] + models].set_index("scenario")
    st.dataframe(
        price_df.style.format("{:.6f}").background_gradient(axis=1, cmap="Blues"),
        width="stretch",
    )

    if alt is not None:
        long = price_df.reset_index().melt("scenario", var_name="model", value_name="price")
        domain, rng = model_colors(models)
        order = list(price_df.index)
        base = alt.Chart(long)

        # A DOT plot, not bars. Every price here sits between roughly 1.02 and
        # 1.13, and the thing worth seeing is the gap BETWEEN models. Bars
        # cannot show that: a bar encodes length, so it needs a zero baseline
        # to be honest, and against a 0-1.2 axis a 1% spread is invisible.
        # (Vega-Lite enforces this -- it overrides scale zero=False on a bar
        # mark, precisely so nobody ships a truncated bar.)
        #
        # A dot encodes position rather than length, carries no baseline claim,
        # and so the axis can legitimately zoom to the data range.
        #
        # Each model gets its own offset line within the scenario band. Without
        # that the dots overlap wherever the models agree closely, and a hidden
        # dot is indistinguishable from an absent model.
        dots = base.mark_circle(
            size=95, opacity=1.0,
            stroke="#1a1a19" if _is_dark() else "#fcfcfb", strokeWidth=1.5,
        ).encode(
            y=alt.Y("scenario:N", sort=order, title=None,
                    axis=alt.Axis(labelLimit=240)),
            yOffset=alt.YOffset("model:N", sort=models),
            x=alt.X("price:Q", scale=alt.Scale(zero=False, nice=True),
                    title="price (per unit notional)"),
            color=alt.Color("model:N", sort=models, title="model",
                            scale=alt.Scale(domain=domain, range=rng)),
            tooltip=["scenario", "model",
                     alt.Tooltip("price:Q", title="price", format=".6f")],
        )
        st.altair_chart(
            dots.properties(height=max(300, 40 * len(price_df))),
            width="stretch",
        )
        st.caption(
            "Axis is zoomed to the data range — legitimate for dots, which encode "
            "position rather than length. Spread between models is quantified below."
        )

    # -- model spread ---------------------------------------------------------
    if "spread_bp" in df.columns:
        st.subheader("Model spread")
        st.caption("Max − min price across models, in bp of notional. This is the model-risk number.")
        spread = df[["scenario", "spread_bp"]].set_index("scenario")
        st.bar_chart(spread, height=260)

    # -- greeks ---------------------------------------------------------------
    greek_names = ["duration", "delta", "gamma", "vega", "vanna", "skew_sens",
                   "rho", "div_sens"]
    present = [g for g in greek_names
               if all(f"{m}_{g}" in df.columns for m in models)]
    if present:
        st.subheader("Greeks")
        chosen_g = st.selectbox("Measure", present, key="res_greek")
        cols = [f"{m}_{chosen_g}" for m in models]
        g_df = df[["scenario"] + cols].copy()
        g_df.columns = ["scenario"] + models

        # Table first. It is the accessible twin of the chart, and it also
        # discharges the light-mode contrast warning on the third colour slot.
        st.dataframe(g_df.set_index("scenario").style.format("{:+.6f}"),
                     width="stretch")

        if alt is not None:
            long = g_df.melt("scenario", var_name="model", value_name="value")
            domain, rng = model_colors(models)
            base = alt.Chart(long)

            # GROUPED, never stacked. A delta under Local Vol and a delta under
            # Heston are competing answers to the same question, not components
            # of one quantity -- their sum has no meaning, so a stacked height
            # would draw a number that does not exist. st.bar_chart() stacks by
            # default, which is exactly the bug this replaces.
            #
            # Horizontal because the scenario labels are long; vertical would
            # need rotated text and a taller axis band. Zero-based x, because
            # for a Greek zero is a real value (no sensitivity) and the sign
            # carries meaning -- several of these measures go negative.
            bars = base.mark_bar().encode(
                y=alt.Y("scenario:N", title=None,
                        sort=list(g_df["scenario"]),
                        axis=alt.Axis(labelLimit=240)),
                yOffset=alt.YOffset("model:N", sort=models),
                x=alt.X("value:Q", title=chosen_g,
                        scale=alt.Scale(zero=True)),
                color=alt.Color("model:N", sort=models, title="model",
                                scale=alt.Scale(domain=domain, range=rng)),
                tooltip=["scenario", "model",
                         alt.Tooltip("value:Q", title=chosen_g, format="+.6f")],
            )
            zero = base.mark_rule(strokeWidth=1, opacity=0.45).encode(
                x=alt.datum(0))
            st.altair_chart(
                (bars + zero).properties(height=max(280, 34 * len(g_df))),
                width="stretch")
        else:
            # stack=False matters here for the same reason as above.
            st.bar_chart(g_df.set_index("scenario"), height=300, stack=False)
    else:
        st.caption("No Greeks in this file — re-run with `--greeks` to populate them.")

    with st.expander("Raw CSV"):
        st.dataframe(df, width="stretch")
        st.download_button("Download", df.to_csv(index=False),
                           file_name=os.path.basename(pick), mime="text/csv")


# ── 4. manual ─────────────────────────────────────────────────────────────────

with tab_manual:
    sections = manual_sections()
    if not sections:
        st.error(f"USER_MANUAL.md not found at `{MANUAL_PATH}`.")
    else:
        titles = [h for h, _ in sections]
        query = st.text_input(
            "Search the manual", placeholder="e.g. sticky leverage, rho, antithetic",
            key="manual_q",
        )

        if query.strip():
            # Match on a normalised form so a natural-language query finds the
            # code-style spelling: "sticky leverage" should hit
            # `sticky_leverage`, and "h rate" should hit `--h-rate`. Underscores
            # and hyphens fold to spaces on BOTH sides, and runs of whitespace
            # collapse. The original text is still what gets rendered.
            def norm(t: str) -> str:
                return " ".join(re.sub(r"[_\-]+", " ", t.lower()).split())

            q = norm(query)
            hits = [(h, b, norm(b).count(q)) for h, b in sections if q in norm(b)]
            if not hits:
                st.info(f"No match for “{query}”.")
            else:
                st.caption(
                    f"{sum(n for _, _, n in hits)} match(es) in "
                    f"{len(hits)} section(s) — most matches first"
                )
                for h, body, n in sorted(hits, key=lambda x: -x[2]):
                    with st.expander(f"{h}  ·  {n} match(es)"):
                        st.markdown(body)
        else:
            col_nav, col_body = st.columns([1, 3])
            with col_nav:
                chosen_sec = st.radio(
                    "Section", titles,
                    index=titles.index("1. Overview") if "1. Overview" in titles else 0,
                    label_visibility="collapsed",
                    key="manual_sec",
                )
            with col_body:
                st.markdown(dict(sections)[chosen_sec])

        st.divider()
        st.caption(
            "Rendered from `USER_MANUAL.md` in the repo, so it is never out of "
            "step with the code. Cross-references like “see §12” are section "
            "numbers in the list on the left — Streamlit has no in-page anchors, "
            "so the document's own links are inert here."
        )
        st.download_button(
            "Download USER_MANUAL.md",
            open(MANUAL_PATH, encoding="utf-8").read(),
            file_name="USER_MANUAL.md", mime="text/markdown",
        )
