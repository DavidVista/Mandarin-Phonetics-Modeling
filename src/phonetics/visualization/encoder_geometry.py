"""Encoder geometry presentation using the reference geometry interface."""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np

from ..evaluation.reference_geometry import TRANSITIONS
from ..objectives import parse_model_id
from .reference_geometry import _encode
from .reference_geometry_template import PAGE

def _system_name(system):
    if system == "DIMSIM":
        return system
    family, weight = parse_model_id(system)
    return f"{family.removesuffix('-acoustic').capitalize()} + Acoustic (λ={weight:g})" if weight is not None else family.capitalize()


def _json(value):
    return json.dumps(value, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")


def _encode_models(values, dtype):
    return [_encode(row, dtype) for row in values]


def _view_profiles(tables, across, systems):
    def get(row, key):
        return row[key + "_mean" if across else key]
    profiles, residuals = [], []
    for system in systems:
        label = _system_name(system)
        means = [get(row, "mean_similarity") for row in tables["mask_profiles"] if row["system"] == system]
        r2 = [get(row, "r_squared") for row in tables["mask_summary"] if row["system"] == system]
        steps = []
        for row, final in zip((row for row in tables["full_match_gains"] if row["system"] == system),
                              ("011 → 111", "101 → 111", "110 → 111")):
            steps.append({
                "labels": [row[f"step_{i}"].replace(" to ", " → ") for i in range(1, 4)] + [final],
                "values": [get(row, f"gain_{i}") for i in range(1, 4)] + [get(row, "full_match_gain")],
                "baseline": get(row, "ordinary_mean"), "absolute": get(row, "j"),
            })
        profiles.append({"name": label, "beta": means, "r2": r2, "steps": steps})
        residuals.append([{"a": r["left"], "b": r["right"], "mask": r["mask"], "residual": r["residual"]}
                          for r in tables.get("residuals_top50", ()) if r["system"] == system])
    return profiles, residuals


def rank_metrics(distances, neighborhood, tolerance):
    n = distances.shape[1]
    tail, top = (n - 1) // 10, min(10, n - 1)
    overlaps, near_far = [], []
    for matrix in distances:
        ranks = np.argsort(matrix, axis=1, kind="stable")
        overlap = wins = 0
        for anchor, order in enumerate(ranks):
            ref = neighborhood[anchor][neighborhood[anchor] != anchor]
            candidates = order[order != anchor]
            overlap += len(set(ref[:top]) & set(candidates[:top]))
            if tail:
                near = matrix[anchor, ref[:tail]]
                far = np.sort(matrix[anchor, ref[-tail:]])
                low = np.searchsorted(far, near - tolerance, side="left")
                high = np.searchsorted(far, near + tolerance, side="right")
                wins += np.sum(tail - high + .5 * (high - low))
        overlaps.append(overlap / (n * top))
        near_far.append(float(wins / (n * tail * tail)) if tail else None)
    return overlaps, near_far


def write_view(path, key, distances, maps, tables, tolerance, *, systems, seeds, rankings=None):
    order = maps["component_order"]
    if 2 * len(seeds) * (len(order) - 1) > np.iinfo(np.uint16).max:
        raise ValueError("encoder view exceeds its 16-bit comparison credit encoding")
    inverse = np.argsort(order)
    ordered = distances[:, order][:, :, order]
    cells = maps["controlled_cells"].copy()
    cells[:, 1:3] = inverse[cells[:, 1:3]]
    change = np.bitwise_xor.reduce(TRANSITIONS, axis=1)[cells[:, 0]]
    cells[:, 0] = (change & 2) | ((change & 1) << 2) | ((change & 4) >> 2)
    values = maps["controlled_values"]
    support = maps["agreement_counts"][:, order][:, :, order]
    projections = maps["agreement"][:, :, order][:, :, :, order]
    across = key == "across-seeds"
    factor = len(seeds) if across else 1
    points = np.rint(np.nan_to_num(projections) * support[:, None] * 2 * factor)
    ties = {row["slice"]: row["dimsim_ties_mean" if across else "dimsim_ties"]
            for row in tables["dimsim_agreement"] if row["system"] == "DIMSIM"}
    profiles, residuals = _view_profiles(tables, across, systems)
    overlap, near_far = rankings if rankings is not None else rank_metrics(distances, maps["neighborhood"], tolerance)
    keys = ("controlled_match", "dimsim_agreement", "same_mask_agreement", "pairwise_spearman",
            "paired_differences", "mask_summary", "full_match_gains")
    view = {
        "distance": _encode_models(ordered, "<f4"),
        "maxima": [float(np.nanmax(ordered[0])), float(np.nanmax(ordered[1:]))],
        "rank": _encode(inverse[maps["neighborhood"][order]], "<u2"),
        "rankStats": [overlap, [row["spearman_mean" if across else "spearman"]
                                for row in tables["pairwise_spearman"] if row["system"] in systems], near_far],
        "rankGroupSize": (len(order) - 1) // 10,
        "prepared": {
            "factor": factor, "cells": _encode(cells, "<u2"),
            "effects": _encode_models(values[:, :, 0].T, "<f8"),
            "wins": _encode_models(np.rint(values[:, :, 1] * cells[:, 3, None] * 2 * factor).T, "<u2"),
            "agree": _encode_models(np.rint(np.nan_to_num(values[:, :, 2]) * cells[:, 4, None] * 2 * factor).T, "<u2"),
            "support": _encode_models(support, "<u2"),
            "points": [_encode_models(group, "<u2") for group in points],
            "ties": [ties["Trade-off"], ties["Same mask"]],
        },
        "profiles": profiles, "residuals": residuals,
        "tables": {key: tables[key] for key in keys},
    }
    Path(path).write_text(f"window.encoderViews[{_json(key)}]={_json(view)};\n", encoding="utf-8")


def _encoder_template(models):
    page = PAGE
    replacements = {
        "wav2vec2 zh-cn": "@@left_name@@",
        "wav2vec2 base": "@@right_name@@",
        "The two wav2vec2 panels share a cosine distance scale.": "The encoder panels share a cosine distance scale.",
        "cosine distances from matching audio for both wav2vec2 systems":
            "character-balanced cosine distances from contextual encoder representations",
        "wav2vec2 ties": "Encoder ties",
        "<p>The exploratory protocol was developed while inspecting these wav2vec2 systems. ":
            "<p>",
        "Other cells, including self-pairs, are shown in light gray.":
            "Unselected cells and the diagonal in this view are shown in light gray.",
        "Self-distances are zero for the current matrices.": "Contextual full match excludes self-comparisons. Grey encoder diagonals lack two distinct occurrences in at least one seed. The DIMSIM matrix shows its zero self-distances.",
        "Mask 111 uses self-pairs, with cosine 1 and DIMSIM distance 0.": "Mask 111 uses contextual within-reading distances and the corresponding zero DIMSIM distances on the shared Full Match inventory.",
        "Mask means, centered R², J and residuals describe each system independently.":
            "Mask means, centered R² and J describe each system independently.",
        "Including self-pairs (111)": "Including Full Match (111)",
        "</style>": ".panels > section:first-child{order:1}.panels > section:nth-child(3){order:2}</style>",
        "const preparedCells=decode(prepared.cells,Uint16Array), preparedEffects=decode(prepared.effects,Float64Array);":
            "const preparedCells=decode(prepared.cells,Uint16Array), preparedEffects=prepared.effects.map(x=>decode(x,Float64Array));",
        "const preparedWins=decode(prepared.wins,Uint16Array), preparedAgreement=decode(prepared.agree,Uint16Array);":
            "const preparedWins=prepared.wins.map(x=>decode(x,Uint16Array)), preparedAgreement=prepared.agree.map(x=>decode(x,Uint16Array));",
        "preparedEffects[i*3+k]": "preparedEffects[k][i]",
        "preparedWins[i*3+k]/2": "preparedWins[k][i]/(2*prepared.factor)",
        "preparedAgreement[i*2+k]": "preparedAgreement[k][i]/prepared.factor",
        "support=new Uint16Array(N*N);points=[new Uint16Array(N*N),new Uint16Array(N*N)]":
            "support=new Uint32Array(N*N);points=[new Float64Array(N*N),new Float64Array(N*N)]",
        "prepared.points[index].map(value=>decode(value,Uint16Array))":
            "prepared.points[index].map(value=>Float64Array.from(decode(value,Uint16Array),v=>v/prepared.factor))",
    }
    for before, after in replacements.items():
        page = page.replace(before, after)
    page = style_encoder_maps(page)
    start = page.index("<h2>Top positive residual pairs</h2>")
    end = page.index("</section>", start)
    page = page[:start] + page[end:]
    page = page.replace("const residualTopPairs=@@residualTopPairs@@;\n", "")
    start = page.index(" document.querySelector('#profile-residuals').innerHTML")
    end = page.index(" renderProfileSteps();", start)
    page = page[:start] + page[end:]
    start = page.index("<a href='figures/dimsim-distances.png'>")
    end = page.index("</p>", start)
    links = " · ".join(f"<a href='figures/{model}-cosine-distances.png'>{name} cosine distances</a>"
                       for model, name in ((model, _system_name(model)) for model in models))
    page = page[:start] + links + page[end:]
    return page


def style_encoder_maps(page):
    """Give the embedded analysis its own heading and a lighter visual hierarchy."""
    start = page.index('<header class="report-head">')
    end = page.index('</header>', start) + len('</header>')
    page = page[:start] + (
        '<header class="report-head"><h2>Mandarin syllable distance maps</h2>'
        '<p>@@syllables@@ syllables · @@pairs@@ unique pairs</p></header>'
    ) + page[end:]
    page = page.replace('<title>Encoder geometry</title>', '<title>Mandarin syllable distance maps</title>')
    page = page.replace('<th>Population</th><th>Count</th>',
        '<th scope="col">Model and seed</th><th scope="col">Retained occurrences</th>'
        '<th scope="col">Target UNK before filtering</th><th scope="col">Valid / planned Full Matches</th>')
    return page.replace('</style>', """
body{padding:16px 18px 32px;background:#fff}
.report-head{padding:0 0 10px;border:0;border-bottom:1px solid var(--line);border-radius:0}
.report-head h2{margin:0;font-size:22px;line-height:1.25}
.report-head p{margin:4px 0 0;color:var(--muted);font-variant-numeric:tabular-nums}
.toolbar{padding:6px 0;border:0;border-radius:0}
.panels>section,#map-results>section,#profile-results,details{border-radius:5px}
@media(max-width:700px){body{padding:12px}}
</style>""", 1)


_SHELL = r"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>@@ARCHITECTURE@@ evaluation report</title>
<style>
*{box-sizing:border-box}
body{margin:0 auto;max-width:1560px;padding:0 clamp(12px,3vw,40px) 28px;background:#fbfafc;color:#292330;font:15px/1.5 system-ui,sans-serif}
header{padding:24px 0 16px;border-bottom:1px solid #ded7e6}
.report-context{margin:0;color:#5e5268;font-weight:600}
header h1{margin:2px 0 14px;font-size:clamp(27px,3.4vw,39px);line-height:1.12;letter-spacing:-.025em}
.selectors{display:flex;flex-wrap:wrap;gap:8px 20px}
label{display:flex;align-items:center;gap:8px;font-weight:600}
select{min-width:0;max-width:min(100%,34ch);font:inherit;padding:5px 8px;border:1px solid #c8bdcf;border-radius:5px;background:white;color:inherit}
a{color:#563671;text-underline-offset:3px}
:focus-visible{outline:2px solid #79549a;outline-offset:2px}
#status,.report-note{margin:9px 0;color:#5e5268;font-size:14px}
iframe{border:1px solid #ded7e6;border-radius:6px;width:100%;height:78vh;min-height:680px;display:block;margin:14px 0;background:#fff}
details{background:#fff;border:1px solid #ded7e6;border-radius:6px;padding:12px 16px}
summary{cursor:pointer;font-weight:650}
table{border-collapse:collapse;font-variant-numeric:tabular-nums}
th,td{padding:6px 10px;border-bottom:1px solid #e7e1eb;text-align:left}
th{background:#f4f0f7}
@media(max-width:520px){.selectors label{width:100%;justify-content:space-between}select{max-width:70%}}
</style>
<header><p class="report-context">@@ARCHITECTURE@@ evaluation report</p><h1>Encoder Geometry</h1>
<div class="selectors"><label>Seed <select id="seed"></select></label>
<label>Left model <select id="left"></select></label><label>Right model <select id="right"></select></label></div>
<p id="status" aria-live="polite"></p><a href="tables/coverage.csv">Coverage and UNK counts</a> · <a href="manifest.json">Manifest</a></header>
@@DIAGNOSTICS@@
<iframe id="view" title="Synchronized encoder distance maps"></iframe>
<details><summary>Selected metrics and seed dispersion</summary><div id="summary"></div></details>
<p class="report-note">DIMSIM Agreement excludes DIMSIM ties and gives model ties half credit. Controlled match win rate
uses the direction predicted by component matching. Both rates have a neutral value of 50%.
Paired differences are the named destination system minus the source system within each seed.</p>
<script>
const settings=@@SETTINGS@@, template=@@TEMPLATE@@;
window.encoderViews={};
const $=id=>document.getElementById(id);
for(const id of ['left','right'])for(let i=1;i<settings.names.length;i++)$(id).add(new Option(settings.names[i],i));
$('right').selectedIndex=Math.min(1,settings.names.length-2);
$('seed').add(new Option('Across seeds','across-seeds'));
for(const seed of settings.seeds)$('seed').add(new Option('Seed '+seed,'seed-'+seed));
let revision=0;
function load(key){
 if(window.encoderViews[key])return Promise.resolve(window.encoderViews[key]);
 return new Promise((resolve,reject)=>{
  const script=document.createElement('script');script.src='views/'+key+'.js';
  script.onload=()=>{script.remove();resolve(window.encoderViews[key]);};
  script.onerror=()=>{script.remove();reject(Error('Cannot load '+key));};
  document.head.append(script);
 });
}
const escapeHTML=value=>String(value??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function selectedTables(view, models){
 const names=new Set(models.map(i=>settings.systems[i]));
 const display=value=>escapeHTML(typeof value==='number'?Number(value.toPrecision(6)):value);
 return Object.entries(view.tables).map(([name, rows])=>{
  const selected=rows.filter(row=>names.has(row.system)||row.system==='Component mismatch count'||
    (names.has(row.from_system)&&names.has(row.to_system)));
  if(!selected.length)return '';
  const keys=Object.keys(selected[0]);
  return '<h3>'+escapeHTML(name.replaceAll('_',' '))+'</h3><div style="overflow:auto"><table><tr>'+
   keys.map(key=>'<th>'+escapeHTML(key.replaceAll('_',' '))+'</th>').join('')+'</tr>'+
   selected.map(row=>'<tr>'+keys.map(key=>'<td>'+display(row[key])+'</td>').join('')+'</tr>').join('')+'</table></div>';
 }).join('');
}
async function render(){
 const key=$('seed').value, l=+$('left').value,r=+$('right').value,token=++revision;
 if(l===r&&settings.names.length>2){$('status').textContent='Choose two different encoder models.';return;}
 $('status').textContent='Loading selected view…';
 try{
  const view=await load(key);
  if(token!==revision)return;
  const systems=[0,l,r], p=view.prepared, select=items=>systems.map(i=>items[i]);
  const data={labels:settings.labels,rank:view.rank,maps:select(view.distance),maxima:view.maxima};
  const prepared={factor:p.factor,cells:p.cells,effects:select(p.effects),wins:select(p.wins),agree:[p.agree[l],p.agree[r]],
                  support:p.support,points:p.points.map(row=>[row[l],row[r]]),ties:p.ties};
  const values={data,prepared,components:settings.components,profileData:select(view.profiles),
   rankStats:view.rankStats.map(select),rankGroupSize:view.rankGroupSize};
  const escapedLessThan=String.fromCharCode(92)+'u003c';
  const substitutions=Object.fromEntries(Object.entries(values).map(([name,value])=>[name,JSON.stringify(value).replaceAll('<',escapedLessThan)]));
  Object.assign(substitutions,{left_name:settings.names[l],right_name:settings.names[r],
    syllables:settings.labels.length.toLocaleString('en-US'),
    pairs:(settings.labels.length*(settings.labels.length-1)/2).toLocaleString('en-US'),
    evaluation_id:settings.evaluation,excluded:settings.excluded,tolerance:settings.tolerance,coverage:settings.coverage});
  $('view').srcdoc=template.replace(/@@([A-Za-z_]+)@@/g,(_,name)=>substitutions[name]);
  const across=key==='across-seeds';
  $('status').textContent=across?'Maps show mean distances and seedwise comparison scores. Tables summarize metrics computed separately by seed.':'Showing distances and metrics for '+$('seed').selectedOptions[0].text+'.';
  $('summary').innerHTML=selectedTables(view,systems);
  for(const loaded of Object.keys(window.encoderViews))if(loaded!==key)delete window.encoderViews[loaded];
 }catch(error){if(token===revision)$('status').textContent=error.message;}
}
for(const id of ['seed','left','right'])$(id).addEventListener('change',render);
render();
</script></html>"""


def write_page(path, labels, components, coverage, *, architecture, evaluation, excluded, tolerance, systems, seeds,
               diagnostics=False):
    order = sorted(range(len(labels)), key=lambda index: (tuple(components[index]), labels[index]))
    rows = "".join(
        f"<tr><th>{html.escape(row['system'])} · Seed {row['seed']}</th>"
        f"<td>{row['occurrences']} retained occurrences</td>"
        f"<td>{row['unknown_fraction']:.2%} target UNK before filtering</td>"
        f"<td>{row['valid_full_matches']} / {row['planned_full_matches']} full matches</td></tr>"
        for row in coverage
    )
    settings = {"labels": [labels[i] for i in order], "components": np.asarray(components)[order].T.tolist(),
                "names": [_system_name(system) for system in systems], "systems": systems, "seeds": seeds, "coverage": rows,
                "evaluation": html.escape(evaluation), "excluded": len(excluded), "tolerance": f"{tolerance:g}"}
    architecture_name = {"char_bilstm": "BiLSTM", "bert": "BERT"}[architecture]
    page = (_SHELL.replace("@@ARCHITECTURE@@", architecture_name)
            .replace("@@SETTINGS@@", _json(settings)).replace("@@TEMPLATE@@", _json(_encoder_template(systems[1:]))))
    links = ""
    if diagnostics:
        from ..research.encoder_diagnostics import TABLE_FIELDS
        links = '<p class="report-note">Diagnostics: ' + " · ".join(
            f'<a href="tables/diagnostic_{name}.csv">{name.replace("_", " ").capitalize()}</a>'
            for name in TABLE_FIELDS
        ) + ' · <a href="data/diagnostics.json">Inventories and provenance</a></p>'
    Path(path).write_text(page.replace("@@DIAGNOSTICS@@", links), encoding="utf-8")
