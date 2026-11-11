"""Encoder geometry presentation using the reference geometry interface."""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np

from ..evaluation.reference_geometry import TRANSITIONS
from ..evaluation.training_summary import TRAINING_MODELS, TRAINING_SEEDS
from .reference_geometry import _encode
from .reference_geometry_template import PAGE

NAMES = ("DIMSIM", "Syllable", "Components", "Acoustic", "Components + Acoustic (λ=1)", "Components + Acoustic (λ=10)")
SYSTEMS = ("DIMSIM", *TRAINING_MODELS)


def _json(value):
    return json.dumps(value, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")


def _encode_models(values, dtype):
    return [_encode(row, dtype) for row in values]


def _view_profiles(tables, across):
    def get(row, key):
        return row[key + "_mean" if across else key]
    profiles, residuals = [], []
    for system, label in zip(SYSTEMS, NAMES):
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


def write_view(path, key, distances, maps, tables, tolerance, *, rankings=None):
    order = maps["component_order"]
    if 2 * len(TRAINING_SEEDS) * (len(order) - 1) > np.iinfo(np.uint16).max:
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
    factor = len(TRAINING_SEEDS) if across else 1
    points = np.rint(np.nan_to_num(projections) * support[:, None] * 2 * factor)
    ties = {row["slice"]: row["dimsim_ties_mean" if across else "dimsim_ties"]
            for row in tables["dimsim_agreement"] if row["system"] == "DIMSIM"}
    profiles, residuals = _view_profiles(tables, across)
    overlap, near_far = rankings if rankings is not None else rank_metrics(distances, maps["neighborhood"], tolerance)
    keys = ("controlled_match", "dimsim_agreement", "pairwise_spearman", "mask_summary", "full_match_gains")
    view = {
        "distance": _encode_models(ordered, "<f4"),
        "maxima": [float(np.nanmax(ordered[0])), float(np.nanmax(ordered[1:]))],
        "rank": _encode(inverse[maps["neighborhood"][order]], "<u2"),
        "rankStats": [overlap, [row["spearman_mean" if across else "spearman"] for row in tables["pairwise_spearman"]], near_far],
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


def _encoder_template():
    page = PAGE
    replacements = {
        "REFERENCE GEOMETRY": "ENCODER GEOMETRY",
        "Mandarin syllable distance maps": "Encoder geometry",
        "wav2vec2 zh-cn": "@@right_name@@",
        "wav2vec2 base": "@@left_name@@",
        "systems=['DIMSIM','zh-cn','base']": "systems=['DIMSIM','@@right_name@@','@@left_name@@']",
        "zh-cn "+chr(39): "@@right_name@@ "+chr(39),
        "base "+chr(39): "@@left_name@@ "+chr(39),
        "the two wav2vec2": "the two encoder",
        "The two wav2vec2": "The two encoder",
        "both wav2vec2 systems": "the contextual encoders",
        "cosine distances from matching audio for the contextual encoders": "weighted cosine distances from the contextual encoders",
        "wav2vec2 ties": "Encoder ties",
        "these wav2vec2 systems": "the acoustic reference systems",
        "Self-distances are zero for the current matrices.": "Contextual full match excludes self-comparisons. Grey diagonals lack two distinct occurrences.",
        "Mask 111 uses self-pairs, with cosine 1 and DIMSIM distance 0.": "Mask 111 uses contextual within-reading distances and the zero DIMSIM diagonal.",
        "Mask means, centered R², J and residuals describe each system independently.": "Mask means, centered R², J and residuals describe each system independently. Residuals are available for individual seeds.",
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
        "const match=!active.length||mask[row*N+j];": "const match=(!active.length||mask[row*N+j])&&Number.isFinite(mat[k][row*N+j]);",
    }
    for before, after in replacements.items():
        page = page.replace(before, after)
    start = page.index("<a href='figures/dimsim-distances.png'>")
    end = page.index("</p>", start)
    links = " · ".join(f"<a href='figures/{model}-cosine-distances.png'>{name} cosine distances</a>"
                       for model, name in zip(TRAINING_MODELS, NAMES[1:]))
    page = page[:start] + links + page[end:]
    return page


_SHELL = r"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Encoder geometry</title>
<style>
body{margin:0 auto;max-width:1740px;background:#f5f7f9;color:#182631;font:15px Arial,sans-serif}
header,details{padding:16px 24px;background:white;border:1px solid #dbe2e8;border-radius:12px;margin:12px}
label{display:inline-block;margin:5px 16px 5px 0}select{font:inherit;padding:7px;border:1px solid #cbd7e0;border-radius:7px;background:#f8fafc}
iframe{border:0;width:100%;height:78vh;min-height:680px;display:block}summary{cursor:pointer;font-weight:bold}
table{border-collapse:collapse}th,td{padding:6px 10px;border-bottom:1px solid #dbe2e8;text-align:left}th{background:#edf3f8}
</style>
<header><h1>Encoder geometry</h1><label>Seed <select id="seed"></select></label>
<label>Left model <select id="left"></select></label><label>Right model <select id="right"></select></label>
<p id="status" aria-live="polite"></p><a href="tables/coverage.csv">Coverage and UNK counts</a> · <a href="manifest.json">Manifest</a></header>
<iframe id="view" title="Synchronized encoder distance maps"></iframe>
<details><summary>Selected metrics and seed dispersion</summary><div id="summary"></div></details>
<script>
const settings=@@SETTINGS@@, template=@@TEMPLATE@@;
window.encoderViews={};
const $=id=>document.getElementById(id);
for(const id of ['left','right'])for(let i=1;i<settings.names.length;i++)$(id).add(new Option(settings.names[i],i));
$('right').selectedIndex=1;
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
function selectedTables(view, models, across){
 const names=new Set(models.map(i=>settings.systems[i]));
 const display=value=>escapeHTML(typeof value==='number'?Number(value.toPrecision(6)):value);
 return Object.entries(view.tables).map(([name, rows])=>{
  const selected=rows.filter(row=>names.has(row.system));
  if(!selected.length)return '';
  const keys=Object.keys(selected[0]);
  return '<h3>'+escapeHTML(name.replaceAll('_',' '))+'</h3><div style="overflow:auto"><table><tr>'+
   keys.map(key=>'<th>'+escapeHTML(key.replaceAll('_',' '))+'</th>').join('')+'</tr>'+
   selected.map(row=>'<tr>'+keys.map(key=>'<td>'+display(row[key])+'</td>').join('')+'</tr>').join('')+'</table></div>';
 }).join('');
}
async function render(){
 const key=$('seed').value, l=+$('left').value,r=+$('right').value,token=++revision;
 if(l===r){$('status').textContent='Choose two different encoder models.';return;}
 $('status').textContent='Loading selected view…';
 try{
  const view=await load(key);
  if(token!==revision)return;
  const systems=[0,r,l], p=view.prepared, select=items=>systems.map(i=>items[i]);
  const data={labels:settings.labels,rank:view.rank,maps:select(view.distance),maxima:view.maxima};
  const prepared={factor:p.factor,cells:p.cells,effects:select(p.effects),wins:select(p.wins),agree:[p.agree[r],p.agree[l]],
                  support:p.support,points:p.points.map(row=>[row[r],row[l]]),ties:p.ties};
  const values={data,prepared,components:settings.components,profileData:select(view.profiles),
   residualTopPairs:select(view.residuals),rankStats:view.rankStats.map(select),rankGroupSize:view.rankGroupSize};
  const escapedLessThan=String.fromCharCode(92)+'u003c';
  const substitutions=Object.fromEntries(Object.entries(values).map(([name,value])=>[name,JSON.stringify(value).replaceAll('<',escapedLessThan)]));
  Object.assign(substitutions,{left_name:settings.names[l],right_name:settings.names[r],
    syllables:settings.labels.length.toLocaleString('en-US'),
    pairs:(settings.labels.length*(settings.labels.length-1)/2).toLocaleString('en-US'),
    evaluation_id:settings.evaluation,excluded:settings.excluded,tolerance:settings.tolerance,coverage:settings.coverage});
  $('view').srcdoc=template.replace(/@@([A-Za-z_]+)@@/g,(_,name)=>substitutions[name]);
  const across=key==='across-seeds';
  $('status').textContent=across?'Maps show mean distances and seedwise comparison scores. Tables summarize metrics computed separately by seed.':'Showing distances and metrics for '+$('seed').selectedOptions[0].text+'.';
  $('summary').innerHTML=selectedTables(view,systems,across);
  for(const loaded of Object.keys(window.encoderViews))if(loaded!==key)delete window.encoderViews[loaded];
 }catch(error){if(token===revision)$('status').textContent=error.message;}
}
for(const id of ['seed','left','right'])$(id).addEventListener('change',render);
render();
</script></html>"""


def write_page(path, labels, components, coverage, *, evaluation, excluded, tolerance):
    order = sorted(range(len(labels)), key=lambda index: (tuple(components[index]), labels[index]))
    rows = "".join(
        f"<tr><th>{html.escape(row['system'])} · Seed {row['seed']}</th>"
        f"<td>{row['occurrences']} occurrences</td><td>{row['unknown_fraction']:.2%} UNK</td>"
        f"<td>{row['valid_full_matches']} / {row['planned_full_matches']} full matches</td></tr>"
        for row in coverage
    )
    settings = {"labels": [labels[i] for i in order], "components": np.asarray(components)[order].T.tolist(),
                "names": NAMES, "systems": SYSTEMS, "seeds": TRAINING_SEEDS, "coverage": rows,
                "evaluation": html.escape(evaluation), "excluded": len(excluded), "tolerance": f"{tolerance:g}"}
    page = _SHELL.replace("@@SETTINGS@@", _json(settings)).replace("@@TEMPLATE@@", _json(_encoder_template()))
    Path(path).write_text(page, encoding="utf-8")
