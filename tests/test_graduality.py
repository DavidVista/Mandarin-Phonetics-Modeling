from __future__ import annotations
import json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parents[1];sys.path.insert(0,str(ROOT/"src"))
from phonetics.evaluation.anchor import weighted_anchor_score
from phonetics.evaluation.graduality import build_analysis_arrays,weighted_mask_means,run_stimulus_bootstrap,prepare_anchor_cache,EDGE_SPECS,q_scores_for_weights,gap_rows,summarize_bootstrap,MASK_ORDER
from phonetics.evaluation.graduality_io import normalized_component,prepare_population
from phonetics.visualization.graduality import render_graduality

class GradualityTests(unittest.TestCase):
 def test_weighted_ties_and_missing_edge(self):
  entry={"low":np.array([0,1]),"low_scores":np.array([.2,.5]),"high":np.array([2]),"high_scores":np.array([.5])};self.assertAlmostEqual(weighted_anchor_score(entry,np.ones(3)),.75);self.assertTrue(np.isnan(weighted_anchor_score(entry,np.array([0,0,1]))))
 def test_full_pair_population_and_pair_weights(self):
  pop=[{"pinyin":"a","initial":"a","final":"x","tone":"1"},{"pinyin":"b","initial":"b","final":"x","tone":"1"},{"pinyin":"c","initial":"c","final":"y","tone":"2"}];vec={"base":{"a":[1,0],"b":[0,1],"c":[1,1]},"zh-cn":{"a":[1,0],"b":[0,1],"c":[1,-1]}};labels,components,i,j,masks,mat,cos=build_analysis_arrays(pop,vec);self.assertEqual(len(i),3);means,counts=weighted_mask_means(cos["base"],masks,i,j,np.array([2,1,0]));self.assertEqual(sum(counts.values()),1)
 def test_none_tone_and_duplicate_group_ids_rejected(self):
  self.assertRaises(ValueError,normalized_component,None,"tone","x",coerce_string=True)
  evaluation={"groups":[{"group_id":"x","group_kind":"other"},{"group_id":"x","group_kind":"other"}],"syllable_pairs":[],"manifest":{},"config":{},"coverage":{}}
  self.assertRaises(ValueError,prepare_population,evaluation,{"base":{"vectors":{}},"zh-cn":{"vectors":{}}})
 def test_parent_and_research_import_without_torch(self):
  import phonetics.evaluation.graduality,phonetics.research
  self.assertNotIn("torch",sys.modules)
 def test_cube_q_bootstrap_gaps_and_coverage_regression(self):
  population=[]
  for i in range(2):
   for f in range(2):
    for t in range(2):population.append({"pinyin":f"p{i}{f}{t}","initial":f"i{i}","final":f"f{f}","tone":str(t)})
  rng=np.random.default_rng(7);vectors={m:{row["pinyin"]:rng.normal(size=6) for row in population} for m in ("base","zh-cn")};labels,components,pi,pj,masks,matrices,cosine=build_analysis_arrays(population,vectors);cache,_=prepare_anchor_cache(components,matrices,EDGE_SPECS);weights=np.array([2,1,0,1,1,0,2,1]);actual=q_scores_for_weights(cache,weights)
  for edge in EDGE_SPECS:
   row=actual[(actual.level=="edge")&(actual.component==edge[0])&(actual.low_mask==edge[1])&(actual.high_mask==edge[2])].iloc[0]
   for model in ("base","zh-cn"):
    vals=[];aws=[]
    for anchor in range(8):
     if weights[anchor]<=0:continue
     mask=np.asarray(["".join(x) for x in np.where(components==components[anchor],"1","0")]);low=np.flatnonzero(mask==edge[1]);high=np.flatnonzero(mask==edge[2]);low=low[low!=anchor];high=high[high!=anchor]
     if not len(low) or not len(high):continue
     num=den=0.
     for lo in low:
      for hi in high:
       w=weights[lo]*weights[hi];num+=w*((matrices[model][anchor,hi]>matrices[model][anchor,lo])+.5*(matrices[model][anchor,hi]==matrices[model][anchor,lo]));den+=w
     if den:vals.append(num/den);aws.append(weights[anchor])
    self.assertAlmostEqual(row[model],np.average(vals,weights=aws))
  edges=actual[actual.level=="edge"].set_index(["component","low_mask","high_mask"]);secondary=[("tone","000","001"),("initial","000","100"),("initial","001","101"),("tone","100","101")];secondary_row=actual[actual.level=="secondary4"].iloc[0]
  for model in ("base","zh-cn"):self.assertAlmostEqual(secondary_row[model],np.mean([edges.loc[key][model] for key in secondary]))
  reps,cov=run_stimulus_bootstrap(5,20261024,labels,cache,cosine,masks,pi,pj);self.assertEqual(len(cov),5);self.assertTrue({"valid_Q_overall9",*{f"pairs_base_{m}" for m in MASK_ORDER}}<=set(cov));self.assertTrue({"Q","gap","mask_mean"}<=set(reps.family));summary=summarize_bootstrap(reps,5);self.assertTrue((summary.n_requested==5).all());means={m:weighted_mask_means(cosine[m],masks,pi,pj,np.ones(8))[0] for m in ("base","zh-cn")};gaps=gap_rows(means);self.assertEqual(set(gaps.kind),{"edge","D","D101_supplemental"});d=gaps[gaps.kind=="D"].iloc[0];self.assertAlmostEqual(d.delta,d["zh-cn"]-d["base"])
  reference_rng=np.random.default_rng(20261024)
  for repeat in range(5):
   draw_weights=np.bincount(reference_rng.integers(0,8,size=8),minlength=8);self.assertEqual(cov.iloc[repeat].unique_syllables,np.count_nonzero(draw_weights));edge_expected={}
   for edge in EDGE_SPECS:
    edge_expected[edge]={}
    for model in ("base","zh-cn"):
     vals=[];aws=[]
     for anchor in range(8):
      if draw_weights[anchor]<=0:continue
      mask=np.asarray(["".join(x) for x in np.where(components==components[anchor],"1","0")]);low=np.flatnonzero(mask==edge[1]);high=np.flatnonzero(mask==edge[2]);low=low[low!=anchor];high=high[high!=anchor];num=den=0.
      for lo in low:
       for hi in high:
        w=draw_weights[lo]*draw_weights[hi];num+=w*((matrices[model][anchor,hi]>matrices[model][anchor,lo])+.5*(matrices[model][anchor,hi]==matrices[model][anchor,lo]));den+=w
      if den:vals.append(num/den);aws.append(draw_weights[anchor])
     edge_expected[edge][model]=np.average(vals,weights=aws) if aws else np.nan
    metric=f"Q_edge:{edge[0]}:{edge[1]}->{edge[2]}";block=reps[(reps.repeat==repeat)&(reps.metric==metric)]
    for model in ("base","zh-cn"):self.assertAlmostEqual(block[block.model==model].iloc[0].value,edge_expected[edge][model])
    self.assertAlmostEqual(block[block.model=="delta_zh_minus_base"].iloc[0].value,edge_expected[edge]["zh-cn"]-edge_expected[edge]["base"])
   for level,component,chosen in [("component","initial",[e for e in EDGE_SPECS if e[0]=="initial"]),("component","final",[e for e in EDGE_SPECS if e[0]=="final"]),("component","tone",[e for e in EDGE_SPECS if e[0]=="tone"]),("overall9","all",list(EDGE_SPECS)),("secondary4","selected4",secondary)]:
    metric=f"Q_{level}:{component}";block=reps[(reps.repeat==repeat)&(reps.metric==metric)]
    for model in ("base","zh-cn"):
     expected=np.mean([edge_expected[e][model] for e in chosen]);self.assertAlmostEqual(block[block.model==model].iloc[0].value,expected)
   direct_means={}
   pair_weights=draw_weights[pi]*draw_weights[pj]
   for model in ("base","zh-cn"):
    direct_means[model]={}
    for mask in MASK_ORDER:
     selected=masks==mask;positive=pair_weights[selected]>0;direct_means[model][mask]=np.average(cosine[model][selected][positive],weights=pair_weights[selected][positive]) if positive.any() else np.nan;self.assertEqual(cov.iloc[repeat][f"pairs_{model}_{mask}"],int(positive.sum()))
   for metric,low,high in (("D","000","110"),("D101_supplemental","000","101")):
    block=reps[(reps.repeat==repeat)&(reps.metric==metric)];expected={model:direct_means[model][high]-direct_means[model][low] for model in ("base","zh-cn")}
    for model in expected:self.assertAlmostEqual(block[block.model==model].iloc[0].value,expected[model])
    self.assertAlmostEqual(block[block.model=="delta_zh_minus_base"].iloc[0].value,expected["zh-cn"]-expected["base"])
  block=reps[(reps.family=="gap")&(reps.metric=="D")&(reps.model=="delta_zh_minus_base")].value.to_numpy(float);block=block[np.isfinite(block)];row=summary[(summary.family=="gap")&(summary.metric=="D")&(summary.model=="delta_zh_minus_base")].iloc[0];self.assertAlmostEqual(row.bootstrap_median,np.median(block));self.assertAlmostEqual(row.ci025,np.quantile(block,.025));self.assertAlmostEqual(row.ci975,np.quantile(block,.975));self.assertEqual(row.n_valid,len(block))
 def test_lossless_bootstrap_plot_persistence(self):
  values=np.random.default_rng(42).uniform(.01,.2,20);values=np.r_[values,np.nan];finite=values[np.isfinite(values)];edges=np.histogram_bin_edges(finite,bins=40)
  with tempfile.TemporaryDirectory() as td:
   path=Path(td)/"plot.npz";np.savez_compressed(path,q_values=values,q_edges=edges)
   with np.load(path,allow_pickle=False) as saved:
    restored=saved["q_values"];saved_edges=saved["q_edges"];self.assertTrue(np.array_equal(restored,values,equal_nan=True));self.assertTrue(np.array_equal(saved_edges,edges));before=np.histogram(finite,bins=edges)[0];after=np.histogram(restored[np.isfinite(restored)],bins=saved_edges)[0];np.testing.assert_array_equal(after,before);self.assertEqual(int(after.sum()),20);self.assertGreaterEqual(restored[np.isfinite(restored)].min(),saved_edges[0]);self.assertLessEqual(restored[np.isfinite(restored)].max(),saved_edges[-1])
   data=Path(td)/"data";figures=Path(td)/"figures";data.mkdir();figures.mkdir();masks=np.repeat(np.asarray(MASK_ORDER),2);np.savez_compressed(data/"pairs.npz",masks=masks,base=np.linspace(-.8,.8,len(masks)),zh_cn=np.linspace(-.7,.9,len(masks)));np.savez_compressed(data/"bootstrap_plot.npz",q_values=values,d_values=-values,q_edges=edges,d_edges=np.histogram_bin_edges(-finite,bins=40));(data/"plot_spec.json").write_text(json.dumps({"mask_order":list(MASK_ORDER),"bootstrap":{"bins":40}}));files=render_graduality(data,figures,"0"*64);self.assertEqual(len(files),6);metadata=json.loads((figures/"bootstrap-differences.metadata.json").read_text());self.assertEqual([x["histogram_count"] for x in metadata["resolved_data"]["panels"]],[20,20])
if __name__=="__main__":unittest.main()
