"""Verified five-seed training-log summaries without weights or ML dependencies."""
from __future__ import annotations

import math
from pathlib import Path
import numpy as np

from .evaluation_outputs import _finite_json, _is_sha, _evaluation_run, _write, _verify_inventory
from ..research.artifacts import file_sha256

TRAINING_MODELS = ('syllable','components','acoustic','components-acoustic-lambda-1','components-acoustic-lambda-10')
TRAINING_SEEDS = (0,1,2,3,4)
METRICS_SOURCE_SHA256 = '10d41722970c5472cfe9c3f09346ae977d95dab520fc8d72a4f26d3e7f935fbc'
SELECTION_RULE = 'minimum per-seed validation objective; strict improvement keeps earliest tie'
TRAINING_CONFIG = {'seed_ids':list(TRAINING_SEEDS), 'selection':'minimum validation objective; earliest exact tie',
    'dispersion':'sample SD across training seeds; not SE or CI', 'classification_component':'mean of initial/final/tone CE',
    'acoustic_component':'unweighted 1 - cosine', 'invalid_if_scope':'components validation only',
    'bootstrap':False, 'test_used_for_selection':False, 'metrics_source_sha256':METRICS_SOURCE_SHA256}


def _training_number(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def _training_metrics(metrics, model, diagnostics, location):
    if not isinstance(metrics,dict) or not _training_number(metrics.get('objective')):
        raise ValueError(f'missing or invalid objective: {location}')
    if type(metrics.get('characters')) is not int or metrics['characters']<=0:
        raise ValueError(f'invalid character count: {location}')
    for key,value in metrics.items():
        if key.endswith(('accuracy','_rate','_ce')) or key in {'objective','cosine','weighted_cosine','component_ce'}:
            if not _training_number(value): raise ValueError(f'invalid metric {key}: {location}')
            if key.endswith(('accuracy','_rate')) and not 0<=value<=1:
                raise ValueError(f'rate outside [0,1]: {location}')
    # Logged batch sums used float32 reductions; compare definitions, not bits.
    def close(a,b): return math.isclose(a,b,rel_tol=1e-6,abs_tol=1e-8)
    components=('initials_ce','finals_ce','tones_ce')
    if all(k in metrics for k in (*components,'component_ce')):
        if not close(metrics['component_ce'],sum(metrics[k] for k in components)/3):
            raise ValueError(f'component CE definition mismatch: {location}')
    if 'invalid_if_rate' in metrics and 'invalid_if_count' in metrics:
        count=metrics['invalid_if_count']
        if type(count) is not int or not 0<=count<=metrics['characters'] or not close(metrics['invalid_if_rate'],count/metrics['characters']):
            raise ValueError(f'invalid-IF definition mismatch: {location}')
    terms = ['syllable_ce'] if model=='syllable' else ['component_ce'] if model=='components' else ['cosine'] if model=='acoustic' else ['component_ce','cosine']
    if all(k in metrics for k in terms):
        weight=10. if model.endswith('lambda-10') else 1.
        expected=metrics[terms[0]] if len(terms)==1 else metrics['component_ce']+weight*metrics['cosine']
        if not close(metrics['objective'],expected): raise ValueError(f'objective definition mismatch: {location}')
        if 'weighted_cosine' in metrics and 'cosine' in metrics and not close(metrics['weighted_cosine'],weight*metrics['cosine']):
            raise ValueError(f'weighted cosine definition mismatch: {location}')
    for key in terms:
        if key not in metrics: diagnostics.append({'location':location,'metric':key,'status':'missing'})


def _training_stats(values):
    return {'seed_values':values,'mean':float(np.mean(values)), 'sample_sd':float(np.std(values,ddof=1))}


def load_training_runs(training_runs):
    """Explicit model→{path,manifest_sha256}; only manifests/history/summary opened."""
    if not isinstance(training_runs,dict) or set(training_runs)!=set(TRAINING_MODELS):
        raise ValueError('exact five training model paths and manifest pins required')
    result={'config':dict(TRAINING_CONFIG),'models':{},'selected':[],'diagnostics':[],'inputs':[]}
    shared=None;split_counts={};selected_summary=[];metrics_fields={}
    for model in TRAINING_MODELS:
        entry=training_runs[model]
        root=Path(entry['path']).resolve(strict=True)
        sha=entry.get('manifest_sha256')
        if not _is_sha(sha) or file_sha256(root/'manifest.json')!=sha:
            raise ValueError(f'explicit training manifest hash mismatch: {model}')
        m=_finite_json(root/'manifest.json')
        if (m.get('schema_version')!=1 or m.get('run_type')!='training_run' or m.get('status')!='complete'
                or m.get('model_id')!=model or m.get('run_id')!=root.name or m.get('seeds')!=list(TRAINING_SEEDS)
                or any(type(s) is not int for s in m.get('seeds',[]))
                or m.get('seed_status')!={str(s):'complete' for s in TRAINING_SEEDS}
                or m.get('selection')!=SELECTION_RULE or m.get('code_sha256',{}).get('metrics.py')!=METRICS_SOURCE_SHA256):
            raise ValueError(f'incompatible training manifest/metric definitions: {model}')
        expected_weight=10. if model.endswith('lambda-10') else 1. if model.endswith('lambda-1') else None
        if m.get('lambda_cosine')!=expected_weight: raise ValueError('training lambda identity mismatch')
        signature=(m['experiment_id'],m['experiment_hashes'],m['config'])
        if shared is None: shared=signature
        if signature!=shared: raise ValueError('training experiments/splits/configs differ')
        epochs=m['config']['training']['epochs']
        if type(epochs) is not int or epochs<1: raise ValueError('invalid epoch count')
        if m['config']['training_seeds']!=list(TRAINING_SEEDS): raise ValueError('config seed identities differ')
        found=sorted(p.name for p in root.glob('seed-*') if p.is_dir())
        if found!=[f'seed-{s}' for s in TRAINING_SEEDS]: raise ValueError('seed directory inventory differs')
        files={}
        def read_log(name):
            expected=m['artifacts'].get(name)
            path=root/name
            if not _is_sha(expected) or file_sha256(path)!=expected: raise ValueError(f'training log hash mismatch: {model}/{name}')
            value=_finite_json(path)
            files[name]=expected
            return value
        run_summary=read_log('summary.json')
        if run_summary.get('status')!='complete' or run_summary.get('seeds_completed')!=5:
            raise ValueError('incomplete run summary')
        declared=run_summary.get('seed_results',[])
        by_seed={row['seed']:row for row in declared}
        if len(declared)!=5 or set(by_seed)!=set(TRAINING_SEEDS) or any(type(r['seed']) is not int for r in declared): raise ValueError('run-summary seed inventory differs')
        histories=[]
        selected=[]
        for seed in TRAINING_SEEDS:
            history=read_log(f'seed-{seed}/history.json')['epochs']
            summary=read_log(f'seed-{seed}/summary.json')
            if [r['epoch'] for r in history]!=list(range(1,epochs+1)) or any(type(r['epoch']) is not int for r in history):
                raise ValueError('epoch grids differ; pointwise five-seed summaries require aligned epochs')
            if summary.get('seed')!=seed or type(summary.get('seed')) is not int or summary.get('status')!='complete':
                raise ValueError('per-seed identity/status mismatch')
            if summary!=by_seed[seed]: raise ValueError('per-seed and run summaries disagree')
            for row in history:
                for split in ('train','validation'):
                    values=row[split]
                    _training_metrics(values,model,result['diagnostics'],f'{model}/seed-{seed}/epoch-{row["epoch"]}/{split}')
                    count=values['characters']
                    if split in split_counts and split_counts[split]!=count: raise ValueError('training split populations differ')
                    split_counts[split]=count
            best=min(history,key=lambda r:(r['validation']['objective'],r['epoch']))
            if (type(summary.get('best_epoch')) is not int or best['epoch']!=summary['best_epoch']
                    or best['validation']['objective']!=summary.get('best_validation_objective')
                    or best['validation']!=summary.get('best_validation_metrics')):
                raise ValueError('selected checkpoint does not minimize validation objective with earliest tie')
            test=summary.get('test_metrics',{})
            if test:
                _training_metrics(test,model,result['diagnostics'],f'{model}/seed-{seed}/test')
                if 'test' in split_counts and split_counts['test']!=test['characters']: raise ValueError('test populations differ')
                split_counts['test']=test['characters']
            else: result['diagnostics'].append({'model':model,'seed':seed,'status':'missing_test_metrics'})
            selected_row={'model_id':model,'seed':seed,'best_epoch':best['epoch'],
                          'validation_objective':best['validation']['objective'],
                          'validation':best['validation'],'test':test}
            result['selected'].append(selected_row);selected.append(selected_row);histories.append(history)
        curves={}
        for split in ('train','validation'):
            names=sorted({k for h in histories for r in h for k in r[split] if k not in {'gradient_norms'}})
            curves[split]={}
            for name in names:
                vectors=[[r[split].get(name) for r in h] for h in histories]
                if not all(_training_number(v) for vector in vectors for v in vector):
                    result['diagnostics'].append({'model':model,'split':split,'metric':name,'status':'incomplete_five_seed_curve'})
                    continue
                array=np.asarray(vectors,dtype=float)
                curves[split][name]={'mean':array.mean(axis=0).tolist(),'sample_sd':array.std(axis=0,ddof=1).tolist(),
                                      'seed_values':array.tolist()}
        result['models'][model]={'epochs':list(range(1,epochs+1)),'curves':curves,'selected':selected}
        fields={}
        for split in ('validation','test'):
            names=sorted({k for r in selected for k in r[split] if k!='gradient_norms'})
            fields[split]={k:_training_stats([r[split][k] for r in selected]) for k in names
                           if all(_training_number(r[split].get(k)) for r in selected)}
        fields['best_epoch']=_training_stats([r['best_epoch'] for r in selected])
        selected_summary.append({'model_id':model,**fields})
        # Catch changed log bytes before returning a verified summary.
        if file_sha256(root/'manifest.json')!=sha or any(file_sha256(root/name)!=value for name,value in files.items()):
            raise ValueError('training logs changed during reading')
        result['inputs'].append({'model_id':model,'path':str(root),'manifest_sha256':sha,'logs':files})
    result['selected_summary']=selected_summary
    result['config']['split_counts']=split_counts
    result['config']['experiment_id']=shared[0]
    result['config']['experiment_hashes']=shared[1]
    return result


def training_tables(data):
    def number(v):return '—' if v is None else f'{v:.6f}'
    header='| Model | Seed | Best epoch | Validation objective | Validation accuracy | Test objective | Test accuracy |'
    lines=[header,'| --- | --- | --- | --- | --- | --- | --- |']
    detail=['| Model | Seed | Split | Metric | Value |','| --- | --- | --- | --- | --- |']
    for r in data['selected']:
        metric='syllable_accuracy' if r['model_id']=='syllable' else 'joint_ift_accuracy'
        lines.append('| '+' | '.join([r['model_id'],str(r['seed']),str(r['best_epoch']),number(r['validation_objective']),
            number(r['validation'].get(metric)),number(r['test'].get('objective')),number(r['test'].get(metric))])+' |')
        for split in ('validation','test'):
            for key,value in sorted(r[split].items()):
                if _training_number(value):detail.append(f"| {r['model_id']} | {r['seed']} | {split} | {key} | {number(value)} |")
    aggregate=['| Model | Split | Metric | Mean | Seed SD |','| --- | --- | --- | --- | --- |']
    for row in data['selected_summary']:
        for split in ('validation','test'):
            for metric,stats in row[split].items():
                aggregate.append(f"| {row['model_id']} | {split} | {metric} | {number(stats['mean'])} | {number(stats['sample_sd'])} |")
        stat=row['best_epoch'];aggregate.append(f"| {row['model_id']} | selection | best_epoch | {number(stat['mean'])} | {number(stat['sample_sd'])} |")
    return {'selected-checkpoints.md':'\n'.join(lines)+'\n', 'selected-metrics.md':'\n'.join(detail)+'\n',
            'selected-summary.md':'\n'.join(aggregate)+'\n'}


def write_training_summary(project_root,run_id,data,*,source_hashes):
    tables=training_tables(data)
    expected={'config.json','data/training_summary.json','diagnostics/summary.json',*('tables/'+k for k in tables)}
    with _evaluation_run(project_root,'training-summary',run_id,input_roots=[r['path'] for r in data['inputs']],
            config=data['config'],source_hashes=source_hashes,expected_files=expected,
            scientific_pins={'training_manifests':{r['model_id']:r['manifest_sha256'] for r in data['inputs']}}) as root:
        _write(root/'data/training_summary.json',data)
        _write(root/'diagnostics/summary.json',{'diagnostics':data['diagnostics'],'selected_count':len(data['selected'])})
        for name,body in tables.items():
            with (root/'tables'/name).open('x',encoding='utf-8',newline='\n') as stream:stream.write(body)
    return root


def load_training_summary(root):
    root=Path(root).resolve(strict=True)
    expected={'config.json','data/training_summary.json','diagnostics/summary.json',
              'tables/selected-checkpoints.md','tables/selected-metrics.md','tables/selected-summary.md'}
    manifest=_verify_inventory(root,'training-summary',root.name,expected)
    data=_finite_json(root/'data/training_summary.json')
    if (data['config']!=_finite_json(root/'config.json') or
            {r['model_id']:r['manifest_sha256'] for r in data['inputs']}!=manifest['scientific_input_pins']['training_manifests']):
        raise ValueError('training summary input identities disagree')
    for name,body in training_tables(data).items():
        if (root/'tables'/name).read_text(encoding='utf-8')!=body:raise ValueError('training table view mismatch')
    return data,manifest
