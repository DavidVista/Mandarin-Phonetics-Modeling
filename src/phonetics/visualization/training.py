"""Five approved training figures from saved pointwise five-seed summaries."""
from __future__ import annotations

from pathlib import Path
import re
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import to_rgb
from matplotlib.ticker import MaxNLocator

from . import load_protocol, apply_style, layout, export_figure
from ..evaluation.training_summary import (
    TRAINING_MODELS, load_training_runs, write_training_summary, load_training_summary,
)
from ..evaluation.evaluation_outputs import _evaluation_run, _write
from ..research.artifacts import file_sha256, safe_run_id, ensure_output_outside_inputs

TRAINING_FIGURES = ('training_objectives','validation_accuracy','validation_loss_components',
                    'component_accuracy','invalid_if_predictions')
TRAINING_STEMS = ('training-objectives','validation-accuracy','validation-loss-components',
                  'component-accuracy','invalid-initial-final-combinations')


def _training_shade(color):
    # Preblend the approved 0.15 gray ribbon onto white: EPS has no alpha channel.
    return tuple(.15*v+.85 for v in to_rgb(color))


def _training_curve(ax,data,model,metric,p,*,split='validation',semantic=None,scale=1.,band=True,linestyle=None):
    item=data['models'][model]
    curve=item['curves'][split].get(metric)
    if curve is None:
        missing=getattr(ax,'_training_missing',[])+[f'{model}: {metric}']
        ax._training_missing=missing
        return
    sem=semantic or p.registry['semantics'][model]
    x=item['epochs'];mean=np.asarray(curve['mean'])*scale;sd=np.asarray(curve['sample_sd'])*scale
    if band:
        ax.fill_between(x,mean-sd,mean+sd,facecolor=_training_shade(sem['color']),edgecolor='none',zorder=0)
    ax.plot(x,mean,color=sem['color'],ls=linestyle or sem['linestyle'],
            marker=sem['marker'],mfc='white',markevery=max(1,len(x)//10),
            lw=p.style['line']['data_pt'],ms=p.style['marker']['size_pt'],zorder=2)


def _training_legend(fig,handles,p,rows=1):
    fig.legend(handles=handles,loc=p.style['legend']['location'],frameon=False,
               ncol=min(p.style['legend']['max_per_row'],int(np.ceil(len(handles)/rows))),
               handlelength=p.style['legend']['handle_length_em'],columnspacing=p.style['legend']['column_spacing_em'])


def _training_model_handle(model,p):
    s=p.registry['semantics'][model]
    return Line2D([],[],color=s['color'],ls=s['linestyle'],marker=s['marker'],mfc='white',
                  label=s['label'],ms=p.style['marker']['size_pt'])


def _training_finish(fig,axes,data,p,fid,*,shared_y=False):
    n=max(len(model['epochs']) for model in data['models'].values())
    for ax in axes:
        ax.set_xlabel('Epoch');ax.set_xlim((1,n) if n>1 else (.5,1.5))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=p.style['axes']['major_target'],integer=True))
        missing=getattr(ax,'_training_missing',[])
        if missing:
            ax.text(.5,.5,'Unavailable metric',transform=ax.transAxes,ha='center',fontsize=p.style['font']['annotation_pt'])
    if shared_y:
        limits=[ax.get_ylim() for ax in axes if ax.lines]
        if limits:
            lo=min(v[0] for v in limits);hi=max(v[1] for v in limits)
            for ax in axes:ax.set_ylim(lo,hi)
    fig._phonetics_resolution={'kind':fid,'dispersion':'pointwise sample SD across five training seeds',
        'seed_ids':[0,1,2,3,4],'confidence_interval':False,'band_encoding':'opaque preblended gray for EPS',
        'missing_metrics':[m for ax in axes for m in getattr(ax,'_training_missing',[])],
        'axis_limits':[{'x':list(ax.get_xlim()),'y':list(ax.get_ylim())} for ax in axes]}


def build_training_figures(data,protocol=None):
    p=protocol or load_protocol();figures=[]
    sd_patch=Patch(facecolor=_training_shade('#171717'),label='± SD across seeds')
    try:
        fig,axes=layout(5,p,columns=3,legend_rows=1,width_mm=180);figures.append(fig)
        for ax,model in zip(axes,TRAINING_MODELS):
            semantic={'color':p.registry['semantics'][model]['color'],'linestyle':'solid','marker':'None'}
            _training_curve(ax,data,model,'objective',p,semantic=semantic,linestyle='solid')
            _training_curve(ax,data,model,'objective',p,split='train',semantic=semantic,band=False,linestyle='dashed')
            rows=data['models'][model]['selected']
            ax.plot([r['best_epoch'] for r in rows],[r['validation_objective'] for r in rows],
                    ls='none',marker='o',mfc='white',mec=semantic['color'],ms=p.style['marker']['size_pt'],zorder=3)
            ax.set(title=p.registry['semantics'][model]['label'],ylabel='Objective')
        _training_legend(fig,[Line2D([],[],color='#171717',label='Validation'),
            Line2D([],[],color='#171717',ls='--',label='Training'),
            Patch(facecolor=_training_shade('#171717'),label='Validation ± SD'),
            Line2D([],[],color='#171717',ls='none',marker='o',mfc='white',label='Selected epoch')],p)
        _training_finish(fig,axes,data,p,'training_objectives')

        fig,axes=layout(4,p,columns=2,legend_rows=1,width_mm=180);figures.append(fig)
        models=[m for m in TRAINING_MODELS if m!='acoustic']
        for ax,model in zip(axes,models):
            _training_curve(ax,data,model,'syllable_accuracy' if model=='syllable' else 'joint_ift_accuracy',p)
            ax.set(title=p.registry['semantics'][model]['label'],ylabel='Accuracy')
        _training_legend(fig,[Line2D([],[],color='#171717',label='Validation'),sd_patch],p)
        _training_finish(fig,axes,data,p,'validation_accuracy',shared_y=True)

        fig,axes=layout(2,p,columns=2,legend_rows=2,width_mm=180);figures.append(fig)
        for ax,models,metric,title in [(axes[0],['components',*TRAINING_MODELS[3:]],'component_ce','Component classification'),
                                      (axes[1],['acoustic',*TRAINING_MODELS[3:]],'cosine','Acoustic alignment')]:
            for model in models:_training_curve(ax,data,model,metric,p)
            ax.set(title=title,ylabel='Mean component CE' if metric=='component_ce' else 'Unweighted cosine loss')
        _training_legend(fig,[_training_model_handle(m,p) for m in TRAINING_MODELS[1:]]+[sd_patch],p,rows=2)
        _training_finish(fig,axes,data,p,'validation_loss_components')

        fig,axes=layout(3,p,columns=2,legend_rows=1,width_mm=180);figures.append(fig)
        for ax,model in zip(axes,['components',*TRAINING_MODELS[3:]]):
            for metric,sid in [('initials_accuracy','initial'),('finals_accuracy','final'),('tones_accuracy','tone')]:
                _training_curve(ax,data,model,metric,p,semantic=p.registry['semantics'][sid])
            ax.set(title=p.registry['semantics'][model]['label'],ylabel='Accuracy')
        _training_legend(fig,[_training_model_handle(m,p) for m in ('initial','final','tone')]+[sd_patch],p)
        _training_finish(fig,axes,data,p,'component_accuracy',shared_y=True)

        fig,axes=layout(1,p,columns=1,legend_rows=1,width_mm=90);figures.append(fig)
        _training_curve(axes[0],data,'components','invalid_if_rate',p,scale=100.)
        axes[0].set(title='Components',ylabel='Invalid predictions (%)')
        _training_legend(fig,[Line2D([],[],color=p.registry['semantics']['components']['color'],label='Validation'),sd_patch],p)
        _training_finish(fig,axes,data,p,'invalid_if_predictions')
        return figures
    except BaseException:
        for fig in figures:plt.close(fig)
        raise


def write_training_report(project_root,run_id,*,summary_output,source_hashes,source_digest,protocol=None):
    if not isinstance(source_digest,str) or not re.fullmatch(r'[0-9a-f]{64}',source_digest):raise ValueError('source SHA-256 digest required')
    data,manifest=load_training_summary(summary_output)
    p=protocol or load_protocol();apply_style(p)
    tables=['selected-checkpoints.md','selected-metrics.md','selected-summary.md']
    expected={'config.json','data/selected-checkpoints.json','diagnostics/summary.json',*('tables/'+n for n in tables)}
    expected.update(f'figures/{stem}.{ext}' for stem in TRAINING_STEMS for ext in ('png','eps','metadata.json'))
    config={'summary_manifest_sha256':file_sha256(Path(summary_output)/'manifest.json'),
            'source_digest':source_digest,'style_sha256':p.style_hash,'registry_sha256':p.registry_hash}
    with _evaluation_run(project_root,'training-report',run_id,input_roots=[summary_output],config=config,
            source_hashes=source_hashes,expected_files=expected,scientific_pins=manifest['scientific_input_pins'],
            profile='training_report',figure_exports_on_complete='complete') as root:
        for name in tables:shutil.copyfile(Path(summary_output)/'tables'/name,root/'tables'/name)
        checkpoints = {'selected':data['selected'],'summary':data['selected_summary']}
        if 'paired_test' in data:
            checkpoints['paired_test'] = data['paired_test']
        _write(root/'data/selected-checkpoints.json',checkpoints)
        figs=build_training_figures(data,p)
        try:
            for fig,stem,fid in zip(figs,TRAINING_STEMS,TRAINING_FIGURES):
                export_figure(fig,root/'figures',stem,fid,p,scientific_provenance={
                    'source_digest':source_digest,'summary_manifest_sha256':config['summary_manifest_sha256']})
        finally:
            for fig in figs:plt.close(fig)
        _write(root/'diagnostics/summary.json',{'figure_count':5,'diagnostics':data['diagnostics'],'refit':False})
    return root


def run_training_visualization(project_root,presentation_id,*,training_runs,summary_run_id,mode,
                               source_hashes,source_digest,protocol=None):
    if mode not in {'prepare','reuse'}:raise ValueError('mode must be prepare or reuse')
    p=protocol or load_protocol();apply_style(p)
    base=Path(project_root).resolve()/'outputs'
    report=base/'training-report'/safe_run_id(presentation_id)
    summary=base/'training-summary'/safe_run_id(summary_run_id)
    for target in ([report,summary] if mode=='prepare' else [report]):
        ensure_output_outside_inputs(target,[v['path'] for v in training_runs.values()])
        if target.exists():raise FileExistsError(f'refusing to overwrite training output: {target}')
        if any((parent/'manifest.json').exists() for parent in target.parents):
            raise ValueError('output is inside an existing run')
    if mode=='prepare':
        data=load_training_runs(training_runs)
        write_training_summary(project_root,summary_run_id,data,source_hashes=source_hashes)
    else:
        data,_=load_training_summary(summary)
        expected={m:v['manifest_sha256'] for m,v in training_runs.items()}
        if expected!={r['model_id']:r['manifest_sha256'] for r in data['inputs']}:
            raise ValueError('reused training summary has different input pins')
    return write_training_report(project_root,presentation_id,summary_output=summary,
        source_hashes=source_hashes,source_digest=source_digest,protocol=p)
