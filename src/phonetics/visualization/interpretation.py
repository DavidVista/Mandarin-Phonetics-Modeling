"""Approved forest, geometry and Q–Q panels from prepared numerical data."""
from __future__ import annotations

import math
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from . import layout, shared_legend


def _grow_height(fig, axes, panel_mm):
    """Grow categorical panels to fit their rows using physical font spacing."""
    width, old_height = fig.get_size_inches() * 25.4
    current = axes[0].get_position().height * old_height
    growth = max(0., panel_mm - current)
    height = old_height + growth
    positions = [ax.get_position().bounds for ax in axes]
    fig.set_size_inches(width / 25.4, height / 25.4)
    for ax, (x, y, w, h) in zip(axes, positions):
        ax.set_position([x, y * old_height / height, w, (h * old_height + growth) / height])


def build_anchor_forest(rows, protocol):
    """Three component panels; each transition has five separate model rows."""
    p = protocol
    spec = p.figure('anchor_q_transitions')
    models = spec['order']
    fig, axes = layout(3, p, columns=3, legend_rows=2, width_mm=180)
    try:
        _grow_height(fig, axes, p.style['font']['tick_pt'] * 25.4 / 72 * 1.3 * 27)
        # Reserve room for binary transition labels on every panel, as in the mock.
        for ax in axes:
            pos=ax.get_position()
            ax.set_position([pos.x0+.055,pos.y0,pos.width-.055,pos.height])
        found = {}
        for row in rows:
            key = (row['model_id'], row['edge_id'])
            if key in found: raise ValueError('duplicate forest model/transition')
            found[key] = row
        values = []
        for ax, comp in zip(axes, ('initial', 'final', 'tone')):
            edges = [r['edge_id'] for r in rows if r['model_id'] == models[0] and r['edge_id'].startswith(comp+':')]
            if len(edges) != 4 or sum(e.endswith('->111') for e in edges) != 1:
                raise ValueError('four transitions per component are required')
            edges = [e for e in edges if not e.endswith('->111')] + [e for e in edges if e.endswith('->111')]
            centers = [0, 7, 14, 22]
            for block, edge in enumerate(edges):
                for mi, model in enumerate(models):
                    row = found[(model, edge)]
                    seed_rows = row['seed_points']
                    if {s['seed'] for s in seed_rows} != set(range(5)) or len(seed_rows) != 5:
                        raise ValueError('five forest training seeds are required')
                    # These are persisted seed estimates; this is display dispersion, not MC SE.
                    sd = row.get('sample_sd')
                    if sd is None: sd = float(np.std([s['q'] for s in seed_rows], ddof=1))
                    mean = row['mean']
                    if not np.isfinite([mean, sd]).all() or sd < 0:
                        raise ValueError('invalid forest estimate')
                    values.extend((mean-sd, mean+sd))
                    sem = p.registry['semantics'][model]
                    ax.errorbar(mean, centers[block]+(mi-2)*.9, xerr=sd,
                                fmt=sem['marker'], color=sem['color'], mfc='white',
                                ms=p.style['marker']['size_pt'], lw=p.style['line']['interval_pt'],
                                capsize=p.style['line']['caps_pt'], label=sem['label'] if block == 0 else None)
            ax.axvline(.5, color=p.style['color']['secondary'], ls='--', lw=p.style['line']['reference_pt'])
            ax.axhline(18.4, color=p.style['color']['secondary'], lw=p.style['line']['reference_pt'])
            ax.set(yticks=centers, yticklabels=[e.split(':')[1].replace('->', ' → ') for e in edges],
                   ylim=(25.5, -4), title=comp.title(), xlabel='Q')
            ax.tick_params(axis='y', length=0)
        if len(found) != 60: raise ValueError('exact 12 transitions × five models required')
        lo, hi = min(0.4, min(values)-.015), max(1., max(values)+.015)
        for ax in axes: ax.set_xlim(lo, hi)
        shared_legend(fig, axes, p, rows=2)
        fig._phonetics_resolution = {'kind': 'forest', 'interval': 'sample SD of five training seeds',
            'reference_q': .5, 'full_match': 'separated final block', 'mc_se_in_figure': False,
            'xlim': [lo, hi], 'model_order': models}
        return fig
    except BaseException:
        import matplotlib.pyplot as plt
        plt.close(fig)
        raise


def build_dimsim_geometry(plot, protocol):
    """Common data-covering cosine range, per-panel counts and support gaps."""
    p = protocol
    models = p.figure('dimsim_geometry')['order']
    curves = plot['curves']
    threshold=plot.get('min_unique_pairs_per_bin',100)
    if type(threshold) is not int or threshold<=0: raise ValueError('invalid unique-pair support threshold')
    values = [v for row in curves for v in
              [row['mean_cosine'], *[s['mean_cosine'] for s in row['seed_points']]] if v is not None]
    if not values or not np.isfinite(values).all():
        raise ValueError('finite geometry means and seed curves required')
    span = max(max(values)-min(values), .1)
    limits = (min(values)-span*.05, max(values)+span*.05)
    # A 0.5 guide is a preference, never a clipping rule.
    max_count = max(s['draw_count'] for r in curves for s in r['seed_points'])
    locator = MaxNLocator(nbins=5, integer=True, min_n_ticks=3)
    ticks = locator.tick_values(0, max(1, max_count))
    ticks = ticks[ticks >= 0]
    count_top = float(ticks[-1])
    fig, axes = layout(10, p, columns=2, legend_rows=2, width_mm=180)
    try:
        for host in axes:
            pos = host.get_position()
            host.set_position([pos.x0, pos.y0, pos.width-.06, pos.height])
        for mi, model in enumerate(models):
            for col, mode in enumerate(('dimsim-natural', 'dimsim-balanced')):
                ax = axes[2*mi+col]
                rows = [r for r in curves if r['model_id']==model and r['mode']==mode]
                if len(rows) != 13: raise ValueError('all 13 DIMSIM bins required')
                if len({r['bin'] for r in rows}) != 13: raise ValueError('duplicate DIMSIM bins')
                cx = ax.twinx(); ax.set_zorder(cx.get_zorder()+1); ax.patch.set_visible(False)
                draws = [r['seed_points'][0]['draw_count'] for r in rows]
                unique = [r['seed_points'][0]['unique_pair_count'] for r in rows]
                for r in rows:
                    if {s['seed'] for s in r['seed_points']} != set(range(5)) or len(r['seed_points']) != 5:
                        raise ValueError('five geometry training seeds required')
                    if any((s['draw_count'],s['unique_pair_count']) !=
                           (r['seed_points'][0]['draw_count'],r['seed_points'][0]['unique_pair_count']) for s in r['seed_points']):
                        raise ValueError('geometry counts differ across seeds')
                cx.bar(range(13), draws, width=.76, color='#e6e6e6', edgecolor='none')
                cx.bar(range(13), unique, width=.76, facecolor='none', edgecolor='#555555', linewidth=p.style['axes']['major_width_pt'])
                for si in range(5):
                    y = [next(s['mean_cosine'] for s in r['seed_points'] if s['seed']==si) for r in rows]
                    for j in range(12):
                        if rows[j]['supported'] and rows[j+1]['supported'] and y[j] is not None and y[j+1] is not None:
                            ax.plot([j,j+1], y[j:j+2], color='#aaaaaa', lw=p.style['line']['reference_pt'])
                    missing = [(j,v) for j,v in enumerate(y) if not rows[j]['supported'] and v is not None]
                    if missing:
                        x, vals = zip(*missing)
                        ax.plot(x,vals,ls='none',marker='o',mfc='white',mec='#aaaaaa',ms=p.style['marker']['size_pt']*.65)
                for j, row in enumerate(rows):
                    y = row['mean_cosine']
                    if y is not None:
                        ax.plot([j],[y],ls='none',marker='o',mfc='#171717' if row['supported'] else 'white',mec='#171717',ms=p.style['marker']['size_pt'])
                    if j<12 and row['supported'] and rows[j+1]['supported'] and y is not None and rows[j+1]['mean_cosine'] is not None:
                        ax.plot([j,j+1],[y, rows[j+1]['mean_cosine']],color='#171717',lw=p.style['line']['data_pt'])
                cx.set(ylim=(0,count_top),yticks=ticks,ylabel='Count')
                cx.tick_params(labelsize=p.style['font']['tick_pt'],direction='in')
                ax.set(ylim=limits,xlim=(-.5,12.5),xticks=[0,4,8,12],xticklabels=['5','45','85','120+'],
                       ylabel='Cosine similarity' if col==0 else '',
                       title=p.registry['semantics'][model]['label']+' · '+('Natural' if col==0 else 'Balanced'))
                if mi==4: ax.set_xlabel('DIMSIM distance')
        handles = [Line2D([],[],color='#171717',label='Mean cosine'),
                   Line2D([],[],color='#aaaaaa',lw=p.style['line']['reference_pt'],label='Seed curves'),
                   Line2D([],[],color='#171717',ls='none',marker='o',mfc='white',label=f'Support < {threshold}'),
                   Patch(facecolor='#e6e6e6',label='All draws'),
                   Patch(facecolor='none',edgecolor='#555555',label='Unique pairs')]
        fig.legend(handles=handles,loc='upper center',ncol=3,frameon=False,
                   handlelength=p.style['legend']['handle_length_em'])
        fig._phonetics_resolution = {'kind': 'dimsim_geometry', 'cosine_limits': list(limits),
            'counts_limits': [0,count_top], 'counts_ticks': ticks.tolist(), 'boundary': 'removed',
            'support_gaps_not_bridged': True, 'counts_encoding': 'light fill and dark outline',
            'natural_counts': 'draws and unique pairs coincide by design',
            'balanced_counts': 'repeated draws retained', 'support_threshold': threshold}
        return fig
    except BaseException:
        import matplotlib.pyplot as plt
        plt.close(fig)
        raise


def build_residual_qq(qq_data, protocol):
    """Render prepared quantiles only, without standardization or model fitting."""
    p = protocol
    spec = p.figure('residual_qq')
    fig, axes = layout(6,p,columns=2,legend_rows=1,width_mm=180)
    try:
        curves = qq_data['curves']
        valid = [r for r in curves if r['status']=='ok']
        x = [v for r in valid for v in r['normal_quantiles']]
        y = [v for r in valid for v in r['standardized_residuals']]
        xlim = (min(x)-.1,max(x)+.1) if x else (-1.,1.)
        ylim = (min(y)-.1,max(y)+.1) if y else (-1.,1.)
        diagnostics = []
        for ax, model in zip(axes,spec['order']):
            rows = [r for r in curves if r['model_id']==model]
            expected = {None} if model=='dimsim' else set(range(5))
            if len(rows)!=len(expected) or {r['seed'] for r in rows}!=expected:
                raise ValueError('invalid Q–Q model/seed curve inventory')
            for row in rows:
                if row['status']!='ok':
                    diagnostics.append({'model_id':model,'seed':row['seed'],'status':row['status']})
                    continue
                xx,yy=row['normal_quantiles'],row['standardized_residuals']
                if len(xx)!=len(yy) or not len(xx) or not np.isfinite(xx+yy).all():
                    raise ValueError('invalid Q–Q coordinates')
                ax.plot(xx,yy,color='#171717' if model=='dimsim' else '#777777',lw=p.style['line']['reference_pt'])
            ax.plot(xlim,xlim,color='#999999',ls='--',lw=p.style['line']['reference_pt'])
            if not any(r['status']=='ok' for r in rows):
                ax.text(.5,.5,'Undefined',transform=ax.transAxes,ha='center')
            ax.set(xlim=xlim,ylim=ylim,xlabel=spec['xlabel'],ylabel=spec['ylabel'],
                   title=p.registry['semantics'][model]['label'])
        fig.legend(handles=[Line2D([],[],color='#777777',label='Training seeds'),
                            Line2D([],[],color='#171717',label='DIMSIM'),
                            Line2D([],[],color='#999999',ls='--',label='Normal reference')],
                   loc='upper center',ncol=3,frameon=False)
        fig._phonetics_resolution={'kind':'residual_qq','standardization':qq_data['config'],
                                  'xlim':list(xlim),'ylim':list(ylim),'undefined':diagnostics,
                                  'normality_test':False}
        return fig
    except BaseException:
        import matplotlib.pyplot as plt
        plt.close(fig)
        raise
