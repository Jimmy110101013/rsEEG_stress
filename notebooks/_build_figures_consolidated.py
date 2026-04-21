"""Build figures_consolidated.ipynb — 5 main figures + 3 appendix.

Consolidation plan:
  Fig 1  user-drawn framework schematic (placeholder)
  Fig 2  representation geometry (UMAP × 6 + variance ratio panel)
  Fig 3  honest evaluation (funnel + perm-null KDE)
  Fig 4  contrast-anchoring (dot plot top + UMAP trajectory bottom)
  Fig 5  causal anchor dissection (already v6 canonical)
  Fig 6  Stress collapse (architecture scatter + drift vector plot)

  Fig A.1  disease variance atlas
  Fig B.1  channel topomap
  Fig B.2  per-FM band-stop breakdown (3-dataset)
"""
from __future__ import annotations
import json
from pathlib import Path

NB_OUT = Path("/raid/jupyter-linjimmy1003.md10/UCSD_stress/notebooks/figures_consolidated.ipynb")

def md(src): return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}
def code(src): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                       "source": src.splitlines(keepends=True)}

# ============================================================
# SETUP
# ============================================================
SETUP = r"""import json, os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, FancyArrowPatch

REPO = Path('/raid/jupyter-linjimmy1003.md10/UCSD_stress')
OUT_MAIN = REPO / 'paper/figures/main'
OUT_APP  = REPO / 'paper/figures/appendix'
OUT_MAIN.mkdir(parents=True, exist_ok=True)
OUT_APP.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8,
    'axes.titlesize': 9, 'axes.labelsize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.linestyle': ':', 'grid.linewidth': 0.4, 'grid.alpha': 0.5,
    'savefig.dpi': 300, 'savefig.bbox': 'tight', 'figure.dpi': 110,
})

FM_COLOR = {'labram': '#1f3a5f', 'cbramod': '#B8442C', 'reve': '#2E8B57'}
DS_COLOR = {'stress': '#7A4B00', 'eegmat': '#1F6B6B', 'sleepdep': '#7A4B9C',
            'adftd': '#9467BD', 'tdbrain': '#E377C2'}

CM = 1/2.54
W_SINGLE = 8.9 * CM
W_DOUBLE = 18.3 * CM
FMS = ['labram','cbramod','reve']

def save(fig, name, out_dir=OUT_MAIN):
    fig.savefig(out_dir/f'{name}.pdf'); fig.savefig(out_dir/f'{name}.png')
    print(f'saved → {(out_dir/f"{name}.pdf").relative_to(REPO)} + .png')
"""

# ============================================================
# FIG 2 — Subject-dominated representation geometry
# ============================================================
FIG2 = r"""# Fig 2 — Representation geometry: 2 rows × 3 datasets
# Row 1: variance stacked bars per dataset (3 FM × frozen/FT)
# Row 2: RSA scatter per dataset (3 FM points, label-r vs subject-r)
va = json.load(open(REPO/'paper/figures/_historical/source_tables/variance_analysis_all.json'))
fit = json.load(open(REPO/'results/studies/exp06_fm_task_fitness/fitness_metrics.json'))['per_model_dataset']
sd = json.load(open(REPO/'paper/figures/_historical/source_tables/sleepdep_variance_rsa.json'))
# merge sleepdep into both dicts (va uses label_frac keys, fit uses rsa_*_r keys)
for k, v in sd.items():
    va[k] = v
    fit[k] = v  # sd cells already carry rsa_label_r / rsa_subject_r

DATASETS = ['stress', 'eegmat', 'sleepdep']
DS_TITLE = {'stress': 'STRESS  (subject-label regime)',
            'eegmat': 'EEGMAT  (within-subject regime)',
            'sleepdep': 'SLEEPDEP  (within-subject regime)'}
FM_MARKER = {'labram': 'o', 'cbramod': 's', 'reve': 'D'}

fig = plt.figure(figsize=(W_DOUBLE, W_DOUBLE*0.65))
gs_top = fig.add_gridspec(1, 3, left=0.07, right=0.98, top=0.95, bottom=0.60, wspace=0.22)
gs_bot = fig.add_gridspec(1, 3, left=0.07, right=0.98, top=0.44, bottom=0.08, wspace=0.32)

legend_handles = [
    Patch(color='#B8442C', label='Label'),
    Patch(color='#1f3a5f', label='Subject'),
    Patch(color='#BBBBBB', label='Residual'),
]

# ---- Row 1: variance stacked bars per dataset ----
for col, ds in enumerate(DATASETS):
    ax = fig.add_subplot(gs_top[0, col])
    fm_tick_pos = []
    for i, fm in enumerate(FMS):
        c = va[f'{fm}_{ds}']
        fr_l, fr_s = c['frozen_label_frac'], c['frozen_subject_frac']
        ft_l, ft_s = c['ft_label_frac'], c['ft_subject_frac']
        if ft_l is None: ft_l = 0
        if ft_s is None: ft_s = 0
        fr_r = max(0, 100-fr_l-fr_s); ft_r = max(0, 100-ft_l-ft_s)
        x0 = i*2.3; x1 = i*2.3 + 0.95
        for xp, (l, sb, r), hatch in zip(
                [x0, x1],
                [(fr_l, fr_s, fr_r), (ft_l, ft_s, ft_r)],
                [None, '///']):
            ax.bar(xp, l,  color='#B8442C', width=0.9, hatch=hatch, edgecolor='white', lw=0.3)
            ax.bar(xp, sb, bottom=l,    color='#1f3a5f', width=0.9, hatch=hatch, edgecolor='white', lw=0.3)
            ax.bar(xp, r,  bottom=l+sb, color='#BBBBBB', width=0.9, hatch=hatch, edgecolor='white', lw=0.3)
        # Mini 'fr'/'FT' label just above the axis (inside plot area, small)
        ax.text(x0, 2, 'fr', ha='center', va='bottom', fontsize=6.2, color='white', fontweight='bold')
        ax.text(x1, 2, 'FT', ha='center', va='bottom', fontsize=6.2, color='white', fontweight='bold')
        fm_tick_pos.append(((x0+x1)/2, fm.upper(), FM_COLOR[fm]))
    ax.set_xticks([p[0] for p in fm_tick_pos])
    ax.set_xticklabels([p[1] for p in fm_tick_pos], fontsize=7.3, fontweight='bold')
    for lbl, (_, _, c) in zip(ax.get_xticklabels(), fm_tick_pos):
        lbl.set_color(c)
    ax.set_ylim(0, 100); ax.set_xlim(-0.8, 2*2.3 + 1.5)
    ax.set_title(DS_TITLE[ds], fontsize=8.5, pad=4)
    if col == 0:
        ax.set_ylabel('Variance fraction (%)', fontsize=8.5)
    else:
        ax.set_yticklabels([])
# Figure-level legend in the strip between rows 1 and 2
legend_handles_full = legend_handles + [
    Patch(facecolor='white', edgecolor='#555', hatch='///', label='hatched = FT'),
]
fig.legend(handles=legend_handles_full, loc='center', bbox_to_anchor=(0.5, 0.515),
           ncol=4, frameon=False, fontsize=7.2, handlelength=1.2,
           handletextpad=0.4, columnspacing=1.8)

# ---- Row 2: RSA scatter per dataset (frozen → FT arrows) ----
# Pull FT-side RSA for stress + eegmat (and sleepdep carries its own FT values)
ft_se = json.load(open(REPO/'paper/figures/_historical/source_tables/ft_rsa_stress_eegmat.json'))

def rsa_pair(fm, ds):
    # returns (frozen_label_r, frozen_subj_r, ft_label_r, ft_subj_r)
    k = f'{fm}_{ds}'
    fr = fit[k]
    if ds == 'sleepdep':
        ft_l = sd[k]['ft_rsa_label_r']; ft_s = sd[k]['ft_rsa_subject_r']
    else:
        ft_l = ft_se[k]['ft_rsa_label_r']; ft_s = ft_se[k]['ft_rsa_subject_r']
    return fr['rsa_label_r'], fr['rsa_subject_r'], ft_l, ft_s

for col, ds in enumerate(DATASETS):
    ax = fig.add_subplot(gs_bot[0, col])
    lo, hi = -0.06, 0.36
    ax.plot([lo, hi], [lo, hi], ls=':', color='#AAA', lw=0.6, zorder=1)
    ax.axhline(0, color='k', lw=0.35); ax.axvline(0, color='k', lw=0.35)
    for fm in FMS:
        fl, fs, tl, ts = rsa_pair(fm, ds)
        c = FM_COLOR[fm]
        # Frozen: hollow marker
        ax.scatter(fl, fs, s=32, facecolor='white', edgecolor=c, lw=1.0,
                   marker=FM_MARKER[fm], zorder=3)
        # Arrow frozen → FT
        ax.annotate('', xy=(tl, ts), xytext=(fl, fs),
                    arrowprops=dict(arrowstyle='-|>,head_length=0.35,head_width=0.22',
                                    color=c, lw=1.0, alpha=0.85))
        # FT: filled marker
        ax.scatter(tl, ts, s=34, facecolor=c, edgecolor='k', lw=0.4,
                   marker=FM_MARKER[fm], zorder=4)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(labelsize=6.8)
    ax.set_xlabel('r(FM RDM, label RDM)', fontsize=7.8)
    if col == 0:
        ax.set_ylabel('r(FM RDM, subject RDM)', fontsize=7.8)
    if col == len(DATASETS) - 1:
        fm_leg = [Line2D([],[], marker=FM_MARKER[fm], ls='', color=FM_COLOR[fm],
                         markeredgecolor=FM_COLOR[fm], markersize=5, label=fm.upper())
                  for fm in FMS]
        state_leg = [
            Line2D([],[], marker='o', ls='', markerfacecolor='white',
                   markeredgecolor='#555', markersize=5, label='frozen'),
            Line2D([],[], marker='o', ls='', markerfacecolor='#555',
                   markeredgecolor='k', markersize=5, label='FT'),
        ]
        l1 = ax.legend(handles=fm_leg, loc='lower right', fontsize=6.2,
                       frameon=False, handletextpad=0.2, borderaxespad=0.3)
        ax.add_artist(l1)
        ax.legend(handles=state_leg, loc='upper left', fontsize=6.2,
                  frameon=False, handletextpad=0.2, borderaxespad=0.3)

save(fig, 'fig2_representation_geometry')"""


# ============================================================
# FIG 3 — Honest evaluation
# ============================================================
FIG3 = r"""# Fig 3 — Honest evaluation closes the Wang gap
# Left: funnel; Right: perm-null KDE × 2 (LaBraM × {Stress, EEGMAT})

def load_ms(ds, fm):
    p = REPO/f'results/studies/perwindow_lp_all/{ds}/{fm}_multi_seed.json'
    return json.load(open(p)) if p.exists() else None

sub_ba = {fm: load_ms('stress', fm)['mean_3seed_42_123_2024'] for fm in FMS}
ba_lo, ba_hi = min(sub_ba.values()), max(sub_ba.values())

cl = json.load(open(REPO/'paper/figures/_historical/source_tables/f03_f16_classical_70rec.json'))
cl_xgb = float(np.mean([f['bal_acc'] for f in cl['models']['XGBoost']['folds']]))

null_dir = REPO/'results/studies/exp27_paired_null/stress'
nulls_stress = np.array([json.load(open(d/'summary.json'))['subject_bal_acc']
                         for d in sorted(null_dir.iterdir()) if (d/'summary.json').exists()])
null_dir_e = REPO/'results/studies/exp27_paired_null/eegmat'
nulls_eeg = np.array([json.load(open(d/'summary.json'))['subject_bal_acc']
                      for d in sorted(null_dir_e.iterdir()) if (d/'summary.json').exists()])

fp = json.load(open(REPO/'results/studies/exp_30_sdl_vs_between/tables/fm_performance.json'))
def real_ft(ds, fm='labram'):
    rows = [r for r in fp if r['dataset']==ds and r['fm']==fm and r['mode']=='ft' and r['bal_acc'] is not None]
    return float(np.mean([r['bal_acc'] for r in rows]))
real_s, real_e = real_ft('stress'), real_ft('eegmat')

fig = plt.figure(figsize=(W_DOUBLE, W_DOUBLE*0.40))
gs = fig.add_gridspec(nrows=2, ncols=2, left=0.07, right=0.98, top=0.92, bottom=0.13,
                       width_ratios=[1.1,1], wspace=0.35, hspace=0.55)
ax_f = fig.add_subplot(gs[:,0])
ax_s = fig.add_subplot(gs[0,1])
ax_e = fig.add_subplot(gs[1,1])

steps = [
    ('Wang 2025 (trial-CV)',            0.9047,              '#B8442C'),
    (f'Subject-disjoint 5-fold\n(3 FM range)', (ba_lo+ba_hi)/2, '#1f3a5f'),
    ('Permutation null\nbaseline',       float(nulls_stress.mean()), '#888'),
    ('Classical XGBoost\n(70 rec)',       cl_xgb,              '#2E8B57'),
]
for i,(lab,v,col) in enumerate(steps):
    ax_f.barh(-i, v, color=col, height=0.6, edgecolor='k', lw=0.3)
    ax_f.text(v+0.01, -i, f'{v:.3f}', va='center', fontsize=7)
ax_f.errorbar((ba_lo+ba_hi)/2, -1, xerr=[[(ba_lo+ba_hi)/2-ba_lo],[ba_hi-(ba_lo+ba_hi)/2]],
              fmt='none', color='k', capsize=3)
ax_f.axvline(0.5, color='k', ls=':', lw=0.6)
ax_f.set_yticks(range(0,-len(steps),-1)); ax_f.set_yticklabels([s[0] for s in steps], fontsize=7.5)
ax_f.set_xlim(0.4,1.0); ax_f.set_xlabel('Balanced accuracy (Stress)', fontsize=8.5)

# perm-null densities
for ax, nulls, real, ds in [(ax_s, nulls_stress, real_s, 'Stress'),
                             (ax_e, nulls_eeg, real_e, 'EEGMAT')]:
    ax.hist(nulls, bins=10, color='#BBBBBB', edgecolor='k', lw=0.3, alpha=0.85)
    ax.axvline(real, color='#B8442C', lw=1.8)
    k = int(np.sum(nulls >= real)); p = (k+1)/(len(nulls)+1)
    ax.set_title(f'LaBraM × {ds}    p = {p:.2f}', fontsize=8.5)
    ax.set_xlabel('Subject-level BA (null)', fontsize=7.5)
    ax.set_ylabel('n seeds', fontsize=7.5)
    ax.text(real+0.01, ax.get_ylim()[1]*0.85, f'real={real:.2f}', color='#B8442C', fontsize=7)

plt.show()
save(fig, 'fig3_honest_evaluation')
"""

# ============================================================
# FIG 4 — Contrast-anchoring
# ============================================================
FIG4 = r"""# Fig 4 — Contrast-anchoring: between-subject + within-subject
# Top: Cleveland dot plot (6 rows); bottom: UMAP feature-space trajectory × 6 cells
import umap, pandas as pd

def load_ms(ds, fm):
    p = REPO/f'results/studies/perwindow_lp_all/{ds}/{fm}_multi_seed.json'
    return json.load(open(p)) if p.exists() else None

def load_feat(fm, ds):
    ch = 30 if ds=='stress' else 19
    d = np.load(REPO/f'results/features_cache/frozen_{fm}_{ds}_{ch}ch.npz', allow_pickle=True)
    return d['features'], d['labels'], d['patient_ids']

def umap2(X, seed=42):
    return umap.UMAP(n_components=2, n_neighbors=10, min_dist=0.25,
                     random_state=seed, metric='cosine').fit_transform(X)

sup = json.load(open(REPO/'results/studies/exp11_longitudinal_dss/within_subject_supplementary.json'))
stress_df = pd.read_csv(REPO/'data/comprehensive_labels.csv')[['Patient_ID','Stress_Score']]

fig = plt.figure(figsize=(W_DOUBLE, W_DOUBLE*0.92))
gs_top = fig.add_gridspec(nrows=1, ncols=1, left=0.22, right=0.95, top=0.97, bottom=0.72)
gs_bot = fig.add_gridspec(nrows=2, ncols=3, left=0.08, right=0.98, top=0.57, bottom=0.09,
                           hspace=0.22, wspace=0.06)

# ---- Top: Cleveland dot plot (between-subject LP) ----
ax_d = fig.add_subplot(gs_top[0])
rows = []  # (label, mean, std, color)
for ds in ['eegmat','stress']:
    for fm in FMS:
        j = load_ms(ds, fm)
        rows.append((f'{fm.upper()}·{ds.upper()}', j['mean_3seed_42_123_2024'],
                     j['std_3seed_42_123_2024_ddof1'], DS_COLOR[ds]))
y = np.arange(len(rows))
for i,(lab, m, s, c) in enumerate(rows):
    ax_d.errorbar(m, y[i], xerr=s, fmt='o', color=c, capsize=3, ms=8,
                  markeredgecolor='k', markeredgewidth=0.4)
ax_d.set_yticks(y); ax_d.set_yticklabels([r[0] for r in rows], fontsize=7.5)
ax_d.axvline(0.5, color='k', ls=':', lw=0.6)
ax_d.set_xlim(0.35, 0.85)
ax_d.set_xlabel('Subject-disjoint 5-fold BA (3-seed mean ± SD)', fontsize=8.5)
ax_d.invert_yaxis()
ax_d.set_title('Between-subject LP', fontsize=9, pad=6)

# ---- Bottom: UMAP trajectory (same as Fig 4.4C v2) ----
for row, ds in enumerate(['eegmat','stress']):
    for col, fm in enumerate(FMS):
        ax = fig.add_subplot(gs_bot[row, col])
        X, y2, pids = load_feat(fm, ds)
        emb = umap2(X)
        ax.scatter(emb[:,0], emb[:,1], s=8, c='#CCC', alpha=0.55, edgecolors='none')
        if ds == 'eegmat':
            for s in np.unique(pids):
                m = (pids==s)
                if m.sum()!=2 or len(np.unique(y2[m]))!=2: continue
                ri = np.where(m & (y2==0))[0][0]; ti = np.where(m & (y2==1))[0][0]
                ax.annotate('', xy=emb[ti], xytext=emb[ri],
                            arrowprops=dict(arrowstyle='->', color='#1f3a5f', lw=0.7, alpha=0.7))
                ax.scatter(*emb[ri], s=22, c='#A9CCE3', edgecolors='#1f3a5f', linewidths=0.4, zorder=3)
                ax.scatter(*emb[ti], s=22, c='#B8442C', edgecolors='#7A2F00', linewidths=0.4, zorder=3)
            dc = sup['frozen']['eegmat'][fm]['dir_consistency']
        else:
            for s in np.unique(pids):
                m_idx = np.where(pids==s)[0]
                if len(m_idx) < 2: continue
                subj = stress_df[stress_df['Patient_ID']==int(s)].sort_values('Stress_Score')
                if len(subj) < 2: continue
                ord_idx = np.argsort(subj['Stress_Score'].values)
                lo = m_idx[ord_idx[0]] if ord_idx[0] < len(m_idx) else m_idx[0]
                hi = m_idx[ord_idx[-1]] if ord_idx[-1] < len(m_idx) else m_idx[-1]
                if lo == hi: continue
                ax.annotate('', xy=emb[hi], xytext=emb[lo],
                            arrowprops=dict(arrowstyle='->', color='#7A4B00', lw=0.7, alpha=0.6))
                ax.scatter(*emb[lo], s=22, c='#A9CCE3', edgecolors='#1f3a5f', linewidths=0.4, zorder=3)
                ax.scatter(*emb[hi], s=22, c='#B8442C', edgecolors='#7A2F00', linewidths=0.4, zorder=3)
            dc = sup['frozen']['stress'][fm]['dir_consistency']
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f'{fm.upper()}  dir={dc:+.3f}', fontsize=8)
        if row == 1:
            ax.set_xlabel('UMAP-1', fontsize=7.5)
        if col == 0:
            ax.set_ylabel('UMAP-2', fontsize=7.5)

# Row labels (EEGMAT / STRESS) as figure-level text on the left gutter
fig.text(0.025, (0.57+0.33)/2, 'EEGMAT', rotation=90, fontsize=9, fontweight='bold',
         va='center', ha='center', color='#1F6B6B')
fig.text(0.025, (0.33+0.09)/2, 'STRESS', rotation=90, fontsize=9, fontweight='bold',
         va='center', ha='center', color='#7A4B00')

# Shared legend strip between dot plot and trajectory grid (y ≈ 0.60–0.70)
traj_leg = [
    Line2D([],[], marker='o', ls='', color='#CCC', markeredgecolor='none',
           markersize=5, label='windows (context)'),
    Line2D([],[], marker='o', ls='', color='#A9CCE3', markeredgecolor='#1f3a5f',
           markersize=8, label='start: rest (EEGMAT) / low-DSS (Stress)'),
    Line2D([],[], marker='o', ls='', color='#B8442C', markeredgecolor='#7A2F00',
           markersize=8, label='end: task (EEGMAT) / high-DSS (Stress)'),
    Line2D([],[], marker=r'$\rightarrow$', ls='', color='#1f3a5f',
           markersize=10, label='EEGMAT rest→task'),
    Line2D([],[], marker=r'$\rightarrow$', ls='', color='#7A4B00',
           markersize=10, label='Stress low→high DSS'),
]
fig.legend(handles=traj_leg, loc='center', bbox_to_anchor=(0.52, 0.645),
           ncol=3, fontsize=6.8, frameon=False, handletextpad=0.3, columnspacing=1.2)

plt.show()
save(fig, 'fig4_contrast_anchoring')
"""

# ============================================================
# FIG 5 — Causal anchor dissection (reuse v6 verbatim)
# ============================================================
FIG5 = r"""# Fig 5 — Causal anchor dissection (PSD + FOOOF scatter + band-stop)
from scipy.signal import welch

SFREQ = 200; FMIN, FMAX = 1, 50
def mean_psd(sig):
    s = sig.reshape(-1, sig.shape[-1])
    f, P = welch(s, fs=SFREQ, nperseg=min(512, s.shape[-1]), axis=-1)
    m = (f>=FMIN) & (f<=FMAX); return f[m], P.mean(axis=0)[m]

def rep_psd(ds):
    p = REPO/f'results/features_cache/fooof_ablation/{ds}_norm_none.npz'
    d = np.load(p, allow_pickle=True)
    ri = int(d['quality_r2'].mean(axis=1).argmax())
    mask = d['window_rec_idx']==ri
    f,P_ap = mean_psd(d['periodic_removed'][mask])
    _,P_pe = mean_psd(d['aperiodic_removed'][mask])
    b = float(d['aperiodic_b'][ri].mean()); chi = float(d['aperiodic_chi'][ri].mean())
    return dict(f=f, ap=P_ap, pe=P_pe, fit=10**b/(f**chi), chi=chi)

psd = {ds: rep_psd(ds) for ds in ['eegmat','sleepdep','stress']}
F = {ds: json.load(open(REPO/f'results/studies/fooof_ablation/{ds}_probes.json'))['results']
     for ds in ['eegmat','sleepdep','stress']}
bs = json.load(open(REPO/'results/studies/exp14_channel_importance/band_stop_ablation.json'))

DS_ORDER = ['eegmat','sleepdep','stress']
DS_SHORT = {'eegmat':'EEGMAT','sleepdep':'SleepDep','stress':'Stress'}
DS_CMAP  = {'eegmat':'#2E8B8B','sleepdep':'#7A4B9C','stress':'#D55E00'}

fig = plt.figure(figsize=(W_DOUBLE, W_DOUBLE*0.70))
gs_top = fig.add_gridspec(1, 3, left=0.07, right=0.98, top=0.95, bottom=0.57, wspace=0.40)
gs_bot = fig.add_gridspec(1, 2, left=0.07, right=0.98, top=0.44, bottom=0.08, wspace=0.32)

for i, ds in enumerate(DS_ORDER):
    ax = fig.add_subplot(gs_top[0,i])
    d = psd[ds]
    ax.loglog(d['f'], d['ap']+d['pe'], color='#222', lw=1.4, label='PSD')
    ax.loglog(d['f'], d['fit'], color='#D55E00', ls='--', lw=1.1, label='aperiodic fit')
    ax.fill_between(d['f'], d['fit'], d['fit']+d['pe'], color='#0072B2', alpha=0.28, label='periodic peaks')
    ax.set_xlabel('Freq (Hz)', fontsize=8)
    if i==0: ax.set_ylabel('PSD (log)', fontsize=8)
    ax.set_title(DS_SHORT[ds], fontsize=10)
    ax.legend(fontsize=6.5, loc='lower left', frameon=False)
    ax.grid(True, which='both', ls=':', lw=0.3, alpha=0.5)

ax_sc = fig.add_subplot(gs_bot[0,0])
points = []
for ds in DS_ORDER:
    for cond, short in [('aperiodic_removed','−aperiodic'),('periodic_removed','−periodic')]:
        dsub = np.mean([F[ds][fm][cond]['subject_probe_mean']-F[ds][fm]['original']['subject_probe_mean'] for fm in FMS])*100
        dsta = np.mean([F[ds][fm][cond]['state_probe_mean']-F[ds][fm]['original']['state_probe_mean'] for fm in FMS])*100
        points.append((ds, short, dsub, dsta))
ax_sc.axhline(0, color='k', lw=0.5); ax_sc.axvline(0, color='k', lw=0.5)
for ds in DS_ORDER:
    pts = [(p[2],p[3]) for p in points if p[0]==ds]
    ax_sc.plot(*zip(*pts), color=DS_CMAP[ds], lw=1.0, alpha=0.35)
markers = {'−aperiodic':'o','−periodic':'s'}
for ds, cond, dsub, dsta in points:
    ax_sc.scatter(dsub, dsta, s=95, color=DS_CMAP[ds], marker=markers[cond],
                  edgecolor='k', lw=0.5, zorder=3)
ax_sc.set_xlabel('Δ Subject-ID probe BA (pp)', fontsize=8.5)
ax_sc.set_ylabel('Δ State probe BA (pp)', fontsize=8.5)
ax_sc.set_xlim(-16, 4); ax_sc.set_ylim(-6, 2)
ax_sc.set_title('FOOOF ablation signature', fontsize=10)
ds_leg = [Line2D([],[], marker='o', ls='', color=DS_CMAP[d], markeredgecolor='k',
                 markersize=8, label=DS_SHORT[d]) for d in DS_ORDER]
cond_leg = [Line2D([],[], marker='o', ls='', color='gray', markeredgecolor='k', markersize=7, label='−aperiodic'),
            Line2D([],[], marker='s', ls='', color='gray', markeredgecolor='k', markersize=7, label='−periodic')]
leg = ax_sc.legend(handles=ds_leg, loc='lower left', fontsize=7, frameon=False)
ax_sc.add_artist(leg)
ax_sc.legend(handles=cond_leg, loc='upper left', fontsize=7, frameon=False)

ax_ln = fig.add_subplot(gs_bot[0,1])
BANDS = ['delta','theta','alpha','beta']
xi = np.arange(len(BANDS))
for ds in DS_ORDER:
    v = [np.mean([bs[ds][fm][b]['mean_distance'] for fm in FMS]) for b in BANDS]
    ax_ln.plot(xi, v, '-o', color=DS_CMAP[ds], lw=1.6, ms=7,
               markeredgecolor='white', markeredgewidth=0.6, label=DS_SHORT[ds])
ax_ln.set_xticks(xi); ax_ln.set_xticklabels([b.title() for b in BANDS], fontsize=8)
ax_ln.set_xlabel('Frequency band', fontsize=8.5)
ax_ln.set_ylabel('Cosine distance', fontsize=8.5)
ax_ln.set_title('Band-stop sensitivity', fontsize=10)
ax_ln.legend(fontsize=7, frameon=False, loc='upper right')
ax_ln.grid(True, ls=':', lw=0.3, alpha=0.5)
ax_ln.set_ylim(-0.003, 0.115)
plt.show()
save(fig, 'fig5_causal_anchor_dissection')
"""

# ============================================================
# FIG 6 — Stress collapse: architecture ceiling + drift vector
# ============================================================
FIG6 = r"""# Fig 6 — Stress collapse: architecture ceiling (with classical) + drift vectors
fp = json.load(open(REPO/'results/studies/exp_30_sdl_vs_between/tables/fm_performance.json'))
cl = json.load(open(REPO/'paper/figures/_historical/source_tables/f03_f16_classical_70rec.json'))

# FM bal_acc on stress FT
fm_ba = {}
for fm in FMS:
    row = next((r for r in fp if r['dataset']=='stress' and r['fm']==fm and r['mode']=='ft'), None)
    if row: fm_ba[fm] = (row['bal_acc'], row.get('bal_acc_std') or 0)

# Non-FM archs (eegnet, shallowconvnet)
sweep_dir = REPO/'results/studies/exp15_nonfm_baselines/sweep'
arch_ba = {}
for arch in ['eegnet','shallowconvnet']:
    vals = []
    for sub in sweep_dir.iterdir():
        if sub.name.startswith(arch+'_'):
            s = sub/'summary.json'
            if s.exists():
                try: vals.append(json.load(open(s))['subject_bal_acc'])
                except: pass
    if vals:
        arch_ba[arch] = (float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals)>1 else 0)

# Classical baselines (70-rec, per-fold)
cl_pts = []
for name in ['LogReg_L2','SVM_RBF','RF','XGBoost']:
    folds = [f['bal_acc'] for f in cl['models'][name]['folds']]
    cl_pts.append((name, float(np.mean(folds)), float(np.std(folds, ddof=1))))

ARCH_PARAMS = {'LogReg_L2': 150, 'SVM_RBF': 200, 'RF': 800, 'XGBoost': 5000,
               'eegnet': 2e3, 'shallowconvnet': 4e4,
               'labram': 5.8e6, 'cbramod': 1e8, 'reve': 1.4e9}

# Drift data
dr = json.load(open(REPO/'results/studies/representation_drift/lp_vs_ft_stress.json'))['results']

# FIGURE
fig = plt.figure(figsize=(W_DOUBLE, W_DOUBLE*0.42))
gs = fig.add_gridspec(1, 2, left=0.07, right=0.98, top=0.93, bottom=0.18, wspace=0.30)

# ---- Left: architecture ceiling ----
ax_a = fig.add_subplot(gs[0,0])
ax_a.axhspan(0.43, 0.58, color='#FFF2CC', alpha=0.6, zorder=0, label='0.43–0.58 ceiling')
ax_a.axhline(0.5, color='k', ls=':', lw=0.6)

# classical points (grey)
for name, m, s in cl_pts:
    ax_a.errorbar(ARCH_PARAMS[name], m, yerr=s, fmt='s', color='#888', ms=5,
                  capsize=2, markeredgecolor='k', markeredgewidth=0.3)
    ax_a.text(ARCH_PARAMS[name]*1.3, m, name.replace('LogReg_L2','LR').replace('XGBoost','XGB').replace('SVM_RBF','SVM'),
              fontsize=6, va='center', color='#555')

# non-FM archs (dark grey)
for name in ['eegnet','shallowconvnet']:
    if name in arch_ba:
        m, s = arch_ba[name]
        ax_a.errorbar(ARCH_PARAMS[name], m, yerr=s, fmt='D', color='#444', ms=6,
                      capsize=2, markeredgecolor='k', markeredgewidth=0.3)
        ax_a.text(ARCH_PARAMS[name]*1.3, m, name, fontsize=6, va='center', color='#333')

# FMs (colored)
for fm in FMS:
    if fm in fm_ba:
        m, s = fm_ba[fm]
        ax_a.errorbar(ARCH_PARAMS[fm], m, yerr=s, fmt='o', color=FM_COLOR[fm], ms=8,
                      capsize=3, markeredgecolor='k', markeredgewidth=0.4)
        ax_a.text(ARCH_PARAMS[fm]*1.3, m, fm.upper(), fontsize=7, va='center', color=FM_COLOR[fm], fontweight='bold')

ax_a.set_xscale('log')
ax_a.set_xlabel('Trainable params (log)', fontsize=8.5)
ax_a.set_ylabel('Subject-CV BA (Stress)', fontsize=8.5)
ax_a.set_ylim(0.30, 0.72)
ax_a.set_title('Architecture ceiling (classical + non-FM + FM)', fontsize=9)
legend_marks = [Line2D([],[],marker='s',ls='',color='#888',markeredgecolor='k',ms=5,label='Classical ML'),
                Line2D([],[],marker='D',ls='',color='#444',markeredgecolor='k',ms=6,label='Non-FM deep'),
                Line2D([],[],marker='o',ls='',color='#1f3a5f',markeredgecolor='k',ms=7,label='FM (3 × FT)')]
ax_a.legend(handles=legend_marks, fontsize=7, loc='lower right', frameon=False)

# ---- Right: drift vector plot ----
ax_v = fig.add_subplot(gs[0,1])
for ds in ['stress','eegmat']:
    for fm in FMS:
        lp = dr[ds][fm]['lp']; ft = dr[ds][fm]['ft']
        x0, y0 = lp['label_frac_pct'], lp['subject_frac_pct']
        x1, y1 = ft['label_frac_pct'], ft['subject_frac_pct']
        verdict = dr[ds][fm]['delta']['interpretation']
        c = {'rescue_consistent_with_subject_shortcut':'#B8442C',
             'rescue_consistent_with_label_signal':'#2E8B57',
             'no_meaningful_drift':'#888'}.get(verdict, '#888')
        ax_v.annotate('', xy=(x1,y1), xytext=(x0,y0),
                      arrowprops=dict(arrowstyle='->,head_length=0.35,head_width=0.2',
                                      color=c, lw=1.3, alpha=0.9))
        ax_v.scatter(x0, y0, s=22, color='white', edgecolors=c, linewidths=1.0, zorder=3)
        ax_v.scatter(x1, y1, s=45, color=c, edgecolors='k', linewidths=0.4, zorder=4,
                     marker={'stress':'o','eegmat':'s'}[ds])
        ax_v.text(x1+0.5, y1+0.5, f'{fm}·{ds[:3]}', fontsize=6, color=c)

ax_v.set_xlabel('Label variance frac (%)', fontsize=8.5)
ax_v.set_ylabel('Subject variance frac (%)', fontsize=8.5)
ax_v.set_title('LP → FT representation drift (arrow = FT direction)', fontsize=9)
leg = [Line2D([],[],marker='>',ls='-',color='#B8442C',markersize=8,label='Subject shortcut'),
       Line2D([],[],marker='>',ls='-',color='#2E8B57',markersize=8,label='Label learning'),
       Line2D([],[],marker='>',ls='-',color='#888',markersize=8,label='No drift'),
       Line2D([],[],marker='o',ls='',color='gray',markeredgecolor='k',markersize=7,label='Stress'),
       Line2D([],[],marker='s',ls='',color='gray',markeredgecolor='k',markersize=7,label='EEGMAT')]
ax_v.legend(handles=leg, fontsize=6.5, loc='upper left', frameon=False)
ax_v.set_xlim(0, 12); ax_v.set_ylim(40, 95)
plt.show()
save(fig, 'fig6_stress_collapse')
"""

# ============================================================
# APPENDIX — A.1 / B.1 / B.2 (keep simple)
# ============================================================
FIG_A1 = r"""# Fig A.1 — ADFTD + TDBRAIN variance atlas
va = json.load(open(REPO/'paper/figures/_historical/source_tables/variance_analysis_all.json'))
DATASETS = ['adftd','tdbrain']
fig, axes = plt.subplots(2, 6, figsize=(W_DOUBLE, W_DOUBLE*0.38), sharey='row')
for col, (fm,ds) in enumerate([(fm,ds) for ds in DATASETS for fm in FMS]):
    c = va[f'{fm}_{ds}']
    ax = axes[0,col]
    fr_l, fr_s = c['frozen_label_frac'], c['frozen_subject_frac']
    ft_l, ft_s = c['ft_label_frac'], c['ft_subject_frac']
    for i,(l,sb,r) in enumerate([(fr_l, fr_s, max(0,100-fr_l-fr_s)),(ft_l, ft_s, max(0,100-ft_l-ft_s))]):
        ax.bar(i, l, color='#D55E00', width=0.7)
        ax.bar(i, sb, bottom=l, color='#0072B2', width=0.7)
        ax.bar(i, r, bottom=l+sb, color='#BBBBBB', width=0.7)
    ax.set_xticks([0,1]); ax.set_xticklabels(['fr','FT'])
    ax.set_title(f'{fm}·{ds}', fontsize=7)
    ax.set_ylim(0, 100)
    if col==0: ax.set_ylabel('Variance frac (%)')
    axes[1,col].set_visible(False)
plt.tight_layout(); plt.show()
save(fig, 'figA1_variance_atlas_disease', out_dir=OUT_APP)
"""

FIG_B1 = r"""# Fig B.1 — Channel ablation topomap (Stress 30-ch, 3 FMs)
import mne
ci = json.load(open(REPO/'results/studies/exp14_channel_importance/channel_importance.json'))
RAW = ci['labram']['channel_names']
montage = mne.channels.make_standard_montage('standard_1005')
mont_map = {c.lower(): c for c in montage.ch_names}
CH = [mont_map.get(c.lower(), c) for c in RAW]
info = mne.create_info(CH, 200, ch_types='eeg'); info.set_montage(montage, on_missing='warn')

fig, axes = plt.subplots(1, 3, figsize=(W_DOUBLE*0.85, W_DOUBLE*0.3))
vmax = max(max(ci[fm]['mean_importance']) for fm in FMS)
for ax, fm in zip(axes, FMS):
    imp = np.array(ci[fm]['mean_importance'])
    im, _ = mne.viz.plot_topomap(imp, info, axes=ax, show=False, cmap='Reds',
                                  vlim=(0, vmax), contours=0, sensors=True)
    ax.set_title(f'{fm.upper()} · max={imp.max():.4f}', fontsize=8)
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, pad=0.02, label='Importance')
plt.show()
save(fig, 'figB1_channel_ablation', out_dir=OUT_APP)
"""

FIG_B2 = r"""# Fig B.2 — Per-FM band-stop breakdown (3 datasets × 3 FM × 4 bands)
bs = json.load(open(REPO/'results/studies/exp14_channel_importance/band_stop_ablation.json'))
BANDS = ['delta','theta','alpha','beta']
fig, axes = plt.subplots(1, 3, figsize=(W_DOUBLE, W_DOUBLE*0.33), sharey=True)
w = 0.25; x = np.arange(len(BANDS))
for ax, ds in zip(axes, ['eegmat','sleepdep','stress']):
    for i, fm in enumerate(FMS):
        vals = [bs[ds][fm][b]['mean_distance'] for b in BANDS]
        errs = [bs[ds][fm][b]['std_distance'] for b in BANDS]
        ax.bar(x + (i-1)*w, vals, w, yerr=errs, capsize=2, color=FM_COLOR[fm],
               edgecolor='k', lw=0.3, label=fm.upper() if ds=='eegmat' else None)
    ax.set_xticks(x); ax.set_xticklabels([b.title() for b in BANDS], fontsize=7)
    ax.set_title(ds.upper(), fontsize=9)
axes[0].set_ylabel('Cosine distance'); axes[0].legend(frameon=False, fontsize=7)
plt.tight_layout(); plt.show()
save(fig, 'figB2_band_stop_breakdown', out_dir=OUT_APP)
"""

# ============================================================
# ASSEMBLE NOTEBOOK
# ============================================================
cells = [
    md("""# Paper figures — consolidated (5 main + 3 appendix)

Each figure answers one claim; ≤2 panels per figure."""),
    code(SETUP),

    md("""## Fig 1 — Framework schematic (user-drawn)

Placeholder; will be imported as `fig1_framework_schematic.pdf`."""),

    md("""## Fig 2 — Representation geometry is subject-dominated

**Top**: UMAP 2×3 (3 FM × {Stress, EEGMAT}), points colored by subject — Stress shows tight subject clusters; EEGMAT clusters are looser.
**Bottom**: Subject/Label variance ratio with 95 % bootstrap CI, log-scale; ratio ≫ 1 indicates subject dominance."""),
    code(FIG2),

    md("""## Fig 3 — Honest evaluation closes the Wang gap

**Left**: funnel collapse — Wang 0.9047 → 3-FM subject-disjoint range → perm-null baseline → classical XGBoost.
**Right**: LaBraM permutation-null densities; red line = real score; p-value annotated."""),
    code(FIG3),

    md("""## Fig 4 — Contrast-anchoring in both eval regimes

**Top**: between-subject dot plot — 3 FM × 2 DS, 3-seed mean ± SD, subject-disjoint 5-fold.
**Bottom**: within-subject feature-space trajectory × 6 cells — EEGMAT (rest→task) arrows align; Stress (low→high DSS) arrows scramble. `dir_consistency` printed per panel."""),
    code(FIG4),

    md("""## Fig 5 — Causal anchor dissection

**Top row**: PSD + FOOOF fit (aperiodic dashed + periodic shaded).
**Bottom-left**: FOOOF ablation signature scatter (Δ subject-ID × Δ state probe BA).
**Bottom-right**: Band-stop cosine distance profile across frequency bands.

Cosine distance = FM feature drift; not a task-probe accuracy. Must be read with scatter."""),
    code(FIG5),

    md("""## Fig 6 — Stress collapse mechanism

**Left**: Architecture ceiling — 4 classical ML + 2 non-FM deep + 3 FMs, param count log-scale. All sit within the 0.43–0.58 band.
**Right**: LP→FT representation drift — each arrow goes from LP (hollow circle) to FT (filled) in (label_frac, subject_frac) space; colour = mechanism verdict (shortcut / label / no-drift)."""),
    code(FIG6),

    md("""## Appendix A.1 — Disease cohort variance atlas"""),
    code(FIG_A1),

    md("""## Appendix B.1 — Channel ablation topomap (Stress)"""),
    code(FIG_B1),

    md("""## Appendix B.2 — Per-FM band-stop breakdown (3 datasets)"""),
    code(FIG_B2),
]

nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"name":"stress","display_name":"Python (stress)","language":"python"},
        "language_info": {"name":"python","version":"3.11"},
    },
    "cells": cells,
}
NB_OUT.write_text(json.dumps(nb, indent=1))
print(f'wrote {NB_OUT}  ({NB_OUT.stat().st_size/1024:.1f} KB, {len(cells)} cells)')
