"""
MICCAI Publication Figure — Clean Version (No Text Labels)
Modified: Removed titles, a/b tags, axis labels, and floating annotations.
"""
import argparse
import numpy as np

# -----------------------------------------------------------------------------
# Data Loading / Synthesis (Unchanged)
# -----------------------------------------------------------------------------
def load_prior(path):
    import torch
    p = torch.load(path, map_location='cpu', weights_only=False)
    for k in ["adjacency_binary", "adjacency_strength", "possible_mask", "raw_contact_counts"]:
        if k in p and hasattr(p[k], 'numpy'):
            p[k] = p[k].numpy()
    return p

def make_synthetic_prior():
    C = 78; np.random.seed(42)
    left_sub = list(range(0,18)); right_sub = list(range(18,32))
    midline = [9,10,11,14,32]; left_ctx = list(range(33,64)); right_ctx = list(range(64,78))
    adj = np.zeros((C,C), dtype=np.float32); strength = np.zeros((C,C), dtype=np.float32)
    def ae(i,j,s): adj[i,j]=adj[j,i]=1; strength[i,j]=strength[j,i]=s
    for g in [left_sub,left_ctx,right_sub,right_ctx]:
        for idx,i in enumerate(g):
            for j in g[idx+1:]:
                if np.random.rand()<0.25: ae(i,j,np.random.uniform(0.05,0.4))
    for c in left_ctx: ae(0,c,np.random.uniform(0.7,1.0))
    for c in right_ctx: ae(18,c,np.random.uniform(0.7,1.0))
    for dg in [5,6,7,8,12,13,15,16]: ae(0,dg,np.random.uniform(0.5,0.8))
    for dg in [23,24,25,26,27,28,29,30]: ae(18,dg,np.random.uniform(0.5,0.8))
    for m in midline:
        if m>=C: continue
        for s in left_sub[:5]+right_sub[:5]:
            if np.random.rand()<0.4: ae(m,s,np.random.uniform(0.2,0.5))
    for c in [3,4,21,22]: ae(11,c,np.random.uniform(0.7,0.9))
    mx=strength.max()
    if mx>0: strength/=mx
    hemi={}
    ls=set(left_sub)|set(left_ctx); rs=set(right_sub)|set(right_ctx); ms=set(midline)
    for i in range(C):
        if i in ms: hemi[i+1]="midline"
        elif i in ls: hemi[i+1]="left"
        elif i in rs: hemi[i+1]="right"
        else: hemi[i+1]="bilateral"
    ln={}
    sl=["CWM-lh","LatV-lh","InfLatV-lh","CbWM-lh","CbCtx-lh","Thal-lh","Caud-lh","Put-lh","Pall-lh","3rdV","4thV","BStem","Hipp-lh","Amyg-lh","CSF","Acc-lh","VDC-lh","ChPl-lh"]
    sr=["CWM-rh","LatV-rh","InfLatV-rh","CbWM-rh","CbCtx-rh","Thal-rh","Caud-rh","Put-rh","Pall-rh","Hipp-rh","Amyg-rh","Acc-rh","VDC-rh","ChPl-rh"]
    for i,n in enumerate(sl): ln[i+1]=n
    for i,n in enumerate(sr): ln[19+i]=n
    ln[33]="WMhypo"
    for i in range(34,65): ln[i]=f"Ctx-lh-{i-33}"
    for i in range(65,79): ln[i]=f"Ctx-rh-{i-64}"
    return {"adjacency_binary":adj,"adjacency_strength":strength,"hemisphere_map":hemi,"label_names":ln,"num_foreground":C}


HEMI_COLORS = {"left":"#2E75B6","right":"#E05A2B","midline":"#2CA02C","bilateral":"#7B4FB5","unknown":"#888888"}
HEMI_COLORS_DARK = {"left":"#1A4E80","right":"#A83A15","midline":"#1B7A1B","bilateral":"#553490","unknown":"#555555"}

# -----------------------------------------------------------------------------
# Visualization Logic (Clean Version)
# -----------------------------------------------------------------------------
def create_figure(prior, output_path="atlas_prior_clean.pdf", dpi=300):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.gridspec as gridspec
    import matplotlib.patheffects as pe
    import networkx as nx

    # --- Huge Fonts Setup ---
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial','Helvetica','DejaVu Sans'],
        'font.size': 24,
        'axes.linewidth': 2.0,
        'axes.edgecolor': '#444',
        'xtick.major.width': 1.5, 'ytick.major.width': 1.5,
        'xtick.major.size': 8, 'ytick.major.size': 8,
    })

    adj = prior["adjacency_binary"]; strength = prior["adjacency_strength"]
    hemi_map = prior.get("hemisphere_map",{}); label_names = prior.get("label_names",{})
    C = adj.shape[0]

    cmap_bin = LinearSegmentedColormap.from_list('vb',['#F7F9FC','#A4C4E0','#2E75B6','#14375E'],N=256)

    # --- Layout ---
    # Top margin reduced since titles are gone (top=0.98)
    fig = plt.figure(figsize=(24, 12))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.4],
                           wspace=0.15, left=0.05, right=0.98, top=0.98, bottom=0.08)

    bounds = [17.5, 32.5, 63.5]
    tpos = [0,17,32,63,77]; tlab = ["1","18","33","64","78"]
    rlabels = [(8.5,"L-Sub"),(25,"R-Sub"),(48,"L-Ctx"),(70.5,"R-Ctx")]

    def style_ax(ax, annot=True):
        for b in bounds:
            ax.axhline(b,color='#999',lw=1.0,alpha=0.5,ls=':')
            ax.axvline(b,color='#999',lw=1.0,alpha=0.5,ls=':')
        ax.set_xticks(tpos); ax.set_xticklabels(tlab, fontsize=20, fontweight='bold')
        ax.set_yticks(tpos); ax.set_yticklabels(tlab, fontsize=20, fontweight='bold')
        # REMOVED: set_xlabel("ROI index")
        # REMOVED: set_ylabel("ROI index")
        
        if annot:
            for yp,txt in rlabels:
                ax.annotate(txt, xy=(C+1.5,yp), fontsize=20, va='center', color='#666', style='italic', annotation_clip=False)
        for sp in ax.spines.values(): sp.set_linewidth(1.5); sp.set_color('#555')

    # ═══ (a) Binary Adjacency (CLEAN) ═══
    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(adj, cmap=cmap_bin, aspect='equal', interpolation='nearest', vmin=0, vmax=1)
    style_ax(ax0, annot=True)
    # REMOVED: ax0.set_title(...)
    # REMOVED: ax0.text(..., 'a', ...)

    # ═══ (b) Graph (CLEAN) ═══
    ax1 = fig.add_subplot(gs[1])

    G = nx.Graph()
    for i in range(C): G.add_node(i, hemisphere=hemi_map.get(i+1,"unknown"))
    for i in range(C):
        for j in range(i+1,C):
            if adj[i,j]>0: G.add_edge(i,j,weight=float(strength[i,j]))

    np.random.seed(42)
    init_pos = {}
    for i in range(C):
        h = hemi_map.get(i+1,"unknown")
        if h=="left":     init_pos[i] = (-1.3+np.random.randn()*0.25, np.random.randn()*0.5)
        elif h=="right":  init_pos[i] = ( 1.3+np.random.randn()*0.25, np.random.randn()*0.5)
        elif h=="midline": init_pos[i] = (np.random.randn()*0.1, np.random.randn()*0.5)
        else:              init_pos[i] = (np.random.randn()*0.35, np.random.randn()*0.5)

    pos = nx.spring_layout(G, pos=init_pos, k=0.55, iterations=120, seed=42)

    ax_ = np.array([pos[i][0] for i in range(C)]); ay_ = np.array([pos[i][1] for i in range(C)])
    cx_,cy_ = np.mean(ax_), np.mean(ay_); sx_,sy_ = np.std(ax_), np.std(ay_)
    for i in range(C):
        x,y = pos[i]
        if abs(x-cx_)>2.2*sx_ or abs(y-cy_)>2.2*sy_:
            pos[i] = (0.6*x+0.4*cx_, 0.6*y+0.4*cy_)

    intra, cross = [], []
    for u,v in G.edges():
        hu,hv = hemi_map.get(u+1,"unknown"), hemi_map.get(v+1,"unknown")
        if (hu=="left" and hv=="right") or (hu=="right" and hv=="left"): cross.append((u,v))
        else: intra.append((u,v))

    for u,v in intra:
        w = G[u][v].get('weight',0.1)
        ax1.plot([pos[u][0],pos[v][0]], [pos[u][1],pos[v][1]],
                 '-', color='#999', lw=0.6+1.5*w, alpha=0.3+0.35*w, zorder=1)
    for u,v in cross:
        w = G[u][v].get('weight',0.1)
        ax1.plot([pos[u][0],pos[v][0]], [pos[u][1],pos[v][1]],
                 '-', color='#D62728', lw=1.5+2.0*w, alpha=0.8, zorder=1.5)

    degs = dict(G.degree()); mx_d = max(degs.values()) if degs else 1
    for i in range(C):
        h = hemi_map.get(i+1,"unknown")
        sz = 80 + 600*(degs[i]/mx_d)
        ax1.scatter(pos[i][0], pos[i][1], s=sz,
                    c=HEMI_COLORS.get(h,"#888"), edgecolors=HEMI_COLORS_DARK.get(h,"#555"),
                    linewidths=1.5, zorder=3, alpha=0.95)

    for ni, txt in {0:"CWM-lh", 18:"CWM-rh", 11:"Brain Stem", 12:"Hipp-lh", 27:"Hipp-rh"}.items():
        if ni < C and ni in pos:
            ax1.annotate(txt, pos[ni], fontsize=22, fontweight='bold', ha='center', va='bottom',
                         xytext=(0,12), textcoords='offset points', color='#000',
                         path_effects=[pe.withStroke(linewidth=4, foreground='white')], zorder=5)

    # Midline dashed line (Keep line, REMOVE text)
    ymin = min(pos[i][1] for i in range(C)); ymax = max(pos[i][1] for i in range(C))
    ax1.plot([0,0],[ymin-0.08,ymax+0.08],'--',color='#AAA',lw=1.5,alpha=0.6,zorder=0)
    # REMOVED: ax1.text(0, ymax+0.16, 'midline', ...) 
    
    # REMOVED: ax1.set_title(...)
    # REMOVED: ax1.text(..., 'b', ...)
    ax1.set_aspect('equal'); ax1.axis('off')

    # Legend
    leg_el = [
        Line2D([0],[0],marker='o',color='w',markerfacecolor=HEMI_COLORS["left"],markeredgecolor=HEMI_COLORS_DARK["left"],markersize=18,markeredgewidth=1.5,label='Left hemi'),
        Line2D([0],[0],marker='o',color='w',markerfacecolor=HEMI_COLORS["right"],markeredgecolor=HEMI_COLORS_DARK["right"],markersize=18,markeredgewidth=1.5,label='Right hemi'),
        Line2D([0],[0],marker='o',color='w',markerfacecolor=HEMI_COLORS["midline"],markeredgecolor=HEMI_COLORS_DARK["midline"],markersize=18,markeredgewidth=1.5,label='Midline'),
        Line2D([0],[0],marker='o',color='w',markerfacecolor=HEMI_COLORS["bilateral"],markeredgecolor=HEMI_COLORS_DARK["bilateral"],markersize=18,markeredgewidth=1.5,label='Bilateral'),
        Line2D([0],[0],color='#999',lw=4,label='Intra-hemi'),
        Line2D([0],[0],color='#D62728',lw=4,label='Cross-hemi'),
    ]
    leg = ax1.legend(handles=leg_el, loc='lower center', bbox_to_anchor=(0.5,-0.12),
                     fontsize=22, framealpha=0.95, edgecolor='#CCC', ncol=3,
                     handletextpad=0.5, columnspacing=1.5)
    leg.get_frame().set_linewidth(1.0)

    for ext in ['pdf','png','svg']:
        out = output_path.replace('.pdf',f'.{ext}')
        fig.savefig(out, dpi=dpi, bbox_inches='tight', facecolor='white', transparent=(ext=='svg'))
        print(f"✓ {out}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior",type=str,default=None)
    parser.add_argument("--synthetic",action="store_true")
    parser.add_argument("--output",type=str,default="atlas_prior_figure_v2.pdf")
    parser.add_argument("--dpi",type=int,default=300)
    args = parser.parse_args()
    if args.prior: prior = load_prior(args.prior)
    elif args.synthetic: prior = make_synthetic_prior()
    else:
        try: prior = load_prior("atlas_prior_78class.pt")
        except: prior = make_synthetic_prior()
    create_figure(prior, args.output, args.dpi)