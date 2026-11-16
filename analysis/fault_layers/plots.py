"""Two paper figures: abnormal-service PCA and post-hoc layer composition."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
                     'font.size':11,'pdf.fonttype':42,'ps.fonttype':42,'axes.unicode_minus':False})
CLUSTER_COLORS=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00']
LAYER_COLORS={'F0':'#999999','F1':'#0072B2','F2':'#E69F00','F3':'#009E73','F4':'#CC79A7'}


def save(fig,path):
    fig.savefig(Path(path).with_suffix('.pdf'))
    fig.savefig(Path(path).with_suffix('.png'),dpi=350)
    plt.close(fig)


def symptom_pca(assignments,details,path):
    fig,ax=plt.subplots(figsize=(7.2,5.3))
    for i,(cluster,s) in enumerate(assignments.groupby('cluster',sort=True)):
        color=CLUSTER_COLORS[i]
        ax.scatter(s.pc1,s.pc2,s=35,c=color,alpha=.78,edgecolors='white',linewidths=.45,
                   label=f'Cluster {cluster} (n={len(s)})',zorder=3)
        center=s[['pc1','pc2']].mean()
        ax.scatter(center.pc1,center.pc2,marker='X',s=125,c=color,edgecolors='black',linewidths=.8,zorder=5)
        ax.annotate(f'C{cluster}',(center.pc1,center.pc2),xytext=(9,9),textcoords='offset points',fontsize=11,weight='bold')
    variance=details['pca_explained_variance_ratio']
    ax.set_xlabel(f'PC1 ({variance[0]:.1%} explained variance)')
    ax.set_ylabel(f'PC2 ({variance[1]:.1%} explained variance)')
    ax.set_title(f'Affected-service clusters | Ward, K={details["selected_k"]} | n={len(assignments)}',pad=12)
    ax.grid(color='#ececec',lw=.7,zorder=0);ax.spines[['top','right']].set_visible(False)
    ax.legend(frameon=False,loc='best',fontsize=10)
    fig.subplots_adjust(left=.13,right=.97,bottom=.22,top=.90)
    fig.text(.13,.11,'Binary service membership + service count; fixed block scaling; PCA for display only.',fontsize=9)
    fig.text(.13,.065,'One point/run (identical patterns overlap); X: projected cluster mean.',fontsize=9)
    note=('Exploratory solution: no candidate K passed the bootstrap stability gate (0.75).'
          if details['selection_rule'].startswith('no K met') else
          'K is selected without fault-layer labels; bootstrap stability is a resampling diagnostic.')
    fig.text(.13,.025,note,fontsize=9)
    save(fig,path)


def layer_composition(composition,path):
    clusters=sorted(composition.cluster.unique());fig,ax=plt.subplots(figsize=(6.8,5.1))
    bottoms=np.zeros(len(clusters));x=np.arange(len(clusters))
    for layer,color in LAYER_COLORS.items():
        s=composition[composition.layer==layer].set_index('cluster').loc[clusters]
        values=s.proportion_within_cluster.to_numpy()
        ax.bar(x,values,bottom=bottoms,width=.55,color=color,edgecolor='white',linewidth=.7,label=layer,zorder=3)
        for position,value,bottom,n in zip(x,values,bottoms,s.n_runs):
            if value>=.075:
                ax.text(position,bottom+value/2,f'{layer}: {int(n)}\n{value:.1%}',ha='center',va='center',fontsize=11,
                        color='white' if layer in ['F1','F3'] else '#222222')
        bottoms+=values
    totals=composition.groupby('cluster').n_cluster.first().reindex(clusters)
    ax.set(xticks=x,xticklabels=[f'Cluster {c}\n(n={int(totals.loc[c])})' for c in clusters],
           ylim=(0,1),ylabel='Fault-layer proportion within cluster')
    ax.yaxis.set_major_formatter(PercentFormatter(1));ax.set_title('Fault-layer composition of affected-service clusters',pad=12)
    ax.grid(axis='y',color='#ececec',zorder=0);ax.spines[['top','right']].set_visible(False)
    ax.legend(title='Injected layer',frameon=False,ncol=5,loc='upper center',bbox_to_anchor=(.5,-.16),fontsize=10,title_fontsize=10)
    fig.subplots_adjust(left=.14,right=.97,bottom=.30,top=.90)
    f0_n=int(composition.loc[composition.layer=='F0','n_runs'].sum())
    note=(f'F0: {f0_n} eligible runs; layer counts describe the observed sample only.' if f0_n else
          'F0 has no eligible early observations; zero counts do not imply zero fault impact.')
    fig.text(.14,.045,note,fontsize=9)
    save(fig,path)
