"""Run-weighted descriptive statistics; no cross-condition layer pooling."""
import numpy as np
import pandas as pd
from loader import SCOPE,LOCAL,LAYERS,TOPOLOGIES,PERSISTENCES

CONDITION=['topology_mode','persistence','load_users']
METRICS=SCOPE+LOCAL+['task_failure_rate']


def condition_groups(df,extra=()):
    faults=df[df.detection_label==1]
    keys=CONDITION+list(extra)
    for values,sub in faults.groupby(keys,dropna=False,sort=True):
        yield dict(zip(keys,values)),sub


def descriptive_statistics(df,extra=()):
    rows=[]
    for meta,sub in condition_groups(df,extra):
        for layer in LAYERS:
            group=sub[sub.layer==layer]
            for metric in METRICS:
                values=group[metric].dropna()
                rows.append({**meta,'layer':layer,'metric':metric,'n_runs':len(group),'n_valid':len(values),
                             'mean':values.mean(),'std':values.std(ddof=1),'median':values.median(),
                             'q25':values.quantile(.25),'q75':values.quantile(.75),
                             'iqr':values.quantile(.75)-values.quantile(.25),'min':values.min(),'max':values.max()})
    return pd.DataFrame(rows)


def markdown_table(headers,rows):
    esc=lambda v:str(v).replace('|',r'\|').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(map(esc,headers))+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(esc,row))+' |' for row in rows])


def formatted_statistics(summary):
    labels=['Services','Tasks','Task ratio','Breadth*','Node error','Link error','Mean span (s)','P95 span (s)','Retry','Task failure']
    blocks=['# 实验一：条件内F0–F4异常表现统计\n\n单元格为运行等权均值 ± 样本SD（ddof=1）；n=1时SD记N/A，缺测不填0。百分比量显示为百分数。中位数、Q25/Q75、IQR、有效n保存在layer_statistics.csv。未记任务配置的common单独保留，不能解释为条件已完全控制。\n']
    for meta,sub in condition_groups_from_summary(summary):
        rows=[]
        for layer in LAYERS:
            s=sub[sub.layer==layer].set_index('metric');cells=[]
            for metric in METRICS:
                r=s.loc[metric]
                if not r.n_valid:cells.append('N/A');continue
                pct=metric.endswith(('rate','ratio'))
                mean=f'{r["mean"]:.1%}' if pct else f'{r["mean"]:.2f}'
                sd=(f'{r["std"]:.1%}' if pct else f'{r["std"]:.2f}') if np.isfinite(r['std']) else 'N/A'
                cell=f'{mean} ± {sd}'
                if r.n_valid<r.n_runs:cell+=f' [n={int(r.n_valid)}]'
                cells.append(cell)
            rows.append([layer,int(s.n_runs.iloc[0]),*cells])
        blocks.append('## '+' × '.join(str(meta[k]) for k in CONDITION)+'\n\n'+markdown_table(['层次','n',*labels],rows))
    blocks.append('\n* Breadth为记录执行图传播代理，非已核验因果路径。局部节点/链路指执行顶点/依赖边。各条件单独统计，表中离散性主要来自同层不同类型/配置，而非相同故障重复注入的置信区间。')
    return '\n\n'.join(blocks)+'\n'


def condition_groups_from_summary(summary):
    for values,sub in summary.groupby(CONDITION,sort=True,dropna=False):yield dict(zip(CONDITION,values)),sub
