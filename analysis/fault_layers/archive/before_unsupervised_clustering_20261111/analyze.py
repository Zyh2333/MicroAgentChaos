#!/usr/bin/env python3
"""Motivation 2: all fault runs, condition-specific summaries and RF ablations."""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

import loader as L
from evaluate import evaluate,MODELS,SEED,MAX_FEATURES,N_REPEATS,TEST_FRACTION,conditions,CONDITION
from scope_labels import EARLY_SECONDS,SERVICE_CUTS,suggest_service_cuts
from report import write_readme

HERE=Path(__file__).resolve().parent
FIG,TAB=HERE/'figures',HERE/'tables'
SCOPE_LABELS=['Affected\nServices','Affected\nTasks','Affected\nTask Ratio','Propagation\nBreadth*']
RATES=['node_error_rate','link_error_rate','task_failure_rate']
RATE_LABELS=['Node Error Rate','Link Error Rate','Task Failure Rate']
MODE_LABEL={'cloud-preference':'Cloud-first','edge-preference':'Edge-first','dynamic':'Dynamic','unrecorded':'Unrecorded'}
matplotlib.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
                           'font.size':11,'pdf.fonttype':42,'ps.fonttype':42,'axes.unicode_minus':False})


def save(fig,name):
    fig.savefig(FIG/f'{name}.pdf')
    fig.savefig(FIG/f'{name}.png',dpi=350)
    plt.close(fig)


def layer_summary(df):
    rows=[]
    for meta,sub,_ in conditions(df):
        for layer in L.LAYERS:
            layer_data=sub[sub.layer==layer]
            for metric in L.SCOPE+RATES:
                values=layer_data[metric].dropna()
                rows.append({**meta,'layer':layer,'metric':metric,'n_runs':len(layer_data),
                             'n_valid':len(values),'mean':values.mean(),'std':values.std(ddof=1),
                             'min':values.min(),'max':values.max()})
    return pd.DataFrame(rows)


def panel_layout(summary):
    known=summary[summary.topology_mode!='unrecorded']
    row_keys=list(known[['load_users','persistence']].drop_duplicates().itertuples(index=False,name=None))
    row_keys.sort(key=lambda k:(k[0],L.PERSISTENCES.index(k[1])))
    cols=[t for t in L.TOPOLOGIES if t!='unrecorded' and t in set(known.topology_mode)]
    return known,row_keys,cols


def fig_scope(summary):
    known,row_keys,cols=panel_layout(summary)
    limits={k:(known[known.metric==k]['mean'].min(),known[known.metric==k]['mean'].max()) for k in L.SCOPE}
    pd.DataFrame([{'metric':k,'column_min_mean':v[0],'column_max_mean':v[1]} for k,v in limits.items()]).to_csv(TAB/'scope_normalization.csv',index=False)
    normalized=[]
    for r in known[known.metric.isin(L.SCOPE)].to_dict('records'):
        low,high=limits[r['metric']]
        r['normalized_mean']=(r['mean']-low)/(high-low) if high>low else (0 if np.isfinite(r['mean']) else np.nan)
        normalized.append(r)
    pd.DataFrame(normalized).to_csv(TAB/'scope_column_normalized.csv',index=False)
    cmap,norm=plt.get_cmap('Blues'),Normalize(0,1)
    fig,axes=plt.subplots(len(row_keys),len(cols),figsize=(13,3.05*len(row_keys)+.8),squeeze=False)
    for i,(load,persistence) in enumerate(row_keys):
        for j,topology in enumerate(cols):
            ax=axes[i,j]
            sub=known[(known.topology_mode==topology)&(known.persistence==persistence)&(known.load_users==load)]
            matrix=sub.pivot(index='layer',columns='metric',values='mean').reindex(index=L.LAYERS,columns=L.SCOPE)
            counts=sub[sub.metric==L.SCOPE[0]].set_index('layer').n_runs.reindex(L.LAYERS).fillna(0)
            for r,layer in enumerate(L.LAYERS):
                for c,metric in enumerate(L.SCOPE):
                    value=matrix.loc[layer,metric];low,high=limits[metric]
                    z=(value-low)/(high-low) if high>low else 0
                    ax.add_patch(Rectangle((c-.5,r-.5),1,1,facecolor=cmap(norm(z)) if np.isfinite(value) else '#e6e6e6',edgecolor='white',linewidth=1))
                    label='N/A' if not np.isfinite(value) else (f'{value:.1%}' if metric.endswith('ratio') else f'{value:.1f}')
                    ax.text(c,r,label,ha='center',va='center',fontsize=11,color='white' if np.isfinite(value) and z>.62 else '#222222')
            ax.set(xlim=(-.5,3.5),ylim=(4.5,-.5),xticks=range(4),yticks=range(5))
            ax.set_xticklabels(SCOPE_LABELS,fontsize=10)
            ax.set_yticklabels([f'{l} [{int(counts[l])}]' for l in L.LAYERS],fontsize=10)
            ax.set_title(f'{MODE_LABEL[topology]} | {persistence.capitalize()} | load {load}',fontsize=12,pad=8)
            ax.tick_params(length=0,pad=5)
            for spine in ax.spines.values():spine.set_visible(False)
    fig.subplots_adjust(left=.07,right=.91,bottom=.11,top=.96,hspace=.58,wspace=.36)
    cax=fig.add_axes([.94,.19,.014,.64])
    fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=cax,label='Metric-wise normalized mean')
    fig.text(.07,.052,'Cells: raw run means; [n]: runs per layer. Normalization is shared across all panels for each metric.',fontsize=11)
    fig.text(.07,.022,'* Recorded-graph propagation proxy. Unrecorded-topology common runs are reported separately in CSV tables.',fontsize=11)
    save(fig,'fig1_impact_scope_by_layer')


def fig_severity(summary):
    known,row_keys,cols=panel_layout(summary)
    fig,axes=plt.subplots(len(row_keys),len(cols),figsize=(13,3*len(row_keys)+.8),squeeze=False,sharey=True)
    x,width=np.arange(5),.24
    for i,(load,persistence) in enumerate(row_keys):
        for j,topology in enumerate(cols):
            ax=axes[i,j]
            sub=known[(known.topology_mode==topology)&(known.persistence==persistence)&(known.load_users==load)]
            for offset,(metric,label,color) in enumerate(zip(RATES,RATE_LABELS,['#56B4E9','#009E73','#D55E00'])):
                s=sub[sub.metric==metric].set_index('layer').reindex(L.LAYERS)
                ax.bar(x+(offset-1)*width,s['mean'],width,yerr=s['std'].fillna(0),color=color,label=label,capsize=2,error_kw={'linewidth':.8},zorder=3)
                if metric==RATES[0]:
                    for pos,l in enumerate(L.LAYERS):
                        if pd.isna(s.loc[l,'mean']):ax.text(pos,.055,'N/A',ha='center',fontsize=9,color='#777777',rotation=90)
            ax.set(xticks=x,xticklabels=L.LAYERS,ylim=(0,1.05))
            ax.set_title(f'{MODE_LABEL[topology]} | {persistence.capitalize()} | load {load}',fontsize=12,pad=8)
            ax.tick_params(labelsize=10)
            if j==0:ax.set_ylabel('Rate',fontsize=11)
            ax.grid(axis='y',color='#ebebeb',zorder=0)
            ax.spines[['top','right']].set_visible(False)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,frameon=False,ncol=3,loc='upper center',bbox_to_anchor=(.5,.995),fontsize=12)
    fig.subplots_adjust(left=.065,right=.98,bottom=.095,top=.92,hspace=.43,wspace=.22)
    fig.text(.065,.04,'Bars: within-condition run means; error bars: between-run SD. SD is undefined for n=1 (no interval shown).',fontsize=11)
    fig.text(.065,.014,'Node/link refer to recorded execution vertices/dependencies; Task Failure Rate remains a whole-run descriptive measure.',fontsize=11)
    save(fig,'fig2_severity_by_layer')


def fig_evaluation(result,task,repeat_metrics):
    sub=result[result.task==task]
    if sub.empty:return False
    metrics=['Precision','Recall','F1','AUPRC'] if task=='detection' else ['Macro-F1','QWK']
    keys=list(sub.stratum_id.unique());nrows=len(keys)
    fig,axes=plt.subplots(nrows,2,figsize=(10.8,3.65*nrows+1.0),squeeze=False)
    colors=['#0072B2','#E69F00'];rng=np.random.default_rng(SEED)
    for row_index,key in enumerate(keys):
        current=sub[sub.stratum_id==key];record=current.iloc[0];x=np.arange(len(metrics));width=.32
        left,right=axes[row_index]
        for j,(method,color) in enumerate(zip(MODELS,colors)):
            s=current[current.method==method].set_index('metric').loc[metrics];positions=x+(j-.5)*width
            left.bar(positions,s['mean'],width,yerr=s['std'],color=color,label=method,capsize=3,
                     alpha=.85,error_kw={'linewidth':1},zorder=3)
            right.bar(positions,s.variance,width,color=color,label=method,zorder=3)
            for pos,metric in zip(positions,metrics):
                values=repeat_metrics[(repeat_metrics.task==task)&(repeat_metrics.stratum_id==key)&
                                      (repeat_metrics.method==method)&(repeat_metrics.metric==metric)].sort_values('repeat').value
                left.scatter(pos+rng.uniform(-.045,.045,len(values)),values,s=15,c='#222222',alpha=.45,zorder=5)
                item=s.loc[metric]
                left.annotate(f'{item["mean"]:.3f}',(pos,item['mean']+item['std']),xytext=(0,5),
                              textcoords='offset points',ha='center',fontsize=10)
                right.annotate(f'{item.variance:.4f}',(pos,item.variance),xytext=(0,5),
                               textcoords='offset points',ha='center',fontsize=10)
        condition=(f'Pooled strategies & fault modes | load {record.load_users}' if record.analysis_scope=='pooled'
                   else f'{MODE_LABEL[record.topology_mode]} | {record.persistence.capitalize()} | load {record.load_users}')
        left.set_title(f'{condition}\n(a) Mean ± SD; {record.n_repeats} repeats; n={record.n_samples}',fontsize=12,pad=10)
        right.set_title(f'{condition}\n(b) Sample variance; {record.n_repeats} repeats',fontsize=12,pad=10)
        values=repeat_metrics[(repeat_metrics.task==task)&(repeat_metrics.stratum_id==key)].value
        low=min(0,float(values.min()),float((current['mean']-current['std']).min()))
        high=max(1.05,float((current['mean']+current['std']).max())+.10)
        left.set(xticks=x,xticklabels=metrics,ylabel='Mean score',ylim=(max(-1,low-.06) if low<0 else 0,high))
        right.set(xticks=x,xticklabels=metrics,ylabel='Sample variance (score²)',
                  ylim=(0,max(.01,float(current.variance.max())*1.4)))
        for ax in [left,right]:
            ax.grid(axis='y',color='#ebebeb',zorder=0);ax.spines[['top','right']].set_visible(False)
            ax.axhline(0,color='#888888',lw=.6)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,frameon=False,ncol=2,loc='upper center',bbox_to_anchor=(.5,.995),fontsize=12)
    fig.subplots_adjust(left=.08,right=.98,bottom=.12 if nrows>1 else .23,
                        top=.85 if nrows>1 else .78,hspace=.65,wspace=.3)
    fig.text(.08,.055 if nrows>1 else .09,'Paired random group holdouts: same split for both methods; model seed fixed; dots show individual repeats.',fontsize=10)
    fig.text(.08,.022 if nrows>1 else .04,'Variance uses ddof=1; error bars show SD, not variance. Re-splitting does not add independent samples.',fontsize=10)
    save(fig,'fig3_fault_detection' if task=='detection' else 'fig4_impact_scope_grading')
    return True


def retire_missing_figure(task):
    names=['fig3_fault_detection'] if task=='detection' else ['fig4_impact_grading','fig4_impact_scope_grading']
    for name in names:
        for ext in ['.pdf','.png']:
            source=FIG/(name+ext)
            if source.exists():
                dest=HERE/'archive/superseded_grading_figures'/source.name;dest.parent.mkdir(parents=True,exist_ok=True)
                number=1
                while dest.exists():
                    dest=dest.with_name(f'{source.stem}_{number}{source.suffix}');number+=1
                source.rename(dest)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extra-data-root',action='append',default=[],help='Additional labelled fault/normal data.')
    args=parser.parse_args();FIG.mkdir(exist_ok=True);TAB.mkdir(exist_ok=True)
    df,tasks,hits,audit=L.build_dataframe(args.extra_data_root)
    for name,frame in [('run_metrics',df),('task_metrics',tasks),('verified_window_hits',hits),('source_audit',audit)]:frame.to_csv(TAB/f'{name}.csv',index=False)
    identity=['sample_id','cohort',*CONDITION,'layer','fault_type','cv_group']
    label_columns=[c for c in df if c.startswith('scope_')]
    df[identity+label_columns].to_csv(TAB/'scope_labels.csv',index=False)
    df[identity+[c for c in df if c.startswith('early_')]].to_csv(TAB/'early_run_metrics.csv',index=False)
    entities=[]
    for r in df.itertuples():
        for task in json.loads(r.scope_boundary_evidence_json):
            for service in task['services']:
                entities.append({'sample_id':r.sample_id,'task_id':task['task_id'],'actual_error_service':service,
                                 'graph_file':str(Path(r.sample_id)/'agent-network/graph/call_chains.json'),
                                 'observed_error_vertices_json':json.dumps(task['error_vertices']),
                                 'cross_service_paths_json':json.dumps(task['cross_service_paths']),
                                 'causal_attribution':'unverified_observed_process_error'})
    pd.DataFrame(entities,columns=['sample_id','task_id','actual_error_service','graph_file','observed_error_vertices_json',
                                  'cross_service_paths_json','causal_attribution']).to_csv(TAB/'scope_label_entities.csv',index=False)
    df.groupby(CONDITION+['fault_type','scope_grade_name'],dropna=False).size().rename('n_runs').reset_index().to_csv(TAB/'scope_grade_distribution.csv',index=False)
    frequency=df[df.detection_label==1].groupby('scope_label_service_count').size().rename('n_runs').reset_index()
    frequency['proportion']=frequency.n_runs/frequency.n_runs.sum()
    frequency['cumulative_proportion']=frequency.proportion.cumsum()
    frequency.to_csv(TAB/'affected_service_count_distribution.csv',index=False)
    candidate_cuts=suggest_service_cuts(df.loc[df.detection_label==1,'scope_label_service_count'])
    (TAB/'service_grade_thresholds.json').write_text(json.dumps({'frozen_cuts':SERVICE_CUTS,
        'selection_rule':'minimize sum((n_class-n/3)^2) over integer service-count cuts; never split ties; no RF score used',
        'reference_cohort':'initial 111 filtered fault runs; frozen at [4,8]',
        'current_cohort_frequency_check':candidate_cuts},indent=2)+'\n')
    summary=layer_summary(df)
    figures_unchanged=False
    if (TAB/'layer_summary.csv').exists() and all((FIG/f'{name}.{ext}').exists() for name in ['fig1_impact_scope_by_layer','fig2_severity_by_layer'] for ext in ['png','pdf']):
        try:
            pd.testing.assert_frame_equal(summary,pd.read_csv(TAB/'layer_summary.csv'),check_dtype=False,check_exact=False,rtol=1e-12,atol=1e-12)
            figures_unchanged=True
        except AssertionError:pass
    summary.to_csv(TAB/'layer_summary.csv',index=False)
    if not figures_unchanged:fig_scope(summary);fig_severity(summary)
    result,predictions,folds,exclusions,status=evaluate(df)
    pairs=pd.DataFrame(status.pop('paired_deltas'));cases=pd.DataFrame(status.pop('paired_cases'))
    trials=pd.DataFrame(status.pop('repeat_metrics'));trial_deltas=pd.DataFrame(status.pop('repeat_deltas'))
    meta=['stratum_id',*CONDITION,'analysis_scope','task']
    if pairs.empty:pairs=pd.DataFrame(columns=meta+['metric','mean_delta','variance_delta','std_delta','min_delta','max_delta','n_repeats','n_samples'])
    if cases.empty:cases=pd.DataFrame(columns=meta+['repeat','split_seed','model_seed','sample_id','fault_type','y_true','baseline_prediction','scope_prediction'])
    if trials.empty:trials=pd.DataFrame(columns=meta+['repeat','split_seed','model_seed','method','metric','value','n_samples','n_train','n_test'])
    if trial_deltas.empty:trial_deltas=pd.DataFrame(columns=meta+['repeat','split_seed','model_seed','metric','scope_minus_baseline','n_train','n_test'])
    for name,frame in [('evaluation_metrics',result),('holdout_predictions',predictions),('repeat_splits',folds),
                       ('repeat_metrics',trials),('paired_scope_repeat_deltas',trial_deltas),
                       ('evaluation_exclusions',exclusions),('paired_scope_deltas',pairs),('paired_case_interventions',cases)]:
        frame.to_csv(TAB/f'{name}.csv',index=False)
    for name in ['cv_splits.csv','oof_predictions.csv']:
        source=TAB/name
        if source.exists():
            destination=HERE/'archive/retired_oof_tables'/name;destination.parent.mkdir(parents=True,exist_ok=True)
            number=1
            while destination.exists():
                destination=destination.with_name(f'{source.stem}_{number}{source.suffix}');number+=1
            source.rename(destination)
    coverage=pd.DataFrame(status['strata'])
    for key in ['class_counts','class_group_counts','observed_scope_counts_before_early_filter','train_sizes','test_sizes']:coverage[key]=coverage[key].map(lambda v:json.dumps(v,sort_keys=True))
    coverage.to_csv(TAB/'stratum_coverage.csv',index=False)
    (TAB/'evaluation_status.json').write_text(json.dumps(status,indent=2,ensure_ascii=False)+'\n')
    if any((FIG/f'fig4_impact_grading.{ext}').exists() for ext in ['pdf','png']):
        retire_missing_figure('grading')
    for task in ['detection','grading']:
        if not fig_evaluation(result,task,trials):retire_missing_figure(task)
    import sklearn
    config={'seed':SEED,'classifier':'RandomForestClassifier','n_estimators':300,'max_depth':4,'min_samples_leaf':2,
            'max_features':MAX_FEATURES,'max_features_semantics':'integer candidate feature count per split',
            'class_weight':'balanced','n_jobs':1,'features':MODELS,
            'condition_factors':CONDITION,'source_scope':'topology_runs and legacy common after explicit no-fault and total_fires=0 exclusions',
            'excluded_topology_run_ids':sorted(L.EXCLUDED_RUN_IDS),
            'exclude_proxy_total_fires_zero':True,
            'missing_proxy_stats_policy':'unknown_not_zero; retain with audit status',
            'grading_rules':{'low':'1 <= observed abnormal logical service count <= 4',
                             'medium':'5 <= observed abnormal logical service count <= 8',
                             'high':'observed abnormal logical service count >= 9'},
            'service_count_cuts':SERVICE_CUTS,'service_grade_cut_selection':candidate_cuts,
            'grading_label':'Impact Scope Grading; observed service-count bins, not verified causal ground truth',
            'scope_causal_ground_truth_verified':False,'physical_boundary_observed':False,
            'early_seconds':EARLY_SECONDS,'early_interval':'[fault_start, min(fault_start+30, fault_end))',
            'early_ratio_denominator':'tasks_completed_in_early_observation_cohort; not future submitted total',
            'feature_time_basis':'retrospective completed-task prefix; no final-run scope inputs',
            'graph_availability_assumption':'task span state is final by unique positive end marker; no timestamped snapshot verification',
            'completion_time_source':'unique_positive_PIPELINE_TIME_COST_TOTAL_marker_in_planner_logs',
            'propagation_basis':'first_observed_error_execution_vertices_proxy',
            'evaluation':'controlled conditions plus separately pooled strategies and fault modes, per load',
            'cv':'10 distinct paired random group holdouts; fault_type/control_collection never crosses train/test',
            'n_repeats':N_REPEATS,'test_group_fraction':TEST_FRACTION,
            'split_seed_sequence':'20261110 + candidate index; accept only distinct splits with all classes on both sides',
            'model_seed_fixed':SEED,
            'required_grade_classes':[0,1,2],'minimum_groups_per_class':2,
            'score_summary':'mean, sample variance and SD across 10 per-repeat scores; ddof=1; no pooled-OOF score',
            'sklearn_version':sklearn.__version__,'numpy_version':np.__version__,
            'pandas_version':pd.__version__,'matplotlib_version':matplotlib.__version__,'extra_data_roots':args.extra_data_root}
    (HERE/'experiment_config.json').write_text(json.dumps(config,indent=2,ensure_ascii=False)+'\n')
    write_readme(df,summary,result,status,audit)
    print(result.to_string(index=False));print(coverage[['stratum_id','task','status','n_eligible','reason']].to_string(index=False))


if __name__=='__main__':main()
