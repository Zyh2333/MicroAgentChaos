#!/usr/bin/env python3
"""Motivation 2: condition-specific statistics and abnormal-service clustering."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
import scipy
import matplotlib
import loader as L
from descriptive import descriptive_statistics,formatted_statistics
from clustering import (eligible_runs,service_features,fit_services,composition,metadata_diagnostics,conditional_diagnostics,
                        entropy,K_VALUES,BOOTSTRAPS,STABILITY_GATE,SEED,SERVICE_PREFIX,COUNT_FEATURE)
from plots import symptom_pca,layer_composition
from report import write_readme

HERE=Path(__file__).resolve().parent
TAB,FIG=HERE/'tables',HERE/'figures'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extra-data-root',action='append',default=[],help='Additional labelled collection roots.')
    args=parser.parse_args();TAB.mkdir(exist_ok=True);FIG.mkdir(exist_ok=True)
    df,tasks,hits,audit=L.build_dataframe(args.extra_data_root)
    for name,frame in [('run_metrics',df),('task_metrics',tasks),('verified_window_hits',hits),('source_audit',audit)]:
        frame.to_csv(TAB/f'{name}.csv',index=False)
    summary=descriptive_statistics(df);summary.to_csv(TAB/'layer_statistics.csv',index=False)
    for name,extra in [('by_workload',['workload_id','edge_deployment_id']),('by_task_class',['task_class_profile'])]:
        descriptive_statistics(df,extra).to_csv(TAB/f'layer_statistics_{name}.csv',index=False)
    (HERE/'layer_statistics_table.md').write_text(formatted_statistics(summary))
    eligible,exclusions=eligible_runs(df)
    exclusions.to_csv(TAB/'clustering_exclusions.csv',index=False)
    # This is the only object passed to fitting/K selection. It has no labels or metadata.
    feature_frame,vocabulary=service_features(eligible)
    features=list(feature_frame.columns)
    pd.concat([eligible[['sample_id']],feature_frame],axis=1).to_csv(TAB/'service_feature_matrix.csv',index=False)
    transformed,candidates,boots,details=fit_services(feature_frame)
    assignments=pd.concat([eligible,feature_frame,transformed],axis=1)
    assignments.insert(0,'sample_index',np.arange(len(assignments)))
    assignments.to_csv(TAB/'cluster_assignments.csv',index=False)
    candidates.to_csv(TAB/'cluster_candidates.csv',index=False)
    boots.to_csv(TAB/'bootstrap_stability.csv',index=False)
    comp,entropies=composition(assignments)
    associations,covariates=metadata_diagnostics(assignments)
    conditional,spreads=conditional_diagnostics(assignments)
    profiles=[]
    for cluster,s in assignments.groupby('cluster'):
        for feature in features:
            x=s[feature]
            profiles.append({'cluster':cluster,'feature':feature,'n_runs':len(x),'mean':x.mean(),'std':x.std(ddof=1),
                             'median':x.median(),'q25':x.quantile(.25),'q75':x.quantile(.75),'iqr':x.quantile(.75)-x.quantile(.25)})
    prevalence=[]
    for cluster,s in assignments.groupby('cluster'):
        for service in vocabulary:
            n=int(s[SERVICE_PREFIX+service].sum())
            prevalence.append({'cluster':cluster,'service':service,'n_affected_runs':n,'n_cluster':len(s),
                               'proportion_within_cluster':n/len(s)})
    pd.DataFrame(prevalence).to_csv(TAB/'cluster_service_prevalence.csv',index=False)
    for name,frame in [('cluster_layer_composition',comp),('cluster_layer_entropy',entropies),
                       ('cluster_covariate_association',associations),('cluster_covariate_composition',covariates),
                       ('conditional_cluster_association',conditional),('condition_layer_cluster_spread',spreads),
                       ('cluster_profiles',pd.DataFrame(profiles))]:frame.to_csv(TAB/f'{name}.csv',index=False)
    selected=candidates[candidates.selected].iloc[0]
    weighted_h=float(np.average(entropies.entropy_bits,weights=entropies.n_runs))
    metrics={'n_fault_runs':int((df.detection_label==1).sum()),'n_cluster_runs':len(assignments),
             'layer_counts':{k:int((assignments.layer==k).sum()) for k in L.LAYERS},
             'n_clusters':details['selected_k'],'silhouette':float(selected.silhouette),
             'n_unique_service_patterns':len(feature_frame.drop_duplicates()),
             'bootstrap_stability_mean_ari':float(selected.stability_mean_ari),
             'bootstrap_stability_std_ari':float(selected.stability_std_ari),
             'layer_ami':float(associations[associations.factor=='layer'].ami.iloc[0]),
             'weighted_cluster_entropy_bits':weighted_h,'overall_layer_entropy_bits':entropy(assignments.layer),
             'causal_attribution_verified':False,'online_snapshot_verified':False,
             'n_condition_strata_with_mixed_layer_cluster':int(((conditional.granularity=='condition')&(conditional.n_mixed_layer_clusters>0)).sum()),
             'n_known_condition_strata_with_mixed_layer_cluster':int(((conditional.granularity=='condition')&conditional.metadata_complete&(conditional.n_mixed_layer_clusters>0)).sum())}
    (TAB/'clustering_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    (TAB/'clustering_model.json').write_text(json.dumps(details,indent=2)+'\n')
    config={'research_question':'abnormal-service identity/scale patterns and fault-layer identifiability',
            'experiments':['within-condition layer descriptive statistics','unsupervised full-run abnormal-service set/count clustering'],
            'source_scope':'all labelled topology_runs plus legacy common after prior no-fault exclusions',
            'excluded_topology_run_ids':sorted(L.EXCLUDED_RUN_IDS),'exclude_proxy_total_fires_zero':True,
            'normal_control_available':bool((df.detection_label==0).any()),
            'sample_unit':'one fault collection run; tasks are not independent clustering samples',
            'features':features,'service_vocabulary':vocabulary,'observation_window':'complete collected execution graphs',
            'input_quality':'complete graph task count and consistent unique abnormal service set/count; no early-time restriction',
            'preprocessing':details['scaling'],'fixed_scale_weights':details['fixed_scale_weights'],
            'count_redundancy':'count is the sum of binary membership; explicit extra scale weight, not independent evidence',
            'algorithm':'AgglomerativeClustering, Ward linkage, Euclidean metric in service-membership plus count space',
            'candidate_k':K_VALUES,'bootstrap_repeats':BOOTSTRAPS,'seed':SEED,'stability_gate':STABILITY_GATE,
            'selection_rule':details['selection_rule'],'layer_labels_used_for_k_selection':False,
            'bootstrap_unit':'run, sampled with replacement; fixed block scaling and refitted Ward; ARI on unique sampled original runs',
            'bootstrap_interpretation':'resampling sensitivity, not independent replications or a confidence interval',
            'pca':'2 dimensions for display only; clustering uses all scaled service-membership and count features',
            'confounding_checks':['topology_mode','persistence','load_users','fault_type','workload_id','edge_deployment_id','task_class_profile','run_id'],
            'scope_and_task_damage_role':'abnormal services/count are clustering inputs; other scope and task damage are descriptive; no severity labels',
            'causal_attribution_verified':False,'online_snapshot_verified':False,
            'graph_availability_assumption':'whole-run graph errors may include background or off-window anomalies; causal propagation not verified',
            'input_fingerprint_sha256':hashlib.sha256(assignments[['sample_id']+features].to_csv(index=False,float_format='%.17g').encode()).hexdigest(),
            'versions':{'sklearn':sklearn.__version__,'scipy':scipy.__version__,'numpy':np.__version__,
                        'pandas':pd.__version__,'matplotlib':matplotlib.__version__},'extra_data_roots':args.extra_data_root}
    (HERE/'experiment_config.json').write_text(json.dumps(config,indent=2,ensure_ascii=False)+'\n')
    symptom_pca(assignments,details,FIG/'fig3_affected_service_clusters')
    layer_composition(comp,FIG/'fig4_cluster_layer_composition')
    write_readme(df,summary,assignments,candidates,comp,entropies,associations,conditional,spreads,metrics,audit)
    print(candidates.to_string(index=False));print(json.dumps(metrics,indent=2));print(associations.to_string(index=False))


if __name__=='__main__':main()
