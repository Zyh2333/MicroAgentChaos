"""Label-blind affected-service set/count clustering and metadata diagnostics."""
import json
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score,adjusted_rand_score,adjusted_mutual_info_score
from loader import LAYERS
from descriptive import CONDITION

SEED=20261111
K_VALUES=(2,3,4,5)
BOOTSTRAPS=100
STABILITY_GATE=.75
MIN_ASSOCIATION_N=20
PERMUTATIONS=999
SERVICE_PREFIX='service__'
COUNT_FEATURE='abnormal_service_count'


def eligible_runs(df):
    reasons=[]
    for r in df.itertuples():
        bad=[]
        if r.detection_label!=1:bad.append('not_fault_run')
        if r.n_submitted!=r.n_graph_tasks or not r.n_graph_tasks:bad.append('incomplete_graph_task_coverage')
        try:
            services=json.loads(r.affected_services_json)
            valid=(isinstance(services,list) and all(isinstance(s,str) and s.startswith('agent-network-') for s in services)
                   and len(services)==len(set(services)) and np.isfinite(r.affected_services)
                   and len(services)==r.affected_services)
        except (ValueError,TypeError,AttributeError):valid=False
        if not valid:bad.append('invalid_or_inconsistent_abnormal_service_set')
        if bad:reasons.append({'sample_id':r.sample_id,'layer':r.layer,'reason':';'.join(bad)})
    rejected=pd.DataFrame(reasons,columns=['sample_id','layer','reason'])
    return df[~df.sample_id.isin(rejected.sample_id)].copy().sort_values('sample_id').reset_index(drop=True),rejected


def service_features(df):
    sets=[set(json.loads(value)) for value in df.affected_services_json]
    vocabulary=sorted(set().union(*sets)) if sets else []
    if not vocabulary:raise ValueError('No observed abnormal service vocabulary to cluster.')
    rows=[[int(s in affected) for s in vocabulary]+[len(affected)] for affected in sets]
    features=pd.DataFrame(rows,columns=[SERVICE_PREFIX+s for s in vocabulary]+[COUNT_FEATURE])
    if not np.array_equal(features[COUNT_FEATURE],df.affected_services.to_numpy()):
        raise ValueError('Abnormal service set cardinality disagrees with run metric.')
    return features,vocabulary


def scale_service_features(raw):
    """Two fixed blocks: membership/sqrt(V), count/V. No label-fitted weights.

    Squared distance = symmetric-difference size/V + (count difference/V)^2.
    Both terms have maximum 1. Count is a redundant, explicit size contribution.
    """
    v=raw.shape[1]-1
    if v<1:raise ValueError('At least one service-membership column is required.')
    weights=np.array([1/np.sqrt(v)]*v+[1/v])
    return raw*weights,weights


def ward(x,k):
    return AgglomerativeClustering(n_clusters=k,linkage='ward',metric='euclidean').fit_predict(x)


def bootstrap_stability(raw,reference,k,repeats=BOOTSTRAPS,seed=SEED):
    """Resample runs, fixed block scaling+Ward; ARI on unique original IDs.

    Copies of a run are not independent evaluation points. If identical copies
    have different labels at a Ward tie, use their modal label (smallest on tie).
    """
    rng=np.random.default_rng(seed);rows=[];n=len(raw)
    for repeat in range(1,repeats+1):
        draw=rng.integers(0,n,n);unique=np.unique(draw)
        if len(unique)<=k:
            rows.append({'repeat':repeat,'ari':np.nan,'n_unique':len(unique),'status':'too_few_unique_runs',
                         'sample_indices_json':json.dumps(draw.tolist())});continue
        z,_=scale_service_features(raw[draw]);labels=ward(z,k)
        mapped=[int(np.bincount(labels[draw==i]).argmax()) for i in unique]
        rows.append({'repeat':repeat,'ari':adjusted_rand_score(reference[unique],mapped),'n_unique':len(unique),
                     'status':'evaluated','sample_indices_json':json.dumps(draw.tolist())})
    return pd.DataFrame(rows)


def choose_k(candidates,stability_gate=STABILITY_GATE):
    """No layers or other metadata accepted. Prefer silhouette among stable Ks."""
    available=candidates.dropna(subset=['silhouette','stability_mean_ari'])
    if available.empty:raise ValueError('No valid unsupervised candidate K.')
    stable=available[available.stability_mean_ari>=stability_gate]
    if len(stable):
        selected=stable.sort_values(['silhouette','stability_mean_ari','k'],ascending=[False,False,True]).iloc[0]
        rule='highest silhouette among K with bootstrap mean ARI >= 0.75; ties: stability then smaller K'
    else:
        selected=available.sort_values(['stability_mean_ari','silhouette','k'],ascending=[False,False,True]).iloc[0]
        rule='no K met stability gate; highest stability, then silhouette, then smaller K; exploratory only'
    return int(selected.k),rule


def fit_services(feature_frame,repeats=BOOTSTRAPS,seed=SEED,k_values=K_VALUES):
    columns=list(feature_frame.columns)
    if (len(columns)<2 or columns[-1]!=COUNT_FEATURE or
            any(not name.startswith(SERVICE_PREFIX) for name in columns[:-1]) or
            columns[:-1]!=sorted(columns[:-1])):
        raise ValueError('Clustering accepts sorted service-membership columns and the count column only.')
    raw=feature_frame.to_numpy(float)
    if not np.isfinite(raw).all():raise ValueError('Missing/nonfinite clustering input; do not impute from labels.')
    if not np.isin(raw[:,:-1],[0,1]).all() or not np.array_equal(raw[:,-1],raw[:,:-1].sum(axis=1)):
        raise ValueError('Membership must be binary and count must equal its row sum.')
    z,weights=scale_service_features(raw)
    candidates=[];all_boot=[];labels_by_k={}
    for k in k_values:
        if not 2<=k<len(raw):continue
        labels=ward(z,k);labels_by_k[k]=labels
        boots=bootstrap_stability(raw,labels,k,repeats,seed);boots.insert(0,'k',k);all_boot.append(boots)
        ari=boots.ari.dropna()
        candidates.append({'k':k,'silhouette':silhouette_score(z,labels),'stability_mean_ari':ari.mean(),
                           'stability_std_ari':ari.std(ddof=1),'stability_q05_ari':ari.quantile(.05),
                           'stability_q95_ari':ari.quantile(.95),'n_bootstraps':len(ari),
                           'minimum_cluster_size':int(np.bincount(labels).min()),
                           'cluster_sizes_json':json.dumps(sorted(np.bincount(labels).tolist(),reverse=True))})
    candidates=pd.DataFrame(candidates);selected,rule=choose_k(candidates)
    labels=labels_by_k[selected]
    # Canonical IDs ordered by count then membership profile, never by layers.
    order=sorted(np.unique(labels),key=lambda c:(np.median(raw[labels==c,-1]),
                 tuple(raw[labels==c,:-1].mean(axis=0)),int(np.flatnonzero(labels==c)[0])))
    mapping={int(c):i+1 for i,c in enumerate(order)}
    canonical=np.array([mapping[int(c)] for c in labels])
    pca=PCA(n_components=2,svd_solver='full');coordinates=pca.fit_transform(z)
    transformed=pd.DataFrame(z,columns=['scaled_'+k for k in columns])
    transformed['cluster']=canonical;transformed['pc1']=coordinates[:,0];transformed['pc2']=coordinates[:,1]
    candidates['selected']=candidates.k==selected
    details={'selected_k':selected,'selection_rule':rule,'stability_gate':STABILITY_GATE,
             'features':columns,'n_samples':len(raw),'seed':seed,'bootstrap_repeats':repeats,
             'service_vocabulary':[name.removeprefix(SERVICE_PREFIX) for name in columns[:-1]],
             'fixed_scale_weights':weights.tolist(),
             'distance_squared':'|A symmetric_difference B|/V + ((|A|-|B|)/V)^2',
             'scaling':'service membership/sqrt(V), service count/V; two equal maximum-distance blocks',
             'pca_explained_variance_ratio':pca.explained_variance_ratio_.tolist(),
             'pca_components':pca.components_.tolist(),'pca_mean':pca.mean_.tolist(),
             'cluster_id_rule':'ascending median abnormal service count, then mean service-membership vector; metadata never used',
             'layer_used_in_fitting_or_k_selection':False}
    return transformed,candidates,pd.concat(all_boot,ignore_index=True),details


def entropy(values):
    counts=pd.Series(values).value_counts().to_numpy(float)
    if not len(counts):return np.nan
    probabilities=counts/counts.sum();return float(-(probabilities*np.log2(probabilities)).sum())


def composition(assignments):
    rows=[];entropies=[];n_observed_layers=assignments.layer.nunique()
    for cluster,s in assignments.groupby('cluster',sort=True):
        h=entropy(s.layer)
        entropies.append({'cluster':cluster,'n_runs':len(s),'n_layers_observed':s.layer.nunique(),
                          'entropy_bits':h,'normalized_entropy':h/np.log2(n_observed_layers) if n_observed_layers>1 else 0.,
                          'normalization_n_layers_observed':n_observed_layers})
        for layer in LAYERS:
            count=int((s.layer==layer).sum());n_layer=int((assignments.layer==layer).sum())
            rows.append({'cluster':cluster,'layer':layer,'n_runs':count,'n_cluster':len(s),'n_layer':n_layer,
                         'proportion_within_cluster':count/len(s),
                         'proportion_within_layer':count/n_layer if n_layer else np.nan})
    return pd.DataFrame(rows),pd.DataFrame(entropies)


def metadata_diagnostics(assignments):
    factors=['layer','topology_mode','persistence','load_users','fault_type','workload_id','edge_deployment_id','task_class_profile','run_id']
    associations=[];composition_rows=[]
    for factor in factors:
        values=assignments[factor].fillna('unrecorded').astype(str)
        observed=values!='unrecorded'
        x=assignments.loc[observed];categories=values[observed]
        varying=len(x)>1 and categories.nunique()>1 and x.cluster.nunique()>1
        associations.append({'factor':factor,'n_runs':len(x),'n_unrecorded_excluded':int((~observed).sum()),
                             'n_categories':categories.nunique(),'ami':adjusted_mutual_info_score(categories,x.cluster) if varying else np.nan,
                             'status':'descriptive_only' if varying else 'constant_or_insufficient_metadata'})
        for cluster,s in assignments.groupby('cluster'):
            for category,count in s[factor].fillna('unrecorded').astype(str).value_counts().items():
                composition_rows.append({'cluster':cluster,'factor':factor,'category':category,'n_runs':int(count),
                                         'n_cluster':len(s),'proportion_within_cluster':count/len(s)})
    return pd.DataFrame(associations),pd.DataFrame(composition_rows)


def conditional_diagnostics(assignments,seed=SEED):
    rows=[];spreads=[];rng=np.random.default_rng(seed)
    for granularity,keys in [('condition',CONDITION),('condition_workload_deployment',CONDITION+['workload_id','edge_deployment_id'])]:
        for values,s in assignments.groupby(keys,dropna=False,sort=True):
            meta={'granularity':granularity,**dict(zip(keys,values))}
            varying=s.layer.nunique()>1 and s.cluster.nunique()>1
            n_mixed=sum(g.layer.nunique()>1 for _,g in s.groupby('cluster'))
            unknown=(s.workload_id=='unrecorded').any() or (s.edge_deployment_id=='unrecorded').any() or (s.topology_mode=='unrecorded').any()
            row={**meta,'n_runs':len(s),'n_layers':s.layer.nunique(),'n_clusters':s.cluster.nunique(),
                 'n_mixed_layer_clusters':n_mixed,'metadata_complete':not unknown,
                 'ami':adjusted_mutual_info_score(s.layer,s.cluster) if varying else np.nan,
                 'permutation_p':np.nan,'n_permutations':0}
            if unknown:row['inference_status']='descriptive_only_unrecorded_condition_metadata'
            elif len(s)<MIN_ASSOCIATION_N:row['inference_status']='descriptive_only_n_lt_20'
            elif not varying:row['inference_status']='descriptive_only_constant_layer_or_cluster'
            elif min(s.layer.value_counts().min(),s.cluster.value_counts().min())<5:row['inference_status']='descriptive_only_sparse_cells'
            else:
                perm=[adjusted_mutual_info_score(rng.permutation(s.layer),s.cluster) for _ in range(PERMUTATIONS)]
                row['permutation_p']=(1+sum(v>=row['ami'] for v in perm))/(PERMUTATIONS+1)
                row['n_permutations']=PERMUTATIONS;row['inference_status']='exploratory_permutation_no_multiple_test_adjustment'
            rows.append(row)
            if granularity=='condition':
                for layer,g in s.groupby('layer'):
                    spreads.append({**meta,'layer':layer,'n_runs':len(g),'n_clusters':g.cluster.nunique(),
                                    'cluster_counts_json':json.dumps({str(int(k)):int(v) for k,v in g.cluster.value_counts().items()},sort_keys=True)})
    return pd.DataFrame(rows),pd.DataFrame(spreads)
