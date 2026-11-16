"""Generate a bilingual manuscript-style README from the current saved results."""
from pathlib import Path
import json
import numpy as np
from descriptive import markdown_table,METRICS
from clustering import SERVICE_PREFIX,COUNT_FEATURE
from loader import LAYERS

HERE=Path(__file__).resolve().parent


def format_value(value,metric,precision=2):
    if not np.isfinite(value):return 'N/A'
    return f'{value:.1%}' if metric.endswith(('rate','ratio')) else f'{value:.{precision}f}'


def write_readme(df,summary,assignments,candidates,comp,entropies,associations,conditional,spreads,metrics,audit):
    faults=df[df.detection_label==1]
    services=sorted(c.removeprefix(SERVICE_PREFIX) for c in assignments if c.startswith(SERVICE_PREFIX))
    n=len(faults);nc=len(assignments);v=len(services);selected=candidates[candidates.selected].iloc[0]
    reason=audit.reason.fillna('')
    excluded=int(reason.str.contains('explicit_no_fault_run|proxy_total_fires_zero',regex=True).sum())
    source_table=markdown_table(['来源 / Source','策略 / Strategy','描述n / Descriptive n','聚类n / Clustering n'],
        [[cohort,mode,len(s),int(assignments.sample_id.isin(s.sample_id).sum())]
         for (cohort,mode),s in faults.groupby(['cohort','topology_mode'],sort=False)])
    layer_table=markdown_table(['层次 / Layer','注入来源含义 / Injection category','实际故障类型 / Observed fault types','描述n','聚类n'],
        [[layer,meaning,', '.join(sorted(faults.loc[faults.layer==layer,'fault_type'].unique())),
          int((faults.layer==layer).sum()),int((assignments.layer==layer).sum())]
         for layer,meaning in [('F0','云边网络 / Cloud–edge network'),('F1','服务运行环境 / Service runtime'),
                               ('F2','外部模型依赖 / External model dependency'),('F3','接口交互 / Interface interaction'),
                               ('F4','执行控制 / Execution control')]])
    example=summary[(summary.topology_mode=='cloud-preference')&(summary.persistence=='persistent')&(summary.load_users==64)]
    examples=[];skew_rows=[]
    for layer in LAYERS:
        s=example[example.layer==layer].set_index('metric');cells=[]
        for metric in METRICS:
            r=s.loc[metric]
            cells.append('N/A' if not r.n_valid else format_value(r['mean'],metric)+' ± '+format_value(r['std'],metric))
        examples.append([layer,int(s.n_runs.iloc[0]),*cells])
        skew_rows.append([layer,*[('N/A' if not s.loc[m,'n_valid'] else
                         format_value(s.loc[m,'median'],m)+' / '+format_value(s.loc[m,'iqr'],m))
                         for m in ['propagation_breadth','mean_span_seconds','p95_span_seconds']]])
    example_table=markdown_table(['Layer','n','S','T','R','B*','NER','LER','Mean(s)','P95(s)','RR','TFR'],examples)
    skew_table=markdown_table(['Layer','B: median / IQR','Mean span(s): median / IQR','P95 span(s): median / IQR'],skew_rows)
    candidate_table=markdown_table(['K','Silhouette','Bootstrap ARI: mean ± SD','ARI: Q05–Q95','簇大小 / Cluster sizes','选用 / Selected'],
        [[r.k,f'{r.silhouette:.4f}',f'{r.stability_mean_ari:.4f} ± {r.stability_std_ari:.4f}',
          f'{r.stability_q05_ari:.3f}–{r.stability_q95_ari:.3f}',r.cluster_sizes_json,'Yes' if r.selected else 'No'] for r in candidates.itertuples()])
    composition_rows=[]
    for cluster,s in comp.groupby('cluster'):
        values=s.set_index('layer');h=entropies[entropies.cluster==cluster].iloc[0]
        composition_rows.append([cluster,int(h.n_runs),*[f'{int(values.loc[layer,"n_runs"])} ({values.loc[layer,"proportion_within_cluster"]:.1%})' for layer in LAYERS],
                                  f'{h.entropy_bits:.3f}',f'{h.normalized_entropy:.3f}'])
    composition_table=markdown_table(['Cluster','n',*LAYERS,'H (bit)','H / log₂L'],composition_rows)
    profiles=[]
    for cluster,s in assignments.groupby('cluster'):
        counts=s[COUNT_FEATURE]
        prevalent=sorted([(service,float(s[SERVICE_PREFIX+service].mean())) for service in services],key=lambda x:(-x[1],x[0]))
        top='; '.join(f'{name.removeprefix("agent-network-")} {fraction:.1%}' for name,fraction in prevalent[:5])
        profiles.append([cluster,len(s),f'{counts.mean():.2f} ± {counts.std(ddof=1):.2f}',
                         f'{counts.median():.1f} [{counts.quantile(.25):.1f}, {counts.quantile(.75):.1f}]',
                         f'{int(counts.min())}–{int(counts.max())}',top])
    profile_table=markdown_table(['Cluster','n','S: mean ± SD','S: median [Q25,Q75]','S: min–max','常见异常服务 / Frequent abnormal services'],profiles)
    dispersion_table=markdown_table(['Layer','n','出现簇数 / Clusters represented','P(cluster | layer)'],
        [[layer,int((assignments.layer==layer).sum()),int(assignments.loc[assignments.layer==layer,'cluster'].nunique()),
          '; '.join(f'C{int(r.cluster)} {r.proportion_within_layer:.1%}' for r in comp[comp.layer==layer].itertuples())
          if (assignments.layer==layer).any() else 'N/A'] for layer in LAYERS])
    covariate_table=markdown_table(['后验因素 / Post-hoc factor','有效n / Available n','类别数 / Categories','AMI','未知排除n / Unknown excluded n'],
        [[r.factor,r.n_runs,r.n_categories,f'{r.ami:.4f}' if np.isfinite(r.ami) else 'N/A',r.n_unrecorded_excluded] for r in associations.itertuples()])
    base=conditional[conditional.granularity=='condition']
    condition_table=markdown_table(['策略 / Strategy','模式 / Mode','Load','n','簇数 / Clusters','跨层混合簇 / Mixed-layer clusters','AMI','推断 / Inference'],
        [[r.topology_mode,r.persistence,r.load_users,r.n_runs,r.n_clusters,r.n_mixed_layer_clusters,
          f'{r.ami:.3f}' if np.isfinite(r.ami) else 'N/A','Descriptive only' if r.n_permutations==0 else 'Exploratory permutation'] for r in base.itertuples()])
    excluded_rows=[]
    for r in faults[faults.n_submitted!=faults.n_graph_tasks].itertuples():
        excluded_rows.append([r.sample_id.removeprefix('benchmark/data/MDOC/agent_faults/'),r.n_submitted,r.n_graph_tasks,r.layer])
    coverage_table=markdown_table(['样本 / Sample (relative to source root)','提交n / Submitted','图任务n / Graph tasks','Layer'],excluded_rows)
    config=json.loads((HERE/'experiment_config.json').read_text())
    model=json.loads((HERE/'tables/clustering_model.json').read_text())
    passed=bool((candidates.stability_mean_ari>=.75).any())
    pca=model['pca_explained_variance_ratio']
    example_f1=example[(example.layer=='F1')&(example.metric=='node_error_rate')].iloc[0]
    # Shared count values across clusters show that identity also affects grouping.
    count_sets=[set(s[COUNT_FEATURE].astype(int)) for _,s in assignments.groupby('cluster')]
    overlap=sorted(set.intersection(*count_sets)) if len(count_sets)>1 else []
    overlap_text=', '.join(map(str,overlap)) if overlap else 'none'
    overlap_zh='、'.join(map(str,overlap)) if overlap else '无'
    topology_ami=float(associations.loc[associations.factor=='topology_mode','ami'].iloc[0])
    topology_n=int(associations.loc[associations.factor=='topology_mode','n_runs'].iloc[0])
    sizes=assignments.groupby('cluster',sort=True).size().tolist()
    size_text='/'.join(map(str,sizes))
    size_en=(', '.join(map(str,sizes[:-1]))+' and '+str(sizes[-1])) if len(sizes)>1 else str(sizes[0])
    all_layers_in_every_cluster=all(set(g.layer)==set(LAYERS) for _,g in assignments.groupby('cluster'))
    layer_mix_zh=('各簇均包含F0–F4' if all_layers_in_every_cluster else '各簇的层次覆盖见组成表')
    layer_mix_en=('All clusters contain F0–F4' if all_layers_in_every_cluster else 'Cluster layer coverage is reported in the composition table')
    every_layer_spreads=all(assignments.loc[assignments.layer==layer,'cluster'].nunique()>1 for layer in LAYERS)
    spread_zh='每个层次都分布于多个簇' if every_layer_spreads else '不同层次的簇间分布见表9'
    spread_en='every layer appears in multiple clusters' if every_layer_spreads else 'Table 9 reports each layer’s distribution across clusters'
    profile_zh=[];profile_en=[]
    for cluster,g in assignments.groupby('cluster'):
        count=g[COUNT_FEATURE]
        profile_zh.append(f'Cluster {cluster}包含{len(g)}次运行，异常服务数均值为{count.mean():.2f}±{count.std(ddof=1):.2f}')
        profile_en.append(f'Cluster {cluster} contains {len(g)} runs with a mean abnormal-service count of {count.mean():.2f}±{count.std(ddof=1):.2f}')
    profile_zh='；'.join(profile_zh);profile_en='; '.join(profile_en)
    f0_count=int((assignments.layer=='F0').sum())
    f4_s=example[(example.layer=='F4')&(example.metric=='affected_services')].iloc[0]
    f2_s=example[(example.layer=='F2')&(example.metric=='affected_services')].iloc[0]
    f2_damage=example[(example.layer=='F2')&(example.metric=='task_failure_rate')]['mean'].iloc[0]
    f4_damage=example[(example.layer=='F4')&(example.metric=='task_failure_rate')]['mean'].iloc[0]
    f1_s_sd=example[(example.layer=='F1')&(example.metric=='affected_services')]['std'].iloc[0]
    parts=[]
    def heading(text):parts.append(text)
    def pair(zh,en):parts.extend(['**中文**\n\n'+zh,'**English**\n\n'+en])
    def block(text):parts.append(text)

    heading('# 动机二：云边智能体微服务的故障层次异构性与异常服务模式可辨识性\n# Motivation II: Fault-Layer Heterogeneity and Layer Identifiability from Abnormal-Service Patterns in Cloud–Edge Agent Microservices')
    pair('本文为基于当前实测结果的论文式实验稿。正文采用中英文逐段对照，表格与公式共用，术语与数值在两种语言中保持一致。研究对象为完整运行执行图中的观测异常；所有结论均受所列数据与观测边界约束。',
         'This manuscript-style empirical study reports the current measured results. Chinese and English paragraphs are aligned, with shared tables and equations to preserve terminology and numerical consistency. The study concerns observed anomalies in complete-run execution graphs, and its conclusions are restricted to the stated data and observation boundaries.')
    heading('## 摘要 / Abstract')
    pair(f'云边智能体微服务中的故障可能作用于不同系统层次，而运行结束后观测到的异常服务未必唯一对应其注入来源。为考察这一问题，本文设计两个互补实验：首先按协作策略、故障模式和负载分层，描述F0–F4故障的影响范围、局部异常与任务损害；随后仅使用异常逻辑服务的身份编码与数量进行无监督聚类，并在聚类完成后分析故障层次组成。数据包含{n}次故障运行，其中{nc}次满足图覆盖与服务集合一致性要求。{v}个服务的二值编码与服务总数组成{v+1}维输入。Ward聚类在K=2–5中选择K={metrics["n_clusters"]}，形成{size_text}的分组，轮廓系数为{metrics["silhouette"]:.4f}，100次bootstrap的ARI为{metrics["bootstrap_stability_mean_ari"]:.4f}±{metrics["bootstrap_stability_std_ari"]:.4f}。{layer_mix_zh}，与注入层次的AMI仅为{metrics["layer_ami"]:.4f}。结果表明，当前异常服务模式与故障层次的整体对应关系较弱，但聚类波动、条件混杂和未验证的因果归属限制了结论强度。本文据此提出研究需求：根因分析应进一步结合故障层次、云边部署依赖与Agent执行关系，而不能仅依据异常服务集合或其规模进行解释。',
         f'Faults in cloud–edge agent microservices may act at different system layers, while the abnormal services observed after a run may not uniquely indicate the injected source. We investigate this issue through two complementary experiments: condition-stratified descriptive analysis of impact scope, local anomalies, and task damage across F0–F4, followed by unsupervised clustering based solely on abnormal logical-service identities and their count, with fault-layer composition examined afterward. The dataset contains {n} fault runs, of which {nc} satisfy graph-coverage and service-set consistency checks. Binary membership for {v} services and the total count yield {v+1} input dimensions. Ward clustering selects K={metrics["n_clusters"]} from K=2–5, producing cluster sizes of {size_en} runs, a silhouette score of {metrics["silhouette"]:.4f}, and bootstrap ARI of {metrics["bootstrap_stability_mean_ari"]:.4f}±{metrics["bootstrap_stability_std_ari"]:.4f} over 100 resamples. {layer_mix_en}, and their adjusted mutual information with injected layers is only {metrics["layer_ami"]:.4f}. These results indicate weak overall alignment between the current abnormal-service patterns and fault layers. However, resampling variability, confounding conditions, and unverified causal attribution limit the strength of this finding. The resulting research need is to incorporate fault layers, cloud–edge deployment dependencies, and agent execution relationships into root-cause analysis rather than interpret abnormal-service identity or scale in isolation.')
    pair('关键词：云边协同；智能体微服务；故障层次；异常服务集合；无监督聚类；根因可辨识性。',
         'Keywords: cloud–edge collaboration; agent microservices; fault layers; abnormal-service sets; unsupervised clustering; root-cause identifiability.')
    heading('## 1. 研究动机与问题 / Research Motivation and Questions')
    pair('智能体服务的异常既可能反映自身运行问题，也可能是基础网络、外部模型依赖、接口交互或执行控制问题的下游表现。因此，观测到某个服务异常，并不自动证明该服务就是根因。本文研究的关键不是建立统一严重程度排序，而是检验不同故障层次的观测表现是否存在异构性，以及异常服务身份与规模是否能够与注入层次形成清晰对应。层次标识来自故障注入元数据，而不是由聚类推断或独立根因定位验证得到。',
         'An anomalous agent service may reflect a problem in its own runtime or a downstream manifestation of network infrastructure, external model dependencies, interface interaction, or execution control. Observing an abnormal service therefore does not automatically establish that service as the root cause. Rather than impose a universal severity ordering, this study examines whether observed manifestations vary across fault layers and whether abnormal-service identities and scale align clearly with injected layers. Layer identifiers come from injection metadata, not from clustering or independently validated root-cause localization.')
    pair('RQ1：在相同的已记录协作策略、故障模式与负载条件内，同层和跨层故障在影响范围、局部异常及任务损害上表现如何？RQ2：不使用故障层次标签，仅依据执行图中的异常服务集合与数量进行聚类时，各簇会呈现怎样的层次组成与条件构成？两项实验分别提供描述性和探索性证据，不训练故障层次分类器，也不将影响范围转换为严重程度标签。',
         'RQ1: Within the same recorded collaboration strategy, fault mode, and load, how do within-layer and cross-layer faults differ in impact scope, local anomalies, and task damage? RQ2: When clustering uses only execution-graph abnormal-service sets and their counts without fault-layer labels, what fault-layer and experimental-condition compositions emerge? The experiments provide descriptive and exploratory evidence, respectively. They neither train a fault-layer classifier nor convert impact scope into severity labels.')
    heading('## 2. 数据与预处理 / Data and Preprocessing')
    heading('### 2.1 数据来源与样本单位 / Sources and Sampling Unit')
    pair(f'数据读取自`benchmark/data/MDOC/agent_faults/topology_runs/`的全部可解析带标签运行，以及`benchmark/data/MDOC/agent_faults/abnormal/load-64/common/`的历史运行。按既定规则排除采集块`20261104-012022-3391d406`、`20261104-034933-2183a1ad`，并排除`proxy_stats.json.total_fires`明确为数值0的样本，两条规则的并集为{excluded}次。计数文件或字段缺失时保留“未知”，不自动视为0。未完成采集、缺图或无法解析标签的目录只保留在源审计中，不补零或补造样本。最终{n}次故障运行进入描述统计，包含{int(faults.n_submitted.sum())}条提交记录与{int(faults.n_graph_tasks.sum())}个图任务，负载均为64个并发用户。64不是提交任务总数。',
         f'We read all resolvable labeled runs under `benchmark/data/MDOC/agent_faults/topology_runs/` and historical runs under `benchmark/data/MDOC/agent_faults/abnormal/load-64/common/`. The established exclusions remove collection blocks `20261104-012022-3391d406` and `20261104-034933-2183a1ad`, together with samples whose `proxy_stats.json.total_fires` is explicitly numeric zero; their union excludes {excluded} samples. A missing counter file or field remains unknown and is not treated as zero. Incomplete collections, missing graphs, and unresolved labels are retained only in the source audit, without zero filling or fabricated samples. The descriptive dataset contains {n} fault runs, {int(faults.n_submitted.sum())} submission records, and {int(faults.n_graph_tasks.sum())} graph tasks, all at a load of 64 concurrent users. This load value is not the total number of submitted tasks.')
    pair('每个样本是一整次故障运行；同一运行中的多个任务不作为独立聚类样本。历史common未记录协作策略，始终标记为`unrecorded`，不合并到dynamic。数据来源与纳入数量见表1。',
         'The sampling unit is one complete fault run; tasks within a run are not treated as independent clustering samples. The historical common cohort has no recorded collaboration strategy and retains the label `unrecorded`; it is not merged into dynamic. Table 1 summarizes the cohorts and inclusion counts.')
    block('**表1 / Table 1. 数据来源与样本量 / Cohorts and sample counts.**\n\n'+source_table)
    heading('### 2.2 注入层次与观测来源 / Injected Layers and Observation Sources')
    block('**表2 / Table 2. 实际覆盖的注入层次与类型 / Observed injected layers and fault types.**\n\n'+layer_table)
    pair('F0–F4仅用于分组与聚类后的组成分析。表2中的类型是实际出现的注入类型，不表示每个层次的全部可能故障都已覆盖，也不表示所有图内异常已经逐项归因到注入根因。',
         'F0–F4 are used only for grouping and post-clustering composition analysis. The types in Table 2 are those actually present in the data, not an exhaustive taxonomy for each layer. They also do not imply that every graph anomaly has been causally linked to the injected source.')
    block(markdown_table(['文件 / File','提取内容与用途 / Extracted information and use'],[
        ['`*_label.txt`','层次、类型、策略、模式、负载、任务列表、边缘服务 / Layer, type, strategy, mode, load, task indices, edge services'],
        ['`proxy_stats.json`','样本级无故障过滤与缺失状态 / Run-level filtering and missing-counter status'],
        ['`task.log`','实际提交数、MDOC/SIMPLE与COMPLEX构成 / Submission counts and SIMPLE/COMPLEX composition'],
        ['`agent-network/graph/call_chains.json`','span/attempt状态、error、executor、耗时、重试、记录边和终态 / Span/attempt status, errors, executors, durations, retries, recorded edges, terminal status'],
        ['`agent-network/log/*planner*.log`','保留早期辅助字段的结束时间出处；不筛选本轮聚类 / Completion provenance for retained early auxiliary fields; not a clustering filter'],
        ['`chaos/log/*_fires.jsonl`','effect_applied=true且t在故障窗的命中审计；不作为聚类特征 / Applied, in-window hit auditing; not a clustering feature']]))
    heading('### 2.3 过程异常、逻辑服务与覆盖检查 / Process Anomalies, Logical Services, and Coverage Checks')
    pair('若span或任一attempt的`status==3`或`error`非空，则记为过程异常；即使重试后恢复成功也保留该过程异常。服务名优先采用实际`executor`，缺失时退回计划服务名，合并`-edge`与服务内部工具到同一逻辑服务，虚拟Summarizer不计为服务。设任务t的异常服务集合为E_t，运行r的异常服务集合与数量定义如下。',
         'A process anomaly is recorded when a span or any attempt has `status==3` or a nonempty `error`; an anomalous attempt remains counted even if a later retry succeeds. Service identity preferentially uses the actual `executor`, falling back to the planned service when unavailable. The `-edge` suffix and service-internal tools are merged into the same logical service, while the virtual Summarizer is excluded. Let E_t be the abnormal-service set of task t. For run r, we define:')
    block(r'$$A_r=\bigcup_{t\in r}E_t,\qquad S_r=|A_r|.$$')
    pair(f'同一服务无论在哪些任务、多少次尝试中异常，均只计一次。`affected_services_json`保存集合，`affected_services`保存数量，二者必须一致。聚类额外要求图任务数等于实际提交数，因此{n-nc}次图覆盖不足运行仅进入描述统计，{nc}次进入聚类；不再要求故障开始后30秒内存在已完成任务。',
         f'A service is counted once regardless of how many tasks or attempts exhibit its anomaly. The set is stored in `affected_services_json`, and its cardinality in `affected_services`; the two must agree. Clustering additionally requires graph-task count to match actual submission count. Consequently, {n-nc} runs with incomplete graph coverage remain descriptive-only, and {nc} runs enter clustering. Completion within the first 30 seconds is no longer an inclusion requirement.')
    pair('该完整性检查是必要条件而非逐任务完备性证明，因为客户端提交UUID与服务端task_id没有逐项映射。更重要的是，本研究使用整次采集图中的观测异常：图缺少绝对span时间，可能包含背景错误或窗外异常。逻辑服务也不等同于Pod、物理节点、网络段或LLM供应商。因此，后续集合和传播量均不能直接解释为已验证的故障因果传播。缺图运行的范围指标仅覆盖已观测部分，T/M也只是观测异常任务比例，不能称为完整真实影响比例。',
         'This coverage check is necessary but does not prove task-level completeness because client submission UUIDs are not mapped individually to server task IDs. More importantly, the analysis uses anomalies observed anywhere in the collected run: absolute span timestamps are unavailable, and background or off-window errors may be included. A logical service is also distinct from a Pod, physical node, network segment, or LLM provider. The resulting sets and propagation quantities must therefore not be interpreted as verified causal fault propagation. For incomplete graphs, scope measures cover only observed entities and T/M is an observed anomalous-task ratio rather than a complete true impact ratio.')
    heading('## 3. 实验一：分条件的异常表现统计 / Experiment I: Condition-Stratified Manifestation Statistics')
    heading('### 3.1 指标与汇总方法 / Measures and Aggregation')
    block('**表3 / Table 3. 十项描述指标 / Ten descriptive measures.**\n\n'+markdown_table(['类别 / Category','指标 / Measure','定义 / Definition'],[
        ['范围 / Scope','Affected Services (S)','唯一异常逻辑服务数 / Number of unique abnormal logical services'],
        ['范围 / Scope','Affected Tasks (T)','至少一次过程异常的唯一任务数 / Unique tasks with at least one process anomaly'],
        ['范围 / Scope','Affected Task Ratio (R)','T/M，M为实际提交记录数 / T/M, where M is the actual submission count'],
        ['范围 / Scope','Propagation Breadth (B*)','记录图错误起点可达的异常后继实体数，排除全部起点 / Abnormal successor entities reachable from observed error roots, excluding all roots'],
        ['局部 / Local','Node Error Rate (NER)','异常执行顶点出现数/可观测顶点出现数 / Anomalous execution-vertex occurrences divided by observable occurrences'],
        ['局部 / Local','Link Error Rate (LER)','目标顶点异常的已解析边数/已解析边数 / Resolved edges with an anomalous target divided by all resolved edges'],
        ['局部 / Local','Mean Span Duration','可观测非负span耗时均值，秒 / Mean observable nonnegative span duration, seconds'],
        ['局部 / Local','P95 Span Duration','同一耗时集合95%分位数，秒 / 95th percentile of the same duration set, seconds'],
        ['局部 / Local','Retry Rate (RR)','attempts>1或attempt_count>1的顶点比例 / Fraction of vertices with multiple attempts'],
        ['损害 / Damage','Task Failure Rate (TFR)','终态0/(终态0+7) / Terminal-status-0 tasks divided by terminal-status-0-or-7 tasks']]))
    pair('传播实体按`(task_id, level, vertex)`去重；观测错误起点是在记录有向图中没有其他错误祖先的错误顶点。只沿真实记录边计算可达关系，不为fallback执行器补造边。B*是观测代理，而非已验证注入根因的传播广度。Node/Link分别指执行顶点和记录依赖边，不是物理节点或网络包错误率。',
         'Propagation entities are deduplicated by `(task_id, level, vertex)`. An observed error root is an erroneous vertex with no other erroneous ancestor in the recorded directed graph. Reachability uses only recorded edges; no edges are invented for fallback executors. B* is therefore an observational proxy rather than verified propagation from the injected root cause. Node and link refer to execution vertices and recorded dependencies, not physical-node or packet error rates.')
    block(r'$$B_r=\sum_{t\in r}\left|\left(\mathcal A_t\cap\operatorname{Reach}(\mathcal R_t)\right)\setminus\mathcal R_t\right|.$$')
    pair('其中𝒜_t为任务的异常执行顶点集合，ℛ_t为上述观测错误起点集合。各指标先在每次运行内计算，再按“协作策略×故障模式×负载×故障层次”分组等权汇总。下式n为该条件、层次与指标的有效运行数。对每项指标同时保存运行数、有效数、均值、样本SD、中位数、Q25/Q75、IQR及min/max；n=1时SD记N/A，缺测不填0。任务列表、边缘部署和实际提交类别构成的细分统计另行保存。',
         'Here, 𝒜_t denotes the anomalous execution vertices and ℛ_t the observed error roots. Each measure is first computed within a run, then aggregated with equal run weights within collaboration-strategy × fault-mode × load × fault-layer strata. In the equation below, n is the number of runs with an observed value for that measure in the condition/layer. We retain run count, valid count, mean, sample SD, median, Q25/Q75, IQR, and minimum/maximum for every measure. SD is unavailable for n=1, and missing observations are not replaced by zero. Additional tables stratify by task list, edge deployment, and observed submission-category composition.')
    block(r'$$\bar y_{c,l}=\frac{1}{n}\sum_{r\in(c,l)}y_r,\qquad s_{c,l}=\sqrt{\frac{\sum_{r\in(c,l)}(y_r-\bar y_{c,l})^2}{n-1}},\qquad \operatorname{IQR}=Q_{0.75}-Q_{0.25}.$$')
    heading('### 3.2 统计结果 / Descriptive Results')
    pair('表4给出云优先×persistent×load-64条件内的全部十项指标。完整条件分块表与更细分统计保存在附录链接中；这里不将不同策略、模式或任务构成混合后解释为层次效应。',
         'Table 4 reports all ten measures for cloud-preference × persistent × load-64. The complete condition-blocked table and finer-stratified results are linked in the appendix. Differences across strategies, modes, or task composition are not pooled and interpreted as fault-layer effects.')
    block('**表4 / Table 4. 同一条件内的均值±SD / Within-condition means ± SD.**\n\n'+example_table)
    pair(f'本条件下F4的异常服务均值为{f4_s["mean"]:.2f}，F2为{f2_s["mean"]:.2f}；F2的最终任务失败率均值为{f2_damage:.1%}，F4为{f4_damage:.1%}。这说明观测服务影响范围与最终任务损害的排序并不必然一致。F1的异常服务数SD为{f1_s_sd:.2f}，节点异常率从'+f'{example_f1["min"]:.1%}到{example_f1["max"]:.1%}'+'，也显示同层不同类型/运行的表现可以存在较大差异。',
         f'Within this condition, F4 affects an average of {f4_s["mean"]:.2f} services, compared with {f2_s["mean"]:.2f} for F2. Mean final task failure is {f2_damage:.1%} for F2 and {f4_damage:.1%} for F4. Thus, observed service scope and final task damage need not have the same ordering. F1 has a service-count SD of {f1_s_sd:.2f} and a node-error range of '+f'{example_f1["min"]:.1%}–{example_f1["max"]:.1%}'+', illustrating substantial variation among fault types/runs within the same layer.')
    block('**表5 / Table 5. 传播广度与耗时的中位数及IQR / Medians and IQRs for breadth and durations.**\n\n'+skew_table)
    pair('这些结果提供的是小样本描述证据。SD主要反映同层不同类型、配置或实际执行的差异，不是同一种故障多次独立重复注入所得的置信区间。表4中的F0为该条件缺测，而非零影响。完整统计不支持跨条件的全局严重程度排序，也不证明某一单独指标在所有条件下都有效或无效。',
         'These are descriptive findings from small samples. SD primarily reflects differences in fault types, configurations, or realized execution within a layer; it is not a confidence interval from independent repeated injections of one identical fault. F0 is unobserved in the condition shown in Table 4, rather than having zero impact. The full statistics do not establish a global severity ordering across conditions or demonstrate that any single measure is universally effective or ineffective.')
    heading('## 4. 实验二：异常服务集合与数量聚类 / Experiment II: Clustering Abnormal-Service Sets and Counts')
    heading('### 4.1 编码与距离 / Encoding and Distance')
    pair(f'本轮聚类样本覆盖{v}个异常逻辑服务。对每次运行构造服务二值编码b_rj与数量S_r，得到{v+1}维原始输入；服务词表见附录。{nc}次运行对应{metrics["n_unique_service_patterns"]}种不同模式。特征相同的运行仍分别保留，因此PCA散点可能重合，视觉点数少于运行数不代表样本被去重。',
         f'The clustering cohort contains {v} observed abnormal logical services. Binary service-membership indicators b_rj and count S_r yield {v+1} raw input dimensions; the vocabulary is listed in the appendix. The {nc} runs represent {metrics["n_unique_service_patterns"]} distinct patterns. Runs with identical features remain separate observations, so PCA points may overlap; fewer visible points do not imply feature-based sample deduplication.')
    block(r'$$b_{rj}=\mathbf 1[s_j\in A_r],\qquad \mathbf x_r=(b_{r1},\ldots,b_{rV},S_r),\qquad S_r=\sum_{j=1}^{V}b_{rj}.$$')
    pair('服务总数由二值编码求和得到，与服务身份不是独立证据。我们预先固定两块缩放：服务身份向量除√V、数量除V。该缩放使集合差异与数量差异两项的最大平方距离均为1，避免量纲造成数量块自动压倒集合块；同时明确保留数量方向的额外权重。没有使用RobustScaler、稀有服务频率加权或根据层次组成调节权重。',
         'The service count is the sum of binary memberships and is not independent evidence. We fix a two-block scaling before fitting: membership indicators are divided by √V and count by V. The maximum squared-distance contribution of each block is then one, preventing the numerical units of count from automatically dominating membership while explicitly retaining extra weight on scale. We do not use RobustScaler, inverse-frequency weighting of rare services, or weights tuned to layer composition.')
    block(r'$$\mathbf z_r=\left(\frac{b_{r1}}{\sqrt V},\ldots,\frac{b_{rV}}{\sqrt V},\frac{S_r}{V}\right),\qquad d^2(A_r,A_q)=\frac{|A_r\triangle A_q|}{V}+\left(\frac{S_r-S_q}{V}\right)^2.$$')
    pair('第一项度量异常服务身份的对称差，第二项度量服务规模差异；相同服务数量但服务身份不同的运行仍有非零距离。F0–F4、策略、模式、故障类型、五个早期局部指标、其他范围指标、最终失败率和命中次数均不进入拟合。当前实验研究运行结束后的异常服务模式，不能继续表述为早期局部症状聚类或在线检测实验。',
         'The first term measures the symmetric difference in abnormal-service identities; the second measures the difference in scale. Runs with equal counts but different service identities therefore remain separated. F0–F4, strategy, mode, fault type, the five early local measures, other scope measures, final failure rate, and hit counts are excluded from fitting. This experiment concerns post-run abnormal-service patterns and should not be described as early-local-symptom clustering or online detection.')
    heading('### 4.2 聚类、K选择与重采样 / Clustering, K Selection, and Resampling')
    pair('在完整的缩放特征空间中使用欧氏距离与Ward凝聚层次聚类[1]，候选K为2、3、4、5。每K执行100次运行级有放回bootstrap（seed=20261111）：从N次原始运行中抽取N次，可重复也可遗漏；固定词表与缩放，重新拟合Ward。比较时仅在本次抽中的唯一原始运行上计算参考分区与重采样分区的ARI[3]，重复副本以众数簇代表，平票取较小编号。不同K使用相同的抽样序列。',
         'We apply Euclidean-distance Ward agglomerative clustering [1] in the complete scaled feature space, considering K=2, 3, 4, and 5. For each K, we perform 100 run-level bootstrap resamples with replacement (seed=20261111): N draws are made from N original runs, allowing duplication and omission. Ward is refitted with the vocabulary and scaling fixed. ARI [3] compares the reference and resampled partitions only on unique original runs appearing in that resample; duplicate copies use their modal cluster, with ties assigned the smaller identifier. All K values share the same resampling sequence.')
    pair('选择规则预先固定：在平均bootstrap ARI≥0.75的候选中选择轮廓系数最高者；若均未达门槛，依稳定性、轮廓系数、小K依次选择探索性解。0.75只是本实验工作门槛，不是普遍合格线或显著性标准。K选择只接收特征矩阵，不接收层次或条件标签。PCA只将同一缩放空间投影到二维用于绘图，不参与实际聚类。',
         'The selection rule is fixed in advance: among candidates with mean bootstrap ARI≥0.75, select the highest silhouette score. If none qualifies, choose an exploratory solution by stability, silhouette, and then smaller K. The value 0.75 is a working threshold for this study, not a universal validity or significance criterion. K selection receives only the feature matrix, without layer or condition labels. PCA projects the same scaled space to two dimensions for display only and is not used for clustering.')
    heading('### 4.3 评价量的定义 / Definitions of Evaluation Measures')
    pair('轮廓系数[2]衡量簇内紧密性与簇间分离度。对样本i，a_i为到同簇其他样本的平均距离，b_i为到最近其他簇的平均距离；取全部样本轮廓值的均值。数值范围为−1到1，接近0提示边界重叠，较大的正值表示相对更好的分离。它不是分类准确率。',
         'The silhouette coefficient [2] measures within-cluster cohesion and between-cluster separation. For sample i, a_i is its mean distance to other members of its cluster and b_i its mean distance to the nearest other cluster; the overall score is the sample average. Values range from −1 to 1, with values near zero indicating overlap and larger positive values indicating relatively better separation. It is not classification accuracy.')
    block(r'$$s_i=\frac{b_i-a_i}{\max(a_i,b_i)},\qquad \operatorname{Silhouette}=\frac1N\sum_i s_i.$$')
    pair('Bootstrap ARI[3]衡量重采样分区与参考分区的一致性，并校正随机一致。1表示两种分组相同，接近0表示随机一致水平，也可为负。本文报告100次ARI的均值、样本SD和第5/95百分位；它评价当前样本组成扰动下的分区敏感性，不是与F0–F4相比的准确率，也不是100次独立物理故障实验。',
         'Bootstrap ARI [3] measures agreement between each resampled partition and the reference partition after adjusting for chance. A value of one indicates identical partitions, values near zero indicate chance-level agreement, and negative values are possible. We report the mean, sample SD, and 5th/95th percentiles across 100 ARIs. This quantifies sensitivity to perturbations of the current sample composition; it is neither accuracy against F0–F4 nor evidence from 100 independent physical fault experiments.')
    block(r'$$\overline{\mathrm{ARI}}=\frac1B\sum_{b=1}^{B}\mathrm{ARI}_b,\qquad s_{\mathrm{ARI}}=\sqrt{\frac{\sum_{b=1}^{B}(\mathrm{ARI}_b-\overline{\mathrm{ARI}})^2}{B-1}},\qquad B=100.$$')
    pair('聚类完成后，以簇标识C和注入层次L计算机会校正互信息AMI[4]。AMI=1表示分区一致，接近0表示整体对应关系接近随机分区的期望，也可为负。簇内层次熵H_c衡量混合程度，并除以log₂L_obs归一化；本轮L_obs=5。AMI和熵不参与K或权重选择。',
         'After clustering, adjusted mutual information (AMI) [4] compares cluster identifiers C with injected layers L. AMI=1 indicates identical partitions, whereas values near zero indicate overall alignment close to the expectation for random partitions; negative values are possible. Within-cluster layer entropy H_c measures mixing and is normalized by log₂L_obs, with L_obs=5 in this cohort. Neither AMI nor entropy is used to choose K or feature weights.')
    block(r'$$\mathrm{AMI}(C,L)=\frac{I(C;L)-\mathbb E[I(C;L)]}{\tfrac12[H(C)+H(L)]-\mathbb E[I(C;L)]},\qquad H_c=-\sum_l p(l\mid c)\log_2p(l\mid c).$$')
    heading('### 4.4 聚类结果 / Clustering Results')
    block('**表6 / Table 6. 标签无关的K选择与重采样结果 / Label-blind K selection and resampling results.**\n\n'+candidate_table)
    pair(f'K={metrics["n_clusters"]}同时取得候选中最高轮廓系数（{metrics["silhouette"]:.4f}）和平均bootstrap ARI（{metrics["bootstrap_stability_mean_ari"]:.4f}），并'+('达到' if passed else '未达到')+'0.75工作门槛。其ARI的SD为'+f'{metrics["bootstrap_stability_std_ari"]:.4f}，第5百分位为{selected.stability_q05_ari:.3f}'+ '，因此不能将均值达标解释为每次重采样都稳定。候选K=3–5并未形成更高质量或更稳定的细分。',
         f'K={metrics["n_clusters"]} attains both the highest candidate silhouette score ({metrics["silhouette"]:.4f}) and mean bootstrap ARI ({metrics["bootstrap_stability_mean_ari"]:.4f}), and '+('meets' if passed else 'does not meet')+f' the 0.75 working threshold. Its ARI SD is {metrics["bootstrap_stability_std_ari"]:.4f}, and its 5th percentile is {selected.stability_q05_ari:.3f}; meeting the threshold on average must not be interpreted as stability in every resample. Candidates K=3–5 do not provide better-separated or more stable refinements.')
    block('![图3 / Figure 3. 异常服务模式聚类 / Abnormal-service pattern clusters.](figures/fig3_affected_service_clusters.png)')
    pair(f'图3展示{nc}次运行在PCA二维空间中的位置，颜色为簇，X为投影簇均值。PC1/PC2分别解释{pca[0]:.1%}/{pca[1]:.1%}的方差。二维图不是聚类输入，重合点也不表示不同注入来源已经具有相同因果机制。[PDF](figures/fig3_affected_service_clusters.pdf)。',
         f'Figure 3 shows the {nc} runs in two-dimensional PCA space. Colors indicate clusters and X markers denote projected cluster means. PC1 and PC2 explain {pca[0]:.1%} and {pca[1]:.1%} of variance, respectively. The two-dimensional plot is not the clustering input, and coincident points do not establish identical causal mechanisms for different injected sources. [PDF](figures/fig3_affected_service_clusters.pdf).')
    block('**表7 / Table 7. 异常服务规模与身份概况 / Abnormal-service scale and identity profiles.**\n\n'+profile_table)
    pair(f'{profile_zh}。分群明显伴随规模差异。服务数量为{overlap_zh}的运行同时出现在不同簇中，说明服务身份也影响划分；但本文未开展移除数量项的消融，因此不主张身份编码相对数量的独立增益，也不将两簇命名为严重程度等级。',
         f'{profile_en}. The grouping is clearly accompanied by differences in scale. Runs with service counts of {overlap_text} occur in different clusters, showing that identity also affects the partition. However, we do not perform a count-removal ablation and therefore do not claim an independent benefit of identity encoding over count or interpret the clusters as severity levels.')
    heading('### 4.5 聚类后的层次组成 / Post-Clustering Fault-Layer Composition')
    block('![图4 / Figure 4. 簇内注入层次组成 / Injected-layer composition within each cluster.](figures/fig4_cluster_layer_composition.png)')
    pair('图4以每个簇的运行数为分母显示F0–F4的占比，表8同时给出计数、比例和熵。簇内比例P(layer|cluster)与同层在各簇比例P(cluster|layer)分别保存，避免混淆分母。[PDF](figures/fig4_cluster_layer_composition.pdf)。',
         'Figure 4 shows F0–F4 proportions with each cluster size as the denominator. Table 8 also provides counts and entropies. Within-cluster proportions P(layer|cluster) and the distribution of each layer across clusters P(cluster|layer) are stored separately to avoid denominator ambiguity. [PDF](figures/fig4_cluster_layer_composition.pdf).')
    block('**表8 / Table 8. 层次计数、簇内占比与混合熵 / Layer counts, within-cluster shares, and mixing entropy.**\n\n'+composition_table)
    block('**表9 / Table 9. 同层次运行在簇间的分布 / Distribution of each layer across clusters.**\n\n'+dispersion_table)
    pair(f'{layer_mix_zh}，且{spread_zh}。整体层次AMI为{metrics["layer_ami"]:.4f}，表明当前二簇分区与注入层次标签的整体对应关系很弱。加权簇内层次熵为{metrics["weighted_cluster_entropy_bits"]:.3f} bit，与总体层次熵{metrics["overall_layer_entropy_bits"]:.3f} bit较接近，反映分群后仍保留大量层次混合。',
         f'{layer_mix_en}, and {spread_en}. Overall layer AMI is {metrics["layer_ami"]:.4f}, indicating weak overall alignment between the current cluster partition and the injected-layer labels. Weighted within-cluster layer entropy is {metrics["weighted_cluster_entropy_bits"]:.3f} bits, close to the overall layer entropy of {metrics["overall_layer_entropy_bits"]:.3f} bits, reflecting substantial residual layer mixing after clustering.')
    pair('这里的弱对应不等于“特征不含任何层次信息”。聚类数量、距离、规模加权、类别不均衡与算法选择都会影响AMI，而且比较的是当前簇划分与注入层次标签。结果只说明当前表征和划分没有自然恢复注入层次，不能证明任意特征、任意分类器或所有根因定位方法都无法区分层次。',
         'Weak alignment does not mean that the features contain no layer information. AMI depends on cluster count, distance, scale weighting, class imbalance, and algorithm choice; the comparison here is between the current cluster partition and injected-layer labels. The result establishes only that the present representation and partition do not naturally recover injected layers. It does not prove that arbitrary features, classifiers, or all root-cause methods cannot distinguish them.')
    heading('## 5. 混杂核查 / Confounding Diagnostics')
    pair('聚类混合了不同已记录策略和模式，以发现服务模式；统计分析则显式核查条件构成。表10给出簇与各因素的后验AMI，未知策略或配置在构成表中保留，但计算该因素AMI时排除未知。任务列表workload_id来自task_indices，边缘部署标识来自edge_services；task_class_profile仅表示该次运行出现的SIMPLE/COMPLEX类别组合，不直接编码混合比例，计数另存task_class_counts_json，也不保证任务语义相同。',
         'Clustering pools recorded strategies and modes to explore service patterns, while statistical diagnostics explicitly examine condition composition. Table 10 reports post-hoc AMI with each factor. Unknown strategy/configuration values remain visible in composition tables but are excluded when computing AMI for that factor. Workload identifiers derive from task_indices and edge-deployment identifiers from edge_services; task_class_profile records which SIMPLE/COMPLEX categories occur, not their mixture proportions; counts are retained separately in task_class_counts_json, and semantic equivalence is not established.')
    block('**表10 / Table 10. 聚类与层次及实验条件的后验关联 / Post-hoc association with layers and experimental conditions.**\n\n'+covariate_table)
    pair(f'已知协作策略与簇的AMI为{topology_ami:.4f}，高于整体层次AMI {metrics["layer_ami"]:.4f}，提示策略构成是不可忽略的潜在混杂。但策略AMI仅用{topology_n}个已知策略运行，而层次AMI用{nc}个运行；不同因素的有效样本量、类别数与分布不同，不能将这些数值当可直接排序的因果影响大小，也不能断言簇完全由策略决定。',
         f'AMI with known collaboration strategy is {topology_ami:.4f}, higher than the overall layer AMI of {metrics["layer_ami"]:.4f}, flagging strategy composition as a potentially important confounder. However, strategy AMI uses only {topology_n} known-strategy runs, whereas layer AMI uses {nc} runs. Factors differ in sample coverage, category counts, and distributions; their AMIs must not be treated as directly ranked causal effect sizes or evidence that strategy completely determines the clusters.')
    block('**表11 / Table 11. 基础条件内的组成核查 / Composition diagnostics within basic conditions.**\n\n'+condition_table)
    pair(f'{metrics["n_condition_strata_with_mixed_layer_cluster"]}个基础条件内至少一个簇包含多个层次，其中{metrics["n_known_condition_strata_with_mixed_layer_cluster"]}个条件的策略、任务列表与边缘部署元数据已记录。这表明跨层次同簇也出现在已记录的相同策略/模式/负载内部，但同条件仍可能伴随配置、故障类型和实际任务执行差异。更细的条件×任务列表×边缘部署核查见附录CSV；未知common不当作严格已控制条件。',
         f'At least one cluster contains multiple layers in {metrics["n_condition_strata_with_mixed_layer_cluster"]} basic conditions, including {metrics["n_known_condition_strata_with_mixed_layer_cluster"]} conditions with recorded strategy, task-list, and edge-deployment metadata. Cross-layer co-clustering therefore also occurs within recorded strategy/mode/load conditions. Nonetheless, configurations, fault types, and realized executions can still vary within them. Finer condition × task-list × edge-deployment diagnostics are linked in the appendix, and unknown common metadata is not treated as strict experimental control.')
    pair(f'条件内关联推断的工作要求为：条件已记录、n≥20、至少两层和两簇、各边际类别至少5个样本。本轮基础条件包含{int(base.n_runs.min())}–{int(base.n_runs.max())}次运行，均不满足n门槛，因此仅提供组成和描述性AMI，不报告显著性。代码保留未来满足条件时的999次探索性置换AMI检验，但本轮未执行；若未来执行，还需处理多重比较。',
         f'The working requirements for within-condition inference are recorded conditions, n≥20, at least two layers and two clusters, and at least five samples in each marginal category. Basic conditions in this cohort contain {int(base.n_runs.min())}–{int(base.n_runs.max())} runs and do not meet the sample-size threshold. We therefore report composition and descriptive AMI without significance claims. The code retains a 999-permutation exploratory AMI test for future eligible strata, but it is not executed here; future use would also require multiple-comparison treatment.')
    heading('## 6. 讨论与有效性边界 / Discussion and Validity Boundaries')
    pair('**启示2：**不同层次故障的局部异常表现存在异构性，同时可能出现跨层次症状重叠。仅依据局部异常指标难以建立稳定的故障层次与异常表现映射关系，需要进一步结合故障所在层次及其系统依赖关系进行根因定位。当前直接证据来自条件内描述统计，以及异常服务身份与数量聚类中的跨层混合；聚类未评价所有局部异常指标的可辨识性。云边部署、共享模型依赖和Agent执行关系是后续根因定位需要结合的系统依赖信息，其带来的性能增益尚未在本文中验证。',
         '**Insight 2:** Faults at different layers exhibit heterogeneous local manifestations, while symptoms may overlap across layers. Local anomaly indicators alone are insufficient to establish a stable mapping between fault layers and observed manifestations. Root-cause localization therefore needs to incorporate fault-layer context and system dependency relationships. The direct evidence here consists of condition-stratified descriptive statistics and cross-layer mixing in abnormal-service identity/count clustering; this clustering does not evaluate identifiability from every local anomaly indicator. Cloud–edge deployment, shared model dependencies, and agent execution relationships provide dependency information for subsequent root-cause localization, whose performance gains are not validated in this study.')
    pair('构念有效性：过程错误标记不能完整覆盖纯延迟或语义损害；逻辑服务集合也不是实例、网络段或独立根因集合。数量项与二值编码冗余，当前权重是明确的设计选择，尚未通过消融验证其必要性。较大或较长的运行可能有更多机会累积异常服务，当前绝对计数未按暴露机会归一化。',
         'Construct validity: process-error markers may miss pure latency or semantic damage. Logical-service sets are not instance sets, network-segment sets, or independent root-cause sets. Count is redundant with binary membership, and its weighting is an explicit design choice whose necessity has not been established by ablation. Larger or longer runs may have more opportunities to accumulate abnormal services, and the current absolute counts are not normalized for exposure opportunity.')
    pair('内部有效性：执行图缺少绝对span时间，异常实体与注入根因未逐项因果关联，可能包含背景或窗外错误；同条件内仍可能存在部署、模型版本、任务构成和注入位置差异。缺图的4次运行退出聚类也可能造成选择偏差。source实例定位不完整，因此不能声称物理节点或网络段传播已经验证。',
         'Internal validity: absolute span timestamps are unavailable, and anomalous entities are not causally linked individually to injected roots, allowing background or off-window errors. Deployment, model version, workload composition, and injection location may differ even within a condition. Excluding four incomplete-graph runs may also introduce selection bias. Source instances are not fully localized, so physical-node or network-segment propagation is not established.')
    pair(f'统计与外部有效性：bootstrap重用现有运行，只反映经验样本扰动，不能替代独立重新采集；相同任务列表或采集块还可能使运行之间存在依赖。当前仅有一个负载水平，F0只有{f0_count}次聚类样本，故障类型覆盖不均衡；没有正常运行对照，不能评价正常/异常二分类Precision、Recall、F1或AUPRC。轮廓系数、稳定性和AMI均不是根因定位准确率，当前结果不能推广到所有工作负载、所有层次故障或所有RCA方法。',
         f'Statistical and external validity: bootstrap reuses existing runs and characterizes empirical perturbations rather than independent recollection; shared task lists or collection blocks may also induce dependencies among runs. The data cover one load level, only {f0_count} F0 clustering samples, and uneven fault-type coverage. There are no normal controls, so normal-versus-fault Precision, Recall, F1, or AUPRC cannot be evaluated. Silhouette, stability, and AMI are not root-cause localization accuracy, and the findings do not generalize to all workloads, layer-specific faults, or RCA methods.')
    heading('## 7. 结论 / Conclusion')
    pair(f'基于{n}次故障运行的条件内描述统计，以及{nc}次运行的异常服务集合与数量聚类，本文观察到故障表现的同层异构性与跨层混合。K={metrics["n_clusters"]}的平均重采样一致性达到本实验工作门槛，但仍存在明显波动；其与F0–F4的AMI为{metrics["layer_ami"]:.4f}，说明当前服务模式分区未自然恢复注入层次。该证据支持进一步研究结合故障层次、云边部署与Agent动态执行依赖的根因定位方法，而不支持统一严重程度标签、检测准确率提升或已验证因果传播的结论。',
         f'Condition-stratified statistics from {n} fault runs and abnormal-service set/count clustering of {nc} runs reveal within-layer heterogeneity and cross-layer mixing. The mean resampling agreement for K={metrics["n_clusters"]} meets the chosen working threshold but exhibits substantial variation. Its AMI of {metrics["layer_ami"]:.4f} with F0–F4 shows that the current service-pattern partition does not naturally recover injected layers. This evidence motivates root-cause methods that incorporate fault layers, cloud–edge deployment, and dynamic agent execution dependencies; it does not establish universal severity labels, improved detection accuracy, or verified causal propagation.')
    heading('## 附录A：复现与参数 / Appendix A: Reproduction and Parameters')
    pair('在仓库根目录运行下列命令，可从原始数据重建所有表格、两张图以及本中英文README。seed、词表、固定权重、输入指纹和软件版本保存在experiment_config.json；本次写作只读取已保存结果，没有重新计算或调整聚类。',
         'Run the following commands from the repository root to rebuild the tables, two figures, and this bilingual README from raw data. The seed, vocabulary, fixed weights, input fingerprint, and software versions are stored in experiment_config.json. This writing revision reads saved results only and does not recompute or tune clustering.')
    block('```bash\nMPLCONFIGDIR=/private/tmp/fault-layers-matplotlib .venv/bin/python analysis/fault_layers/analyze.py\nMPLCONFIGDIR=/private/tmp/fault-layers-matplotlib .venv/bin/python -m unittest discover -s analysis/fault_layers -p \'test_*.py\' -v\n```')
    pair('可用`--extra-data-root /path/to/labelled_data`追加带标签数据；再次运行会重新扫描实际源目录，因此新增完整数据时样本量可能更新。bootstrap的固定词表和缩放权重在同一轮内保持一致，不依据后验层次结果调整。',
         'Additional labeled data can be supplied with `--extra-data-root /path/to/labelled_data`. Reproduction rescans actual source directories, so newly complete collections may change sample counts. The bootstrap vocabulary and scaling remain fixed within a run of the analysis and are not adjusted using post-hoc layer results.')
    block(markdown_table(['参数 / Parameter','当前值 / Current value'],[
        ['Observation unit','One fault collection run'],['Clustering window','Complete collected execution graphs'],
        ['Features',f'{v} binary service indicators + one count'],['Scaling',r'membership / √V; count / V'],
        ['Clustering','Ward agglomerative; Euclidean; K=2,3,4,5'],['Bootstrap / seed','100 / 20261111'],
        ['Working stability threshold','Mean bootstrap ARI ≥ 0.75'],['PCA','2 dimensions; display only'],
        *[[name,version] for name,version in config['versions'].items()],
        ['Input SHA-256',config['input_fingerprint_sha256']]]))
    pair('完整服务词表如下。二值列名为`service__<完整服务名>`，计数列名为`abnormal_service_count`，两者构成唯一的实际聚类输入。',
         'The complete service vocabulary follows. Binary columns are named `service__<full-service-name>`, and the count column is `abnormal_service_count`; together they form the sole clustering input.')
    block('\n'.join('- `'+service+'`' for service in services))
    pair(f'以下{n-nc}次图覆盖不足运行保留在描述统计中，但不进入聚类。完整排除审计与质量字段见对应CSV。',
         f'The following {n-nc} runs with incomplete graph coverage remain in descriptive statistics but are excluded from clustering. The corresponding CSV files retain the complete exclusion audit and quality fields.')
    block(coverage_table)
    heading('## 附录B：数据文件与结果追溯 / Appendix B: Data and Result Traceability')
    block(markdown_table(['文件 / File','内容 / Contents'],[
        ['[layer_statistics_table.md](layer_statistics_table.md)','所有条件的十指标均值±SD表 / All-condition ten-measure mean±SD table'],
        ['[layer_statistics.csv](tables/layer_statistics.csv)','n、有效n、均值、SD、中位数、Q25/Q75、IQR、min/max / Counts and full descriptive summaries'],
        ['[layer_statistics_by_workload.csv](tables/layer_statistics_by_workload.csv)','任务列表与边缘部署细分 / Task-list and edge-deployment strata'],
        ['[layer_statistics_by_task_class.csv](tables/layer_statistics_by_task_class.csv)','实际提交任务类别构成细分 / Observed submission-category strata'],
        ['[run_metrics.csv](tables/run_metrics.csv)','每运行指标与异常服务集合；early_字段不用于聚类 / Run metrics and sets; retained early_ fields are excluded from clustering'],
        ['[task_metrics.csv](tables/task_metrics.csv)','逐任务过程信号、记录图实体与日志出处 / Task-level signals, recorded entities, and log provenance'],
        ['[source_audit.csv](tables/source_audit.csv)','源目录纳入/排除、计数缺失状态 / Source inclusion, exclusion, and counter status'],
        ['[verified_window_hits.csv](tables/verified_window_hits.csv)','已验证故障窗命中审计，不作聚类特征 / Verified in-window hit audit, not features'],
        ['[clustering_exclusions.csv](tables/clustering_exclusions.csv)','聚类质量排除及原因 / Clustering-quality exclusions'],
        ['[service_feature_matrix.csv](tables/service_feature_matrix.csv)','唯一实际输入：服务二值位与计数 / Sole input: memberships and count'],
        ['[cluster_candidates.csv](tables/cluster_candidates.csv)','K候选、轮廓与稳定性 / K candidates, silhouette, and stability'],
        ['[bootstrap_stability.csv](tables/bootstrap_stability.csv)','每次抽样索引、ARI、唯一运行数 / Resampling indices, ARI, and unique-run count'],
        ['[cluster_assignments.csv](tables/cluster_assignments.csv)','运行路径、原/缩放输入、簇、PCA和后验元数据 / Run paths, inputs, clusters, PCA, metadata'],
        ['[clustering_model.json](tables/clustering_model.json)','词表、权重、K选择与PCA变换 / Vocabulary, weights, K rule, PCA transform'],
        ['[cluster_profiles.csv](tables/cluster_profiles.csv)','每簇原始特征分布 / Per-cluster raw feature distributions'],
        ['[cluster_service_prevalence.csv](tables/cluster_service_prevalence.csv)','各服务在每簇的计数与比例 / Service prevalence by cluster'],
        ['[cluster_layer_composition.csv](tables/cluster_layer_composition.csv)','层次计数与两个方向的比例 / Layer counts and both conditional proportions'],
        ['[cluster_layer_entropy.csv](tables/cluster_layer_entropy.csv)','簇内层次熵及归一化 / Layer entropy and normalization'],
        ['[cluster_covariate_association.csv](tables/cluster_covariate_association.csv)','后验因素AMI / Post-hoc factor AMI'],
        ['[cluster_covariate_composition.csv](tables/cluster_covariate_composition.csv)','各混杂因素的簇内构成 / Within-cluster factor composition'],
        ['[conditional_cluster_association.csv](tables/conditional_cluster_association.csv)','基础与任务/部署细条件核查 / Basic and finer conditional diagnostics'],
        ['[condition_layer_cluster_spread.csv](tables/condition_layer_cluster_spread.csv)','条件内同层次簇分散 / Within-condition layer dispersion'],
        ['[clustering_metrics.json](tables/clustering_metrics.json)','当前总指标 / Current aggregate results'],
        ['[experiment_config.json](experiment_config.json)','算法参数、版本、输入指纹 / Parameters, versions, and fingerprint'],
        ['[csv_schema.md](csv_schema.md)','各中间字段的详细定义 / Detailed intermediate-file schema']]))
    pair(f'历史分级实验已停用；上一版五个早期局部特征聚类位于[独立归档](archive/early_local_clustering_20261111/ARCHIVE_NOTICE.md)。其100例、91/9簇和相应分数不与本文{nc}例、服务集合/数量特征混用。原始数据与当前数值结果在此次写作中保持不变。',
         f'Historical grading experiments are retired. The preceding five-feature early-local clustering is kept in a [separate archive](archive/early_local_clustering_20261111/ARCHIVE_NOTICE.md). Its 100 runs, 91/9 clusters, and scores are not mixed with the present {nc}-run service-set/count analysis. Raw data and the current numerical results remain unchanged by this writing revision.')
    heading('## 参考文献 / References')
    block('1. Ward, J. H., Jr. (1963). **Hierarchical Grouping to Optimize an Objective Function**. *Journal of the American Statistical Association*, 58(301), 236–244. [Publisher / DOI](https://www.tandfonline.com/doi/abs/10.1080/01621459.1963.10500845).\n\n'
          '2. Rousseeuw, P. J. (1987). **Silhouettes: A graphical aid to the interpretation and validation of cluster analysis**. *Journal of Computational and Applied Mathematics*, 20, 53–65. [Author-hosted paper](https://wis.kuleuven.be/statdatascience/robust/papers/publications-1987/rousseeuw-silhouettes-jcam-sciencedirectopenarchiv.pdf).\n\n'
          '3. Hubert, L., & Arabie, P. (1985). **Comparing partitions**. *Journal of Classification*, 2, 193–218. [Publisher / DOI](https://link.springer.com/article/10.1007/BF01908075).\n\n'
          '4. Vinh, N. X., Epps, J., & Bailey, J. (2010). **Information Theoretic Measures for Clusterings Comparison: Variants, Properties, Normalization and Correction for Chance**. *Journal of Machine Learning Research*, 11(95), 2837–2854. [Journal page](https://jmlr.org/papers/v11/vinh10a.html).\n\n'
          '实现定义 / Implementation definitions: [Ward](https://scikit-learn.org/1.5/modules/generated/sklearn.cluster.AgglomerativeClustering.html), [Silhouette](https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.silhouette_score.html), [ARI](https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.adjusted_rand_score.html), [AMI](https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.adjusted_mutual_info_score.html).')
    (HERE/'README.md').write_text('\n\n'.join(parts)+'\n')
    (HERE/'fault_layer_scope_report.md').write_text('# 动机二 / Motivation II\n\n中英文对照论文式正文、实验结果与复现附录统一见[README](README.md)。\n\nThe bilingual manuscript, experimental results, and reproduction appendices are maintained in [README](README.md).\n')
