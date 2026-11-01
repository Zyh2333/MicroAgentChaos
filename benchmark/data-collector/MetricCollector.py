import os
import Config
import pandas as pd
from util.PrometheusClient import PrometheusClient
from util.KubernetesClient import KubernetesClient


def _write_metric_csv(metric_df: pd.DataFrame, path: str, is_header: bool, missing_value=-1):
    """Write metrics after replacing every NaN-like value with its sentinel."""
    metric_df = metric_df.copy()
    metric_columns = [column for column in metric_df.columns if column != 'timestamp']
    metric_df[metric_columns] = metric_df[metric_columns].replace(
        r'(?i)^\s*nan\s*$', float('nan'), regex=True
    ).fillna(missing_value)
    metric_df.to_csv(path, index=False, mode='a', header=is_header)


def _prometheus_matrix(responses):
    """Join Prometheus range series on their actual sample timestamps."""
    columns = {}
    for results, suffix, name_of in responses:
        for result in results:
            values = result['values']
            if not values:
                continue
            name = name_of(result['metric']) + suffix
            columns[name] = pd.Series(
                [float(value) for _, value in values],
                index=pd.to_datetime([timestamp for timestamp, _ in values], unit='s'),
            )
    if not columns:
        return pd.DataFrame(columns=['timestamp'])
    frame = pd.DataFrame(columns).sort_index()
    frame.index.name = 'timestamp'
    return frame.reset_index()


def collect_graph(config: Config, _dir: str, is_header: bool):
    graph_df = pd.DataFrame(columns=['source', 'destination'])
    prom_util = PrometheusClient(config)
    # prom_sql = 'sum(istio_tcp_received_bytes_total{destination_workload_namespace=\"%s\"}) by (source_workload, destination_workload)' % config.namespace
    # results = prom_util.execute_prom(config.prom_range_url, prom_sql)

    prom_sql = 'sum(istio_requests_total{destination_workload_namespace=\"%s\"}) by (source_workload, destination_workload)' % config.namespace
    results = prom_util.execute_prom(config.prom_range_url, prom_sql)
    # results = results + prom_util.execute_prom(config.prom_range_url, prom_sql)

    for result in results:
        metric = result['metric']
        source = metric['source_workload']
        destination = metric['destination_workload']
        # config.svcs.add(source)
        # config.svcs.add(destination)
        config.svcs = KubernetesClient(config).get_svc_list_name()
        values = result['values']
        values = list(zip(*values))
        for timestamp in values[0]:
            new_row = pd.DataFrame({'source': [source], 'destination': [destination], 'timestamp': [timestamp]})
            graph_df = pd.concat([graph_df, new_row], ignore_index=True)

    prom_sql = 'sum(container_cpu_usage_seconds_total{namespace=\"%s\", container!~\'POD|istio-proxy\'}) by (instance, pod)' % config.namespace
    results = prom_util.execute_prom(config.prom_range_url_node, prom_sql)

    for result in results:
        metric = result['metric']
        if 'pod' in metric:
            source = metric['pod']
            config.pods.add(source)
            destination = metric['instance']
            # if node ip
            if ":" in destination:
                destination = destination.split(":")[0]
                for node in KubernetesClient(config).get_nodes():
                    if node.ip == destination:
                        destination = node.name
                        break
            values = result['values']
            values = list(zip(*values))
            for timestamp in values[0]:
                new_row = pd.DataFrame({'source': [source], 'destination': [destination], 'timestamp': [timestamp]})
                graph_df = pd.concat([graph_df, new_row], ignore_index=True)
                # graph_df = graph_df.append({'source': source, 'destination': destination, 'timestamp': timestamp},
                #                             ignore_index=True)

    graph_df['timestamp'] = graph_df['timestamp'].astype('datetime64[s]')
    graph_df = graph_df.sort_values(by='timestamp', ascending=True)
    path = os.path.join(_dir, 'graph.csv')
    graph_df.to_csv(path, index=False, mode='a', header=is_header)


# Get the response time of the invocation edges
def collect_call_latency(config: Config, _dir: str, is_header: bool):
    prom_util = PrometheusClient(config)
    # P50，P90，P99
    prom_50_sql = 'histogram_quantile(0.50, sum(irate(istio_request_duration_milliseconds_bucket{destination_workload_namespace=\"%s\"}[1m])) by (destination_workload, destination_workload_namespace, source_workload, le))' % config.namespace
    prom_90_sql = 'histogram_quantile(0.90, sum(irate(istio_request_duration_milliseconds_bucket{destination_workload_namespace=\"%s\"}[1m])) by (destination_workload, destination_workload_namespace, source_workload, le))' % config.namespace
    prom_99_sql = 'histogram_quantile(0.99, sum(irate(istio_request_duration_milliseconds_bucket{destination_workload_namespace=\"%s\"}[1m])) by (destination_workload, destination_workload_namespace, source_workload, le))' % config.namespace
    responses_50 = prom_util.execute_prom(config.prom_range_url, prom_50_sql)
    responses_90 = prom_util.execute_prom(config.prom_range_url, prom_90_sql)
    responses_99 = prom_util.execute_prom(config.prom_range_url, prom_99_sql)

    call_df = _prometheus_matrix([
        (responses_50, '&p50', lambda m: m['source_workload'] + '_' + m['destination_workload']),
        (responses_90, '&p90', lambda m: m['source_workload'] + '_' + m['destination_workload']),
        (responses_99, '&p99', lambda m: m['source_workload'] + '_' + m['destination_workload']),
    ])

    path = os.path.join(_dir, 'call.csv')
    _write_metric_csv(call_df, path, is_header)


# Get the response time for the microservices
def collect_svc_latency(config: Config, _dir: str, is_header: bool):
    prom_util = PrometheusClient(config)
    # P50，P90，P99
    prom_50_sql = 'histogram_quantile(0.50, sum(irate(istio_request_duration_milliseconds_bucket{reporter=\"destination\", destination_workload_namespace=\"%s\"}[1m])) by (destination_workload, destination_workload_namespace, le))' % config.namespace
    prom_90_sql = 'histogram_quantile(0.90, sum(irate(istio_request_duration_milliseconds_bucket{reporter=\"destination\", destination_workload_namespace=\"%s\"}[1m])) by (destination_workload, destination_workload_namespace, le))' % config.namespace
    prom_99_sql = 'histogram_quantile(0.99, sum(irate(istio_request_duration_milliseconds_bucket{reporter=\"destination\", destination_workload_namespace=\"%s\"}[1m])) by (destination_workload, destination_workload_namespace, le))' % config.namespace
    responses_50 = prom_util.execute_prom(config.prom_range_url, prom_50_sql)
    responses_90 = prom_util.execute_prom(config.prom_range_url, prom_90_sql)
    responses_99 = prom_util.execute_prom(config.prom_range_url, prom_99_sql)

    latency_df = _prometheus_matrix([
        (responses_50, '&p50', lambda m: m['destination_workload']),
        (responses_90, '&p90', lambda m: m['destination_workload']),
        (responses_99, '&p99', lambda m: m['destination_workload']),
    ])

    path = os.path.join(_dir, 'latency.csv')
    _write_metric_csv(latency_df, path, is_header)


# Get the vCPU and memory usage of the machine
def collect_resource_metric(config: Config, _dir: str, is_header: bool):
    metric_df = pd.DataFrame()
    vCPU_sql = 'sum(rate(container_cpu_usage_seconds_total{image!="",namespace="%s"}[1m]))' % config.namespace
    mem_sql = 'sum(rate(container_memory_usage_bytes{image!="",namespace="%s"}[1m])) / (1024*1024)' % config.namespace
    prom_util = PrometheusClient(config)
    vCPU = prom_util.execute_prom(config.prom_range_url_node, vCPU_sql)
    mem = prom_util.execute_prom(config.prom_range_url_node, mem_sql)

    def handle(result, metric_df, col):
        values = result['values']
        values = list(zip(*values))
        if 'timestamp' not in metric_df:
            timestamp = values[0]
            metric_df['timestamp'] = timestamp
            metric_df['timestamp'] = metric_df['timestamp'].astype('datetime64[s]')
        metric = values[1]
        metric_df[col] = pd.Series(metric)
        metric_df[col] = metric_df[col].fillna(-1)
        metric_df[col] = metric_df[col].astype('float64')

    [handle(result, metric_df, 'vCPU') for result in vCPU]
    [handle(result, metric_df, 'memory') for result in mem]

    path = os.path.join(_dir, 'resource.csv')
    _write_metric_csv(metric_df, path, is_header)


# Get the number of pods for all microservices
def collect_pod_num(config: Config, _dir: str, is_header: bool):
    prom_util = PrometheusClient(config)
    pod_sql = 'kube_pod_info{namespace="%s"}' % config.namespace
    response = prom_util.execute_prom(config.prom_range_url_node, pod_sql)
    pod_frames = []
    for result in response:
        if 'created_by_name' in result['metric'] and 'pod_ip' in result['metric']:
            owner = result['metric']['created_by_name']
            service = owner.rsplit('-', 1)[0] if '-' in owner else owner
            values = result['values']
            if not values:
                continue
            pod_frames.append(pd.DataFrame({
                'timestamp': pd.to_datetime([value[0] for value in values], unit='s'),
                'service': service + '&count',
                'value': [float(value[1]) for value in values],
            }))

    if pod_frames:
        samples = pd.concat(pod_frames, ignore_index=True)
        instance_num_df = samples.pivot_table(
            index='timestamp', columns='service', values='value',
            aggfunc='sum', fill_value=0,
        ).reset_index().sort_values('timestamp')
        instance_num_df.columns.name = None
    else:
        instance_num_df = pd.DataFrame(columns=['timestamp'])

    path = os.path.join(_dir, 'instances_num.csv')
    _write_metric_csv(instance_num_df, path, is_header)


# get qps for microservice
def collect_svc_qps(config: Config, _dir: str, is_header: bool):
    prom_util = PrometheusClient(config)
    qps_sql = 'sum(rate(istio_requests_total{reporter="destination",namespace="%s"}[30s])) by (destination_workload)' % config.namespace
    response = prom_util.execute_prom(config.prom_range_url, qps_sql)

    qps_df = _prometheus_matrix([(response, '', lambda m: m['destination_workload'])])

    path = os.path.join(_dir, 'svc_qps.csv')
    _write_metric_csv(qps_df, path, is_header)


# Get metric for microservices
def collect_svc_metric(config: Config, _dir: str, is_header: bool):
    prom_util = PrometheusClient(config)
    final_df = prom_util.get_svc_metric_range()
    path = os.path.join(_dir, 'svc_metric.csv')
    _write_metric_csv(final_df, path, is_header)


# Collect container CPU, memory, network
def collect_ctn_metric(config: Config, _dir: str, is_header: bool):
    pod_df = pd.DataFrame()
    prom_util = PrometheusClient(config)

    pod_info_sql = 'kube_pod_info{namespace=\'%s\'}' % config.namespace
    response = prom_util.execute_prom(config.prom_range_url_node, pod_info_sql)
    for result in response:
        if 'created_by_name' in result['metric'] and 'pod_ip' in result['metric']:
            pod_name = result['metric']['pod']
            container = result['metric']['container']
            if 'istio' in container:
                continue
            if 'horsecoder-pay' in pod_name:
                continue
            values = result['values']
            values = list(zip(*values))
            container_df = pd.DataFrame()
            timestamp = values[0]
            metric = pd.Series(values[1])
            container_df['timestamp'] = timestamp
            container_df['timestamp'] = container_df['timestamp'].astype('datetime64[s]')
            container_df[pod_name] = metric
            container_df = container_df.fillna(0)
            container_df[pod_name] = container_df[pod_name].astype('float64')
            if pod_df.empty:
                pod_df = container_df
            elif pod_name in pod_df.columns:
                pod_df = pod_df.set_index('timestamp').combine_first(container_df.set_index('timestamp')).reset_index()
                for i in range(len(container_df[pod_name])):
                    pod_df_index = pod_df.loc[pod_df['timestamp'] == container_df['timestamp'][i]].index[0]
                    pod_df.at[pod_df_index, pod_name] = 1 if pod_df[pod_name][pod_df_index] == 0 and container_df[pod_name][i] == 1 else pod_df[pod_name][pod_df_index]
            else:
                pod_df = pd.merge(pod_df, container_df, on='timestamp', how='outer')
            pod_df = pod_df.fillna(0)
    pod_df = pod_df.sort_values(by='timestamp')
    pod_df = pod_df.reset_index(drop=True)

    prom_cpu_sql = 'sum(rate(container_cpu_usage_seconds_total{namespace=\'%s\',container!~\'POD|istio-proxy|\',container!~\'POD|rabbitmq-exporter|\',pod!~\'jaeger.*\'}[1m])* 1000)  by (pod, instance, container)' % config.namespace
    prom_memory_sql = 'sum(container_memory_working_set_bytes{namespace=\'%s\',container!~\'POD|istio-proxy|\',container!~\'POD|rabbitmq-exporter|\',pod!~\'jaeger.*\'}) by(pod, instance, container)  / 1000000' % (
        config.namespace)
    response = prom_util.execute_prom(config.prom_range_url_node, prom_cpu_sql)
    cpu_rename = {}
    cpu_df = pd.DataFrame()
    for result in response:
        pod_name = result['metric']['pod']
        container = result['metric']['container']
        if 'istio' in container:
            continue
        if 'horsecoder-pay' in pod_name:
            continue
        config.pods.add(pod_name)
        values = result['values']
        values = list(zip(*values))
        container_df = pd.DataFrame()
        timestamp = values[0]
        container_df['timestamp'] = timestamp
        container_df['timestamp'] = container_df['timestamp'].astype('datetime64[s]')
        metric = pd.Series(values[1])
        col_name = pod_name + '_cpu'
        cpu_rename[pod_name] = col_name
        container_df[pod_name] = metric
        container_df = container_df.fillna(0)
        container_df[pod_name] = container_df[pod_name].astype('float64')
        if cpu_df.empty:
            cpu_df = container_df
        else:
            cpu_df = pd.merge(cpu_df, container_df, on='timestamp', how='outer')
        cpu_df = cpu_df.fillna(0)
    cpu_df = cpu_df.sort_values(by='timestamp')
    cpu_df = cpu_df.reset_index(drop=True)
    cpu_df = cpu_df.mask(((cpu_df == 0) & (pod_df == 0)).fillna(False).astype(bool), -1)
    cpu_df.rename(columns=cpu_rename, inplace=True)

    mem_rename = {}
    mem_df = pd.DataFrame()
    response = prom_util.execute_prom(config.prom_range_url_node, prom_memory_sql)
    for result in response:
        pod_name = result['metric']['pod']
        container = result['metric']['container']
        if 'istio' in container:
            continue
        if 'horsecoder-pay' in pod_name:
            continue
        config.pods.add(pod_name)
        container_df = pd.DataFrame()
        values = result['values']
        values = list(zip(*values))
        timestamp = values[0]
        container_df['timestamp'] = timestamp
        container_df['timestamp'] = container_df['timestamp'].astype('datetime64[s]')
        metric = pd.Series(values[1])
        col_name = pod_name + '_memory'
        mem_rename[pod_name] = col_name
        container_df[pod_name] = metric
        container_df = container_df.fillna(0)
        container_df[pod_name] = container_df[pod_name].astype('float64')
        if mem_df.empty:
            mem_df = container_df
        else:
            mem_df = pd.merge(mem_df, container_df, on='timestamp', how='outer')
        mem_df = mem_df.fillna(0)
    mem_df = mem_df.sort_values(by='timestamp')
    mem_df = mem_df.reset_index(drop=True)
    mem_df = mem_df.mask(((mem_df == 0) & (pod_df == 0)).fillna(False).astype(bool), -1)
    mem_df.rename(columns=mem_rename, inplace=True)

    net_rename = {}
    net_df = pd.DataFrame()
    for pod_name in config.pods:
        if 'horsecoder-pay' in pod_name:
            continue
        prom_network_sql = 'sum(rate(container_network_transmit_packets_total{namespace=\"%s\", pod="%s"}[1m])) * sum(rate(container_network_transmit_packets_total{namespace=\"%s\", pod="%s"}[1m]))' % (
            config.namespace, pod_name, config.namespace, pod_name)
        response = prom_util.execute_prom(config.prom_range_url_node, prom_network_sql)
        if response == []:
            continue
        container_df = pd.DataFrame()
        values = response[0]['values']
        values = list(zip(*values))
        timestamp = values[0]
        container_df['timestamp'] = timestamp
        container_df['timestamp'] = container_df['timestamp'].astype('datetime64[s]')
        metric = pd.Series(values[1])
        col_name = pod_name + '_network'
        net_rename[pod_name] = col_name
        container_df[pod_name] = metric
        container_df = container_df.fillna(0)
        container_df[pod_name] = container_df[pod_name].astype('float64')
        if net_df.empty:
            net_df = container_df
        else:
            net_df = pd.merge(net_df, container_df, on='timestamp', how='outer')
        net_df = net_df.fillna(0)
    net_df = net_df.sort_values(by='timestamp')
    net_df = net_df.reset_index(drop=True)
    net_df = net_df.mask(((net_df == 0) & (pod_df == 0)).fillna(False).astype(bool), -1)
    net_df.rename(columns=net_rename, inplace=True)

    df = pd.merge(cpu_df, mem_df, on='timestamp', how='outer')
    df = pd.merge(df, net_df, on='timestamp', how='outer')
    path = os.path.join(_dir, 'instance.csv')
    _write_metric_csv(df, path, is_header)
    return df


# Get the success rate for microservices
def collect_succeess_rate(config: Config, _dir: str, is_header: bool):
    prom_util = PrometheusClient(config)
    success_rate_sql = '(sum(rate(istio_requests_total{reporter="destination", response_code!~"5.*",namespace="%s"}[1m])) by (destination_workload, destination_workload_namespace) / sum(rate(istio_requests_total{reporter="destination",namespace="%s"}[1m])) by (destination_workload, destination_workload_namespace))' % (
        config.namespace, config.namespace)
    response = prom_util.execute_prom(config.prom_range_url, success_rate_sql)

    success_df = _prometheus_matrix([(response, '', lambda m: m['destination_workload'])])

    path = os.path.join(_dir, 'success_rate.csv')
    _write_metric_csv(success_df, path, is_header)


def collect_node_metric(config: Config, _dir: str, is_header: bool):
    df = pd.DataFrame()
    prom_util = PrometheusClient(config)
    nodes = KubernetesClient(config).get_nodes()
    for node in nodes:
        prom_sql = 'rate(node_network_transmit_packets_total{device="cni0", instance="%s"}[1m]) / 1000' % node.node_name
        response = prom_util.execute_prom(config.prom_range_url_node, prom_sql)
        # change
        if response == []:
            continue
        values = response[0]['values']
        values = list(zip(*values))
        timestamp = values[0]
        node_df = pd.DataFrame()
        node_df['timestamp'] = timestamp
        node_df['timestamp'] = node_df['timestamp'].astype('datetime64[s]')
        metric = pd.Series(values[1])
        col_name = '(node)' + node.name + '_network'
        node_df[col_name] = metric
        node_df[col_name] = node_df[col_name].astype('float64')
        node_df = node_df.fillna(-1)
        if df.empty:
            df = node_df
        else:
            df = pd.merge(df, node_df, on='timestamp', how='outer')
        df = df.fillna(-1)

        # prom_sql = 'rate(node_network_transmit_packets_total{device="raven0", instance="%s"}[3m]) / 1000' % node.node_name
        # response = prom_util.execute_prom(config.prom_range_url_node, prom_sql)
        # values = response[0]['values']
        # values = list(zip(*values))
        # node_df = pd.DataFrame()
        # node_df['timestamp'] = timestamp
        # node_df['timestamp'] = node_df['timestamp'].astype('datetime64[s]')
        # metric = pd.Series(values[1])
        # col_name = '(node)' + node.name + '_edge_network'
        # node_df[col_name] = metric
        # node_df[col_name] = node_df[col_name].astype('float64')
        # node_df = node_df.fillna(0)
        if df.empty:
            df = node_df
        else:
            df = pd.merge(df, node_df, on='timestamp', how='outer')
        df = df.fillna(-1)

        prom_sql = '1-(sum(increase(node_cpu_seconds_total{instance="%s",mode="idle"}[1m]))/sum(increase(node_cpu_seconds_total{instance="%s"}[1m])))' % (
            node.node_name, node.node_name)
        response = prom_util.execute_prom(config.prom_range_url_node, prom_sql)
        values = response[0]['values']
        values = list(zip(*values))
        node_df = pd.DataFrame()
        node_df['timestamp'] = timestamp
        node_df['timestamp'] = node_df['timestamp'].astype('datetime64[s]')
        metric = pd.Series(values[1])
        col_name = '(node)' + node.name + '_cpu'
        node_df[col_name] = metric
        node_df[col_name] = node_df[col_name].astype('float64')
        node_df = node_df.fillna(-1)
        if df.empty:
            df = node_df
        else:
            df = pd.merge(df, node_df, on='timestamp', how='outer')
        df = df.fillna(-1)

        prom_sql = '(node_memory_MemTotal_bytes{instance="%s"}-(node_memory_MemFree_bytes{instance="%s"}+ node_memory_Cached_bytes{instance="%s"} + node_memory_Buffers_bytes{instance="%s"})) / node_memory_MemTotal_bytes{instance="%s"}' % (
            node.node_name, node.node_name, node.node_name, node.node_name, node.node_name)
        response = prom_util.execute_prom(config.prom_range_url_node, prom_sql)
        values = response[0]['values']
        values = list(zip(*values))
        node_df = pd.DataFrame()
        node_df['timestamp'] = timestamp
        node_df['timestamp'] = node_df['timestamp'].astype('datetime64[s]')
        metric = pd.Series(values[1])
        col_name = '(node)' + node.name + '_memory'
        node_df[col_name] = metric
        node_df[col_name] = node_df[col_name].astype('float64')
        node_df = node_df.fillna(-1)
        if df.empty:
            df = node_df
        else:
            df = pd.merge(df, node_df, on='timestamp', how='outer')
        df = df.fillna(-1)

    path = os.path.join(_dir, 'node.csv')
    _write_metric_csv(df, path, is_header)

    return df


def collect(config: Config, _dir: str, is_header: bool):
    print('collect metrics')
    # Create the folder
    if not os.path.exists(_dir):
        os.makedirs(_dir)
    # Collect the various data
    collect_graph(config, _dir, is_header)
    collect_call_latency(config, _dir, is_header)
    collect_svc_latency(config, _dir, is_header)
    collect_resource_metric(config, _dir, is_header)
    collect_succeess_rate(config, _dir, is_header)
    collect_svc_qps(config, _dir, is_header)
    collect_svc_metric(config, _dir, is_header)
    collect_pod_num(config, _dir, is_header)
    collect_ctn_metric(config, _dir, is_header)


def collect_node(config: Config, _dir: str, is_header: bool):
    print('collect node metrics')
    if not os.path.exists(_dir):
        os.makedirs(_dir)
    collect_node_metric(config, _dir, is_header)
