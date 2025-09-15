# 数据说明

## 总体概述

本数据集包含系统采集的三类关键数据：监控指标（Metric）、分布式调用链（Trace）和容器日志（Log）。数据均以 Parquet 格式存储，以便高效读取与分析。整体结构如下：

```text
├── log-parquet
│   ├── log_filebeat-server_2025-05-27_00-00-00.parquet
│   ├── log_filebeat-server_2025-05-27_01-00-00.parquet
│   └── log_filebeat-server_2025-05-27_02-00-00.parquet
├── metric-parquet
│   ├── apm
│   │   ├── pod
│   │   │   └── pod_adservice-0_2025-05-27.parquet
│   │   ├── pod_ns_hipstershop_2025-05-27.parquet
│   │   └── service
│   │       └── service_adservice_2025-05-27.parquet
│   ├── infra
│   │   ├── infra_node
│   │   │   ├── infra_node_node_cpu_usage_rate_2025-05-27.parquet
│   │   │   └── infra_node_node_disk_read_bytes_total_2025-05-27.parquet
│   │   ├── infra_pod
│   │   │   ├── infra_pod_pod_cpu_usage_2025-05-27.parquet
│   │   │   └── infra_pod_pod_fs_reads_bytes_2025-05-27.parquet
│   │   └── infra_tidb
│   │       ├── infra_tidb_block_cache_size_2025-05-27.parquet
│   │       └── infra_tidb_connection_count_2025-05-27.parquet
│   └── other
│       ├── infra_pd_abnormal_region_count_2025-05-27.parquet
│       ├── infra_pd_leader_count_2025-05-27.parquet
│       └── infra_tikv_available_size_2025-05-27.parquet
└── trace-parquet
    ├── trace_jaeger-span_2025-05-01_00-59-00.parquet
    └── trace_jaeger-span_2025-05-01_01-59-00.parquet
```

接下来将分三大部分详细说明各类数据的存储结构与字段含义。

## 数据文件说明

本节介绍三类数据在文件系统中的组织结构及命名规范，帮助快速定位所需文件。

### Metric

Metric 目录下又分为两级子目录：apm/ 与 infra/，其中还包括一个 other/ 目录，用于存放不属于前两者范畴的监控指标。具体结构如下：

```text
├── metric-parquet
│   ├── apm
│   │   ├── pod
│   │   │   └── pod_adservice-0_2025-05-27.parquet
│   │   ├── pod_ns_hipstershop_2025-05-27.parquet
│   │   └── service
│   │       └── service_adservice_2025-05-27.parquet
│   ├── infra
│   │   ├── infra_node
│   │   │   └── infra_node_node_disk_read_bytes_total_2025-05-27.parquet
│   │   ├── infra_pod
│   │   │   └── infra_pod_pod_fs_reads_bytes_2025-05-27.parquet
│   │   └── infra_tidb
│   │       └── infra_tidb_connection_count_2025-05-27.parquet
│   └── other
│       └── infra_tikv_available_size_2025-05-27.parquet
```

APM 指标 （apm/）

- 目录说明

  应用性能监控（APM）指标由 DeepFlow 系统在集群中采集，按照业务命名空间、Pod、Service 分类存储。

  - 根目录下的文件以 {namespace}_{日期}.parquet 命名，表示该业务命名空间中所有对象（Pod 和 Service）在指定日期的 APM 数据。
  - pod/ 子目录：每个 Pod 对应一个 Parquet 文件，格式为 pod_{podName}_{日期}.parquet，其中 podName 为 Pod 的名称（通常带有副本序号）。
  - service/ 子目录：每个 Service 对应一个 Parquet 文件，格式为 service_{serviceName}_{日期}.parquet，serviceName 为 Service 名称。

Infra 指标（infra/）

- 目录说明

  机器性能指标由 Prometheus 采集，细分为三个子目录：infra_node/、infra_pod/ 和 infra_tidb/。

  - infra_node/：Node 级别的指标，如 CPU、内存、磁盘、网络等；文件名称前缀为 infra_node_node_{kpiKey}_{日期}.parquet。
  - infra_pod/：Pod 级别的指标，如 Pod 的 CPU 使用率、内存使用量、文件系统读写、网络吞吐等；文件命名为 infra_pod_{kpiKey}_{日期}.parquet，其中 kpiKey 表示具体的指标编码。
  - infra_tidb/：TiDB 组件相关的指标，如连接数、慢查询、Block Cache 大小等；文件命名为 infra_tidb_{kpiKey}_{日期}.parquet。

Other（其他组件）指标（other/）

- 目录说明

  用于存储集群里其他关键组件的指标。

  - 文件通常以 infra_{component}\_{metricKey}_{日期}.parquet 命名，例如 infra_pd_abnormal_region_count_2025-05-27.parquet 表示 PD（Placement Driver）中异常 Region 数量指标。
  - 其他常见示例：infra_pd_leader_count_2025-05-27.parquet（PD Leader 数量）、infra_tikv_available_size_2025-05-27.parquet（TiKV 可用存储容量）等。

### Trace

Trace 目录下的 Parquet 文件以小时为粒度存储了 Jaeger 采集的调用链信息。结构示例：

```text
├── trace-parquet
    ├── trace_jaeger-span_2025-05-01_00-59-00.parquet
    └── trace_jaeger-span_2025-05-01_01-59-00.parquet
```
- 文件命名 

  格式为 trace_jaeger-span_{日期}_{HH}-59-00.parquet，如 trace_jaeger-span_2025-05-27_13-59-00.parquet 表示该文件包含 2025-05-27 当天 13:00–14:00 期间所有采集到的 Span 信息，时区为 CST。

### Log

Log 目录与 Trace 类似，也以小时为单位分文件，存储 Filebeat 从各 Pod 收集的容器日志。本示例只列举部分文件：

```text
├── log-parquet
│   ├── log_filebeat-server_2025-05-27_00-00-00.parquet
│   ├── log_filebeat-server_2025-05-27_01-00-00.parquet
│   └── log_filebeat-server_2025-05-27_02-00-00.parquet
```

- 文件命名

  格式为 log_filebeat-server_{日期}_{HH}-00-00.parquet，如 log_filebeat-server_2025-05-27_13-59-00.parquet 表示 2025-05-27 13:00–14:00 时间段收集到的所有日志，时区为 CST。

## 数据格式说明

接下来针对各类别数据的字段含义、典型示例及采集粒度进行详细说明，以便后续数据清洗、建模和可视化分析。

### Metric

Metric 数据既包括业务指标（APM）也包括性能指标（Infra），本节分为两部分说明。

#### 业务指标 （APM指标）

APM 指标主要反映业务服务在一定时间窗口内的请求与响应情况，包括错误率、时延等核心指标。主要字段和含义如下表所示。

| 指标编码             | 指标名称       | 指标粒度 |
| -------------------- | -------------- | -------- |
| request              | 请求数量       | 60       |
| response             | 响应数量       | 60       |
| rrt                  | 平均时延       | 60       |
| rrt_max              | 最大时延       | 60       |
| error                | 异常           | 60       |
| client_error         | 客户端异常     | 60       |
| server_error         | 服务端异常     | 60       |
| timeout              | 超时           | 60       |
| error_ratio          | 异常比例       | 60       |
| client_error_ratio   | 客户端异常比例 | 60       |
| server_error_ratio   | 服务端异常比例 | 60       |


>示例数据（部分）
>| time                  | client_error | client_error_ratio | error | error_ratio | object_id     | object_type | request | response |     rrt | rrt_max | server_error | server_error_ratio | timeout |
>| --------------------- | ------------ | ------------------ | ----- | ----------- | ------------- | ----------- | ------- | -------- | ------- | ------- | ------------ | ------------------ | ------- |
>| 2025-05-05 16:04:00+00:00  | 0            | 0.00               | 0     | 0.00        | adservice-0   | pod         | 325     | 328      | 3661.83 | 43319   | 0            | 0                  | 0       |
>| 2025-05-05 16:04:00+00:00  | 11           | 2.68               | 11    | 2.68        | adservice-0   | pod         | 410     | 411      | 3708.56 | 43864   | 0            | 0                  | 0       |
>| 2025-05-05 16:04:00+00:00  | 0            | 0.00               | 0     | 0.00        | adservice-0   | pod         | 319     | 320      | 4140.42 | 51831   | 0            | 0                  | 0       |
>| 2025-05-05 16:04:00+00:00  | 6            | 1.44               | 6     | 1.44        | adservice-0   | pod         | 412     | 416      | 3401.22 | 43868   | 0            | 0                  | 0       |
>| 2025-05-05 16:04:00+00:00  | 0            | 0.00               | 0     | 0.00        | adservice-0   | pod         | 294     | 296      | 3480.99 | 43820   | 0            | 0                  | 0       |

以上示例展示 adservice-0 Pod 在不同分钟的业务调用情况。当需要计算某个时段的平均时延或错误率，可以按分钟粒度聚合 rrt、error_ratio 等字段。

#### 性能指标（Infra指标）

性能指标反映系统的底层资源使用情况，包括 Pod、Node、TiDB 组件在内的多种对象。下表列出了常见指标编码及其含义（以 Parquet 文件中的 kpi_key 为主）：

| object_type | kpi_key                          | kpi_name               | promql                                                                                                                      |
|-------------|----------------------------------|------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| pod         | pod_cpu_usage                    | CPU 使用率             | `sum(irate(container_cpu_usage_seconds_total{container=~"server\|redis"}[1m])) by (namespace,instance,pod)`               |
| pod         | pod_processes                    | 进程数                 | `sum(container_processes{container=~"server\|redis"}) by (namespace,instance,pod)`                                         |
| pod         | pod_memory_working_set_bytes     | 内存使用大小           | `sum(rate(container_memory_working_set_bytes{container=~"server\|redis"}[1m])) by (namespace,instance,pod)`                |
| pod         | pod_fs_writes_bytes              | 写入字节的累积计数     | `sum(irate(container_fs_writes_bytes_total{container=~"server\|redis"}[1m])) by (namespace,instance,pod,device)`         |
| pod         | pod_fs_reads_bytes               | 累计读取字节数         | `sum(irate(container_fs_reads_bytes_total{container=~"server\|redis"}[1m])) by (namespace,instance,pod,device)`          |
| pod         | pod_network_receive_bytes        | 接收字节的累积计数     | `sum(irate(container_network_receive_bytes_total{namespace="hipstershop",interface="eth0"}[1m])) by (namespace,instance,pod)` |
| pod         | pod_network_transmit_bytes       | 传输字节的累积计数     | `sum(irate(container_network_transmit_bytes_total{namespace="hipstershop",interface="eth0"}[1m])) by (namespace,instance,pod)`|
| pod         | pod_network_receive_packets      | 接收数据包的累积计数   | `sum(irate(container_network_receive_packets_total{namespace="hipstershop",interface="eth0"}[1m])) by (namespace,instance,pod)`|
| pod         | pod_network_transmit_packets     | 传输数据包的累积计数   | `sum(irate(container_network_transmit_packets_total{namespace="hipstershop",interface="eth0"}[1m])) by (namespace,instance,pod)`|
| node        | node_cpu_usage_rate             | CPU使用率      | `100 - (avg(irate(node_cpu_seconds_total{job="kubernetes-nodes",mode="idle"}[1m])) by (kubernetes_node,instance) * 100)`                                                                                                                |
| node        | node_memory_usage_rate          | 内存使用率     | `100 - (node_memory_MemFree_bytes{job="kubernetes-nodes"} + node_memory_Cached_bytes{job="kubernetes-nodes"} + node_memory_Buffers_bytes{job="kubernetes-nodes"}) / node_memory_MemTotal_bytes{job="kubernetes-nodes"} * 100`           |
| node        | node_filesystem_usage_rate      | 磁盘使用率     | `100 - (node_filesystem_free_bytes{job="kubernetes-nodes",fstype=~"ext4\|xfs"} / node_filesystem_size_bytes{job="kubernetes-nodes",fstype=~"ext4\|xfs"} * 100)`                                                                           |
| node        | node_memory_MemAvailable_bytes  | 空闲内存大小   | `node_memory_MemAvailable_bytes{job="kubernetes-nodes"}`                                                                                                                                                                                  |
| node        | node_memory_MemTotal_bytes      | 内存总大小     | `node_memory_MemTotal_bytes{job="kubernetes-nodes"}`                                                                                                                                                                                      |
| node        | node_filesystem_size_bytes      | 空闲磁盘大小   | `node_filesystem_size_bytes{job="kubernetes-nodes",mountpoint=~"/rootfs\|/"}`                                                                                                                                                              |
| node        | node_filesystem_free_bytes      | 磁盘总大小     | `node_filesystem_free_bytes{job="kubernetes-nodes",mountpoint=~"/rootfs\|/"}`     
| node        | node_disk_read_bytes_total             | 成功读取的字节数                                           | `sum(irate(node_disk_read_bytes_total{job="kubernetes-nodes",device="vda"}[1m])) by (kubernetes_node,instance,device)`                                                   |
| node        | node_disk_read_time_seconds_total      | Read time ms 每个磁盘分区读花费的毫秒数                     | `sum(irate(node_disk_read_time_seconds_total{job="kubernetes-nodes",device="vda"}[1m])) by (kubernetes_node,instance,device)`                                           |
| node        | node_disk_written_bytes_total          | 成功写入的字节数                                           | `sum(irate(node_disk_written_bytes_total{job="kubernetes-nodes",device="vda"}[1m])) by (kubernetes_node,instance,device)`                                                |
| node        | node_disk_write_time_seconds_total     | Write time ms 每个磁盘分区写操作花费的毫秒数               | `sum(irate(node_disk_write_time_seconds_total{job="kubernetes-nodes",device="vda"}[1m])) by (kubernetes_node,instance,device)`                                          |
| node        | node_network_receive_bytes_total       | `{{device}}` – Receive 各个网络接口接收速率               | `sum(irate(node_network_receive_bytes_total{job="kubernetes-nodes"}[1m])) by (kubernetes_node,instance)`                                                                 |
| node        | node_network_receive_packets_total     | `{{device}}` – Receive 各个接口每秒接收的数据包总数       | `sum(irate(node_network_receive_packets_total{job="kubernetes-nodes"}[1m])) by (kubernetes_node,instance)`                                                               |
| node        | node_network_transmit_bytes_total      | `{{device}}` – Transmit 各个网络接口发送速率              | `sum(irate(node_network_transmit_bytes_total{job="kubernetes-nodes"}[1m])) by (kubernetes_node,instance)`                                                                |
| node        | node_network_transmit_packets_total    | `{{device}}` – Transmit 各个接口每秒发送的数据包总数      | `sum(irate(node_network_transmit_packets_total{job="kubernetes-nodes"}[1m])) by (kubernetes_node,instance)`                                                              |
| node        | node_sockstat_TCP_inuse                | TCP_inuse – 正在使用（正在侦听）的 TCP 套接字数量         | `sum(irate(node_sockstat_TCP_inuse{job="kubernetes-nodes"}[1m])) by (kubernetes_node,instance)`                                                                          |
| tidb        | connection_count      | 连接数           | `sum(tidb_server_connections) by (namespace,instance)`                                                                                                                                                                                                                |
| tidb        | failed_query_ops      | 失败请求数       | `sum(increase(tidb_server_execute_error_total[1m])) by (namespace,type,instance)`                                                                                                                                                                                    |
| tidb        | duration_99th         | 99 分位请求延迟  | `histogram_quantile(0.99, sum(rate(tidb_server_handle_query_duration_seconds_bucket{sql_type!="internal"}[1m])) by (namespace,instance,le))`                                                                                                                            |
| tidb        | duration_95th         | 95 分位请求延迟  | `histogram_quantile(0.95, sum(rate(tidb_server_handle_query_duration_seconds_bucket{sql_type!="internal"}[1m])) by (namespace,instance,le))`                                                                                                                            |
| tidb        | duration_avg          | 平均请求延迟     | `sum(rate(tidb_server_handle_query_duration_seconds_sum{sql_type!="internal"}[2m])) by (namespace,instance) / sum(rate(tidb_server_handle_query_duration_seconds_count{sql_type!="internal"}[2m])) by (namespace,instance)`                                           |
| tidb        | qps                   | 请求数量         | `sum(rate(tidb_executor_statement_total[1m])) by (namespace,instance,type)`                                                                                                                                                                                           |
| tidb        | slow_query            | 慢查询           | `histogram_quantile(0.90, sum(rate(tidb_server_slow_query_process_duration_seconds_bucket{sql_type="general"}[1m])) by (namespace,instance,le,sql_type))`                                                                                                                |
| tidb        | block_cache_size      | block_cache_size | `avg(tikv_engine_block_cache_size_bytes{db="kv"}) by (namespace,instance,cf)`                                                                                                                                                                                           |
| tidb        | uptime                | 服务存活时长     | `max(time() - process_start_time_seconds{cluster="tidb",component="tidb"}) by (namespace,instance)`                                                                                                                                                                     |
| tidb        | cpu_usage             | CPU 使用率      | `sum(rate(process_cpu_seconds_total{cluster="tidb",component="tidb"}[1m])) by (namespace,instance)`                                                                                                                                                                     |
| tidb        | memory_usage          | 内存使用量       | `avg(process_resident_memory_bytes{cluster="tidb",component="tidb"}) by (namespace,instance)`                                                                                                                                                                          |
| tidb        | statement_avg_queries      | 多语句平均查询数       | `sum(rate(tidb_server_multi_query_num_sum[1m])) / sum(rate(tidb_server_multi_query_num_count[1m]))`                                                                 |
| tidb        | transaction_retry          | 事务重试次数           | `sum(rate(tidb_session_retry_num_sum[1m])) by (namespace,instance)`                                                                                                 |
| tidb        | ddl_job_count              | DDL 作业数             | `sum(tidb_ddl_running_job_count) by (namespace,instance,state)`                                                                                                     |
| tidb        | top_sql_cpu                | TopSQL CPU 消耗        | `sum(rate(tidb_topsql_report_duration_seconds_sum[1m])) by (namespace,instance,sql_type)`                                                                          |
| tidb        | server_is_up               | 服务存活节点数         | `count(up{cluster="tidb",component="tidb"} == 1) by (namespace)`                                                                                                    |
| tikv        | store_size                 | 已用存储容量           | `sum(tikv_store_size_bytes{type="used"}) by (namespace,instance)`                                                                                                   |
| tikv        | available_size             | 可用存储容量           | `sum(tikv_store_size_bytes{type="available"}) by (namespace,instance)`                                                                                              |
| tikv        | capacity_size              | 总存储容量             | `sum(tikv_store_size_bytes{type="capacity"}) by (namespace,instance)`                                                                                               |
| tikv        | cpu_usage                  | CPU 使用率            | `sum(rate(process_cpu_seconds_total{cluster="tidb",component="tikv"}[1m])) by (namespace,instance)`                                                                 |
| tikv        | memory_usage               | 内存使用量             | `avg(process_resident_memory_bytes{cluster="tidb",component="tikv"}) by (namespace,instance)`                                                                       |
| tikv        | io_util                    | IO 利用率              | `rate(node_disk_io_time_seconds_total[1m])`                                                                                                                         |
| tikv        | write_wal_mbps                       | 写入速率 (Mbps)            | `sum(rate(tikv_engine_flow_bytes{type="wal_file_bytes"}[1m])) by (namespace,instance)`                                                                                                       |
| tikv        | read_mbps                            | 读取速率 (Mbps)            | `sum(rate(tikv_engine_flow_bytes{type=~"bytes_read\|iter_bytes_read"}[1m])) by (namespace,instance)`                                                                                         |
| tikv        | qps                                  | QPS                        | `sum(rate(tikv_storage_command_total[1m])) by (namespace,instance,type)`                                                                                                                     |
| tikv        | grpc_qps                             | 各类 gRPC 请求 QPS         | `sum(rate(tikv_grpc_msg_duration_seconds_count{type=~"coprocessor\|kv_get\|kv_batch_get\|kv_prewrite\|kv_commit"}[1m])) by (namespace,instance,type)`                                         |
| tikv        | threadpool_readpool_cpu              | StorageReadPool 线程池 CPU | `sum(rate(tikv_thread_cpu_seconds_total{name=~"unified_read_pool_.*"}[1m])) by (namespace,instance,name)`                                                                                   |
| tikv        | raft_propose_wait                    | RaftPropose 等待延迟 P99   | `histogram_quantile(0.99, sum(rate(tikv_raftstore_request_wait_time_duration_secs_bucket[1m])) by (le,namespace,instance))`                                                                    |
| tikv        | raft_apply_wait                      | RaftApply 等待延迟 P99     | `histogram_quantile(0.99, sum(rate(tikv_raftstore_apply_wait_time_duration_secs_bucket[1m])) by (le,namespace,instance))`                                                                      |
| tikv        | region_pending                       | PendingRegion 数量         | `sum(tikv_raftstore_read_index_pending_duration_count) by (namespace,instance)`                                                                                                              |
| tikv        | rocksdb_write_stall                  | RocksDB 写阻塞次数         | `sum(tikv_engine_write_stall) by (namespace,instance)`                                                                                                                                        |
| tikv        | snapshot_apply_count                 | SnapshotApply 次数         | `sum(rate(tikv_raftstore_apply_duration_secs_count[1m])) by (namespace,instance)`                                                                                                           |
| tikv        | server_is_up                         | 服务存活节点数             | `count(up{cluster="tidb",component="tikv"} == 1) by (namespace)`                                                                                                                              |
| tikv        | cpu_usage                            | CPU 使用率                 | `sum(rate(process_cpu_seconds_total{cluster="tidb",component="tikv"}[1m])) by (namespace,instance)`                                                                                           |
| pd          | storage_capacity           | 集群总容量           | `sum(pd_cluster_status{type="storage_capacity"}) by (namespace)`                                                                                                                                                                     |
| pd          | storage_size               | 已用容量             | `sum(pd_cluster_status{type="storage_size"}) by (namespace)`                                                                                                                                                                         |
| pd          | storage_used_ratio         | 已用容量比           | `sum(pd_cluster_status{type="storage_size"}) / sum(pd_cluster_status{type="storage_capacity"})`                                                                                                                                      |
| pd          | store_up_count             | 健康 Store 数量      | `sum(pd_cluster_status{type="store_up_count"}) by (namespace)`                                                                                                                                                                       |
| pd          | store_down_count           | Down Store 数量      | `sum(pd_cluster_status{type="store_down_count"}) by (namespace)`                                                                                                                                                                     |
| pd          | store_unhealth_count       | Unhealth Store 数量  | `sum(pd_cluster_status{type="store_unhealth_count"}) by (namespace)`                                                                                                                                                                 |
| pd          | store_low_space_count      | 低空间 Store 数量    | `sum(pd_cluster_status{type="store_low_space_count"}) by (namespace)`                                                                                                                                                                |
| pd          | store_slow_count           | 慢 Store 数量        | `sum(pd_cluster_status{type="store_slow_count"}) by (namespace)`                                                                                                                                                                     |
| pd          | abnormal_region_count      | 异常 Region 数量     | `sum(pd_regions_status{type=~"pending-peer-region-count\|down-peer-region-count\|offline-peer-region-count\|miss-peer-region-count\|learner-peer-region-count"}) by (namespace,type)`                                               |
| pd          | region_health              | Region 健康状态      | `max(pd_regions_status) by (namespace,type)`                                                                                                                                                                                         |
| pd          | leader_primary             | PDLeader-Primary 节点 | `service_member_role{cluster="tidb",component="pd",service="PD"}`                                                                                                                                                                     |
| pd          | region_count               | Region 总数          | `sum(pd_cluster_status{type="region_count"}) by (namespace)`                                                                                                                                                                         |
| pd          | leader_count               | Leader 总数          | `sum(pd_cluster_status{type="leader_count"}) by (namespace)`                                                                                                                                                                         |
| pd          | learner_count              | Learner 总数         | `sum(pd_cluster_status{type="learner_count"}) by (namespace)`                                                                                                                                                                        |
| pd          | witness_count              | Witness 总数         | `sum(pd_cluster_status{type="witness_count"}) by (namespace)`                                                                                                                                                                        |
| pd          | cpu_usage                  | CPU 使用率          | `sum(rate(process_cpu_seconds_total{cluster="tidb",component="pd"}[1m])) by (namespace,instance)`                                                                                                                                   |
| pd          | memory_usage               | 内存使用量           | `avg(process_resident_memory_bytes{cluster="tidb",component="pd"}) by (namespace,instance)`                                                                                                                                         |

字段释义（infra 样例数据文件）

| 字段名            | 含义                                                         |
| ----------------- | ------------------------------------------------------------ |
| time              | 记录时间（UTC 格式），示例：`2025-05-05 16:04:00+00:00`            |
| cf                | 保留字段，当前为空，可用于后续扩展或标记                   |
| device            | 若指标涉及网络、文件系统等设备，此处记录设备名称，否则为空                 |
| instance          | 数据采集节点名称，例如 `aiops-k8s-01`、`aiops-k8s-03`        |
| kpi_key           | 指标编码，例如 `pod_cpu_usage`                               |
| kpi_name          | 指标名称，例如 `CPU 使用率`                                  |
| kubernetes_node   | 保留字段，后续可标记 Kubernetes Node 名称       |
| mountpoint        | 若指标涉及文件系统等设备，此处记录文件系统挂载点路径，否则为空            |
| namespace         | Kubernetes 命名空间，例如 `hipstershop`                      |
| object_type       | 对象类型，例如 `pod`、`node`、`tidb`                         |
| pod               | Pod 名称，例如 `emailservice-2`、`productcatalogservice-2`     |
| pod_cpu_usage     | 指标值（根据 kpi_key 变化，例如 CPU 使用率时就是该字段为百分比；若 kpi_key 为其他，则由具体列命名）             |
| sql_type          | 保留字段，后续可标记 SQL 类型                           |
| type              | 保留字段，后续可标记数据行类型                   |


>infra 样例数据（部分）
>| time                  | cf   | device | instance       | kpi_key         | kpi_name  | kubernetes_node | mountpoint | namespace    | object_type | pod                          | pod_cpu_usage | sql_type | type |
>| --------------------- | ---- | ------ | -------------- | ---------------- | --------- | --------------- | ---------- | ------------ | ----------- | ---------------------------- | ------------- | -------- | ---- |
>| 2025-05-05 16:04:00+00:00  | null | null   | aiops-k8s-01   | pod_cpu_usage    | CPU使用率 | null            | null       | hipstershop  | pod         | emailservice-2               | 0.0           | null     | null |
>| 2025-05-05 16:04:00+00:00  | null | null   | aiops-k8s-01   | pod_cpu_usage    | CPU使用率 | null            | null       | hipstershop  | pod         | productcatalogservice-2       | 0.0           | null     | null |
>| 2025-05-05 16:04:00+00:00  | null | null   | aiops-k8s-01   | pod_cpu_usage    | CPU使用率 | null            | null       | hipstershop  | pod         | recommendationservice-1       | 0.0           | null     | null |
>| 2025-05-05 16:04:00+00:00  | null | null   | aiops-k8s-01   | pod_cpu_usage    | CPU使用率 | null            | null       | hipstershop  | pod         | shippingservice-2             | 0.0           | null     | null |
>| 2025-05-05 16:04:00+00:00  | null | null   | aiops-k8s-03   | pod_cpu_usage    | CPU使用率 | null            | null       | hipstershop  | pod         | adservice-2                   | 0.0           | null     | null |

在实际分析中，可根据不同 kpi_key 将 Parquet 文件加载为 DataFrame，结合 time 字段按时间序列绘制曲线图，观察资源使用趋势，并与业务指标跨表关联，评估应用性能与底层资源负载的关系。

### Trace

Trace 数据描述了微服务调用过程中各个 Span（子调用）的详细信息，帮助定位跨服务请求链路中的瓶颈与异常。主要字段与含义如下：

| 字段名             | 含义                                               |
| ------------------ | -------------------------------------------------- |
| traceID            | Trace 唯一标识，用于将同一次请求在不同服务间关联。                                      |
| spanID             | Span 唯一标识，用于表示该调用链中某个具体子调用。                                      |
| flags              | Trace flags 值，通常表示采样与上下文传递信息。                                  |
| operationName      | 操作名称，通常为微服务中某个 gRPC 或 HTTP 接口全路径，如 hipstershop.CartService/GetCart。                                            |
| references         | 引用关系列表（如 CHILD_OF），用于表示该 Span 在调用链中的父子关系（例如当前 Span 是哪个上游 Span 的子调用）。                      |
| startTime          | Span 开始时间，纳秒级时间戳（通常是 epoch 纳秒）。                            |
| startTimeMillis    | Span 开始时间，毫秒级时间戳（对齐人类可读时间）。                            |
| duration           | Span 持续时长，微秒级。     |
| tags               | 标签列表，包含一系列键值对，例如 RPC 系统（rpc.system）、Span 类型（span.kind）、gRPC 状态码（rpc.grpc.status_code）等。                              |
| logs               | 日志列表，表示在 Span 执行过程中记录的事件，如 ServerRecv、ServerSend 等。                      |
| process            | 进程信息，包含 serviceName（当前 Span 所属的服务名称）及一些附加标签（如 hostname）。             |


> 样例数据（Trace 数据）
>| traceID                             | spanID                             | flags | operationName                                            | references                                                                                                    | startTime    | startTimeMillis | duration | tags                                                                                                                                                                                                                                                 | logs                                                                                                                                                                                                                                    | process                                                                                                                         |
>| ----------------------------------- | ---------------------------------- | ----- | -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- | ------------ | --------------- | -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
>| 063346d9fb108c5fd56ecdeb9aae4e97    | a5dbaca343f5bf6b                   | 1.0   | hipstershop.CurrencyService/GetSupportedCurrencies       | [{'refType': 'CHILD_OF', 'spanID': '0473d09282f6f37b'}]                                                       | 1746028800342964 | 1746028800342  | 4202     | [{'key': 'rpc.system', 'type': 'string', 'value': 'grpc'}, {'key': 'span.kind', 'type': 'string', 'value': 'server'}, {'key': 'rpc.grpc.status_code', 'type': 'int64', 'value': 0}]                                                               | [{'fields': [{'key': 'message.type', 'type': 'string', 'value': 'EVENT'}, {'key': 'message.event', 'type': 'string', 'value': 'ServerRecv'}], 'timestamp': 1746028800343000}]                                                           | {'serviceName': 'frontend', 'tags': [{'key': 'hostname', 'type': 'string', 'value': 'frontend-xyz'}]}                           |
>| 44d06fcdceb3be247b1665f7affc4507    | c451558641c213e0                   | 1.0   | hipstershop.CartService/GetCart                           | [{'refType': 'CHILD_OF', 'spanID': '8e66f5b2da1c2e8f'}]                                                       | 1746028800375529 | 1746028800375  | 7034     | [{'key': 'rpc.system', 'type': 'string', 'value': 'grpc'}, {'key': 'span.kind', 'type': 'string', 'value': 'server'}, {'key': 'rpc.grpc.status_code', 'type': 'int64', 'value': 0}]                                                              | [{'fields': [{'key': 'message.type', 'type': 'string', 'value': 'EVENT'}, {'key': 'message.event', 'type': 'string', 'value': 'ServerRecv'}], 'timestamp': 1746028800376000}]                                                           | {'serviceName': 'checkoutservice', 'tags': [{'key': 'hostname', 'type': 'string', 'value': 'checkout-abc'}]}                    |


关联与使用

- 通过 `traceID` 可将同一次请求在多个微服务间的所有 Span 串联起来，绘制调用链图。
- `references` 字段中每个元素指明当前 Span 的父调用（CHILD_OF）。
- `startTimeMillis` 与 `duration` 可用于计算服务端响应时长和调用延迟。
- `tags` 中的指标（如 gRPC 状态码）有助于快速定位错误调用。
- `process` 包含采集时配置的服务名等信息，与 APM 数据中 object_id或日志中的 k8_pod 配合，可实现跨表联动，深入分析调用链根源。
  

### Log

Log 数据由 Filebeat 代理从容器中读取并推送至存储后，供后续文本检索与日志分析。字段说明如下：

| 字段名         | 含义                             |
| -------------- | -------------------------------- |
| k8_namespace   | Kubernetes 命名空间              |
| @timestamp     | 日志时间戳（ISO8601 格式，UTC时区）       |
| agent_name     | Filebeat 采集代理名称            |
| k8_pod         | Pod 名称                         |
| message        | 日志消息内容                     |
| k8_node_name   | Kubernetes Node 名称             |



> 样例数据（Log 数据）
>| k8_namespace | @timestamp               | agent_name                | k8_pod                 | message                                             | k8_node_name |
>| ------------ | ------------------------ | ------------------------- | ---------------------- | --------------------------------------------------- | ------------ |
>| hipstershop  | 2025-05-27T00:00:00Z | filebeat-filebeat-bdkxq   | cartservice-2          | Executed endpoint 'gRPC - /hipstershop.C...          | aiops-k8s-03 |
>| hipstershop  | 2025-05-27T00:00:00Z | filebeat-filebeat-bdkxq   | cartservice-2          | Executed endpoint 'gRPC - /hipstershop.C...          | aiops-k8s-03 |
>| hipstershop  | 2025-05-27T00:00:00Z | filebeat-filebeat-bdkxq   | recommendationservice-0 | {"timestamp": 1748275229.6932063, "severity": ...}   | aiops-k8s-03 |
>| hipstershop  | 2025-05-27T00:00:00Z | filebeat-filebeat-bdkxq   | cartservice-2          | Request finished HTTP/2 POST http://cart...          | aiops-k8s-03 |
>| hipstershop  | 2025-05-27T00:00:00Z | filebeat-filebeat-bdkxq   | frontend-0             | {"http.req.id":"9e697136-031f-40c4-abd5-6bccd5..."}  | aiops-k8s-03 |


## 注意事项

- 时区信息
  
  所有文件名上的时间为 CST 时区，指标文件的`time`字段，日志文件的`@timestamp`字段为 UTC 时区。

- 字段命名不一致需注意对齐

  例如 APM 表中使用 `time`、`object_id`；Infra 表中使用 `time`、`pod`（或 `instance` 代表 Node 名称）。跨表关联时，需要对齐字段名称与含义，便于合并分析。

- 保留字段（`cf`，`device`，`kubernetes_node`，`mountpoint`等）

  当前样例数据中为 null 或空值，部分指标用这些字段标记设备信息、节点名称、挂载点等。

## LICENSE

Unless otherwise agreed by the organizers and the contestant, the contestant shall ensure that it only uses the basic data for non-commercial purposes such as scientific research or classroom teaching, and take full responsibility for the use of conversion basic data, also ensure the organizer and its affiliated party are free from expenses or litigation caused by the any use of basic data.
