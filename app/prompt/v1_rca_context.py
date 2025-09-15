import textwrap

CONTEXT_BLOCK = textwrap.dedent(
    """
**5. 服务依赖图 (MANDATORY REFERENCE):**
- **frontend**: -> `checkoutservice`, `recommendationservice`, `productcatalogservice`, `shippingservice`, `adservice`, `cartservice`, `currencyservice`.
- **checkoutservice**: -> `shippingservice`, `paymentservice`, `currencyservice`, `emailservice`, `cartservice`.
- **recommendationservice**: -> `productcatalogservice`.
- **adservice**: -> `tidb` (具体组件: `tidb-tidb`, `tidb-pd`, `tidb-tikv`).
- **productcatalogservice**: -> `tidb` (具体组件: `tidb-tidb`, `tidb-pd`, `tidb-tikv`).
- **cartservice**: -> `redis-cart`.
- **shippingservice**: (无下游依赖)
- **currencyservice**: (无下游依赖)
- **paymentservice**: (无下游依赖)
- **emailservice**: (无下游依赖)

**6. 系统部署拓扑:**
- **虚拟机 (Nodes)**: 共 8 台(aiops-k8s-01到aiops-k8s-08)。
- **核心微服务**: 共 10 个，每个服务部署 3 个 Pod 副本 (e.g., adservice-0, adservice-1, adservice-2，特别注意redis-cart只有1个pod为redis-cart-0)。
- **数据库/缓存 (Databases/Caches)**: `redis-cart`, `tidb-tidb`, `tidb-pd`, `tidb-tikv`.
- **Pod-Node 关系**: Pod **动态调度**到 Node。必须通过工具返回的 `related_node` 字段来确定其物理位置。

"""
)
# V12: 新增结构化的拓扑信息，以支持服务级联故障分析
TOPOLOGY = {
    "services": {
        # 10个核心微服务，每个3个副本
        "frontend": {"pods": ["frontend-0", "frontend-1", "frontend-2"]},
        "checkoutservice": {"pods": ["checkoutservice-0", "checkoutservice-1", "checkoutservice-2"]},
        "recommendationservice": {"pods": ["recommendationservice-0", "recommendationservice-1", "recommendationservice-2"]},
        "productcatalogservice": {"pods": ["productcatalogservice-0", "productcatalogservice-1", "productcatalogservice-2"]},
        "shippingservice": {"pods": ["shippingservice-0", "shippingservice-1", "shippingservice-2"]},
        "adservice": {"pods": ["adservice-0", "adservice-1", "adservice-2"]},
        "cartservice": {"pods": ["cartservice-0", "cartservice-1", "cartservice-2"]},
        "currencyservice": {"pods": ["currencyservice-0", "currencyservice-1", "currencyservice-2"]},
        "paymentservice": {"pods": ["paymentservice-0", "paymentservice-1", "paymentservice-2"]},
        "emailservice": {"pods": ["emailservice-0", "emailservice-1", "emailservice-2"]},
        
        # 节点Node
        "aiops-k8s-01"
        "aiops-k8s-02"
        "aiops-k8s-03"
        "aiops-k8s-04"
        "aiops-k8s-05"
        "aiops-k8s-06"
        "aiops-k8s-07"
        "aiops-k8s-08"
        
        # 特例：缓存/数据库服务
        "redis-cart": {"pods": ["redis-cart-0"]},
        "tidb": {"pods": ["tidb-tidb", "tidb-pd", "tidb-tikv"]}
    }
} 