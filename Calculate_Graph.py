import networkx as nx
from collections import Counter

cls = "mix"

# 그래프 파일 경로
graph_path = f"/workspace/PathCoRAG/multihoprag/graph_atomic_entity_relation.graphml"  # 실제 경로로 수정하세요

# 그래프 불러오기
G = nx.read_graphml(graph_path)

# 전체 노드 수
num_nodes = G.number_of_nodes()

# 전체 엣지 수
num_edges = G.number_of_edges()

# 연결된 컴포넌트 수 (subgraph 수)
num_subgraphs = nx.number_connected_components(G.to_undirected())

# 평균 노드 degree
avg_degree = sum(dict(G.degree()).values()) / num_nodes

# 고립된 노드 (degree가 0인 노드)
isolated_nodes = list(nx.isolates(G))
num_isolated_nodes = len(isolated_nodes)
isolated_ratio = num_isolated_nodes / num_nodes


# 결과 출력
print("Graph Statistics")
print(f"Total Nodes: {num_nodes}")
print(f"Total Edges: {num_edges}")
print(f"Number of Connected Subgraphs: {num_subgraphs}")
print(f"Average Node Degree: {avg_degree:.4f}")
print(f"Isolated Nodes: {num_isolated_nodes}")
print(f"Isolated Node Ratio: {isolated_ratio:.2%}")

connected_components = list(nx.connected_components(G.to_undirected()))
component_sizes = [len(c) for c in connected_components]
size_counts = Counter(component_sizes)

print("\nSubgraph Size Distribution:")
for size in sorted(size_counts):
    print(f" - Subgraph with {size} nodes: {size_counts[size]}개")