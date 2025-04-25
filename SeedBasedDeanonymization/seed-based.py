import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment
import os
import glob
import time

def load_graph(filename):
    """Load graph from edgelist file."""
    return nx.read_edgelist(filename, nodetype=int)

def load_seeds(seed_file):
    """
    Load seed mappings from file 'seed_mapping.txt'.
    Each line: node_index_in_G1 node_index_in_G2
    Converts all node IDs to integers.
    """
    seeds = {}
    with open(seed_file, 'r') as f:
        for line in f:
            g1_node, g2_node = line.strip().split()
            seeds[int(g1_node)] = int(g2_node)
    return seeds

def extract_node_features(graph, max_nodes=None):
    """
    Extract structural features for each node in the graph.
    Uses simpler/faster metrics for large graphs.
    """
    features = {}
    is_large_graph = len(graph.nodes()) > 1000 if max_nodes is None else len(graph.nodes()) > max_nodes

    print("Computing degree...")
    degree = dict(graph.degree())
    degree_cent = {n: d / (len(graph) - 1) for n, d in degree.items()}

    if is_large_graph:
        print("Large graph detected. Using simplified metrics...")
        print("Computing clustering...")
        clustering = nx.clustering(graph)

        for node in graph.nodes():
            features[node] = [
                degree_cent[node],
                clustering.get(node, 0),
                degree[node],
                len(list(graph.neighbors(node)))
            ]
    else:
        print("Computing closeness centrality...")
        close_cent = nx.closeness_centrality(graph)

        print("Computing clustering...")
        clustering = nx.clustering(graph)

        print("Computing approximate betweenness centrality...")
        if len(graph) > 500:
            between_cent = nx.betweenness_centrality(graph, k=min(500, len(graph)//2))
        else:
            between_cent = nx.betweenness_centrality(graph)

        print("Computing PageRank...")
        pagerank = nx.pagerank(graph, max_iter=100, tol=1e-4)

        for node in graph.nodes():
            features[node] = [
                degree_cent[node],
                close_cent[node],
                between_cent[node],
                clustering.get(node, 0),
                pagerank.get(node, 0),
                degree[node]
            ]

    return features

def similarity_matrix(g1_features, g2_features):
    """Compute similarity matrix between nodes in g1 and g2."""
    print("Building similarity matrix...")
    g1_nodes = list(g1_features.keys())
    g2_nodes = list(g2_features.keys())

    n1 = len(g1_nodes)
    n2 = len(g2_nodes)

    batch_size = 1000
    use_batching = n1 * n2 > 1000000

    if use_batching:
        print(f"Using batch processing for large similarity matrix ({n1}x{n2})...")
        sim_matrix = np.zeros((n1, n2))

        for i_start in range(0, n1, batch_size):
            i_end = min(i_start + batch_size, n1)
            batch_time = time.time()

            for i in range(i_start, i_end):
                node1 = g1_nodes[i]
                vec1 = np.array(g1_features[node1])
                norm1 = np.linalg.norm(vec1)

                if norm1 == 0:
                    continue

                for j, node2 in enumerate(g2_nodes):
                    vec2 = np.array(g2_features[node2])
                    norm2 = np.linalg.norm(vec2)

                    if norm2 == 0:
                        continue

                    sim = np.dot(vec1, vec2) / (norm1 * norm2)
                    sim_matrix[i, j] = sim

            print(f"Processed batch {i_start//batch_size + 1}/{(n1-1)//batch_size + 1} " 
                  f"in {time.time() - batch_time:.2f}s")
    else:
        g1_matrix = np.array([g1_features[node] for node in g1_nodes])
        g2_matrix = np.array([g2_features[node] for node in g2_nodes])

        g1_norms = np.linalg.norm(g1_matrix, axis=1, keepdims=True)
        g2_norms = np.linalg.norm(g2_matrix, axis=1, keepdims=True)

        g1_norms[g1_norms == 0] = 1e-10
        g2_norms[g2_norms == 0] = 1e-10

        g1_normalized = g1_matrix / g1_norms
        g2_normalized = g2_matrix / g2_norms

        sim_matrix = np.dot(g1_normalized, g2_normalized.T)

    return sim_matrix, g1_nodes, g2_nodes

def match_nodes(g1_features, g2_features, seeds=None):
    """Match nodes between two graphs based on features and optional seeds."""
    sim_matrix, g1_nodes, g2_nodes = similarity_matrix(g1_features, g2_features)
    cost_matrix = 1 - sim_matrix

    g1_index = {node: i for i, node in enumerate(g1_nodes)}
    g2_index = {node: j for j, node in enumerate(g2_nodes)}

    matched_g1 = set()
    matched_g2 = set()
    mapping = {}

    if seeds:
        print(f"Applying {len(seeds)} seed constraints...")
        for g1_node, g2_node in seeds.items():
            if g1_node in g1_index and g2_node in g2_index:
                i = g1_index[g1_node]
                j = g2_index[g2_node]
                mapping[g1_node] = g2_node
                cost_matrix[i, :] = np.inf
                cost_matrix[:, j] = np.inf
                cost_matrix[i, j] = 0
                matched_g1.add(i)
                matched_g2.add(j)

    print("Running Hungarian algorithm for remaining matches...")
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    for i, j in zip(row_ind, col_ind):
        if i in matched_g1 or j in matched_g2:
            continue
        mapping[g1_nodes[i]] = g2_nodes[j]

    return mapping

def save_mapping(mapping, output_file):
    """Save node mapping to file in sorted order by source node."""
    with open(output_file, 'w') as f:
        for node in sorted(mapping, key=lambda x: int(x)):
            f.write(f"{node} {mapping[node]}\n")

def find_graph_files():
    """Find edgelist files in the current directory."""
    files = sorted(glob.glob("seed_G*.edgelist"))

    if len(files) < 2:
        raise FileNotFoundError("Could not find at least two seed_G*.edgelist files in the current directory")

    return files[0], files[1]

def main():
    start_time = time.time()

    try:
        g1_file, g2_file = find_graph_files()
        print(f"Found graph files: {g1_file} and {g2_file}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    seed_file = "seed_mapping.txt"
    seeds = None
    if os.path.exists(seed_file):
        print(f"Found seed file: {seed_file}")
        seeds = load_seeds(seed_file)

    output_file = "complete_mapping.txt"

    print("Loading graphs...")
    load_time = time.time()
    G1 = load_graph(g1_file)
    G2 = load_graph(g2_file)
    print(f"Graphs loaded in {time.time() - load_time:.2f}s")

    print(f"Graph 1: {len(G1.nodes())} nodes, {len(G1.edges())} edges")
    print(f"Graph 2: {len(G2.nodes())} nodes, {len(G2.edges())} edges")

    threshold = 10000

    print("Extracting node features...")
    feature_time = time.time()
    g1_features = extract_node_features(G1, max_nodes=threshold)
    g2_features = extract_node_features(G2, max_nodes=threshold)
    print(f"Features extracted in {time.time() - feature_time:.2f}s")

    print("Matching nodes...")
    match_time = time.time()
    mapping = match_nodes(g1_features, g2_features, seeds=seeds)
    print(f"Nodes matched in {time.time() - match_time:.2f}s")

    print("Saving mapping...")
    save_mapping(mapping, output_file)

    print(f"De-anonymization complete in {time.time() - start_time:.2f}s")
    print(f"Mapping saved to {output_file}")

if __name__ == "__main__":
    main()
