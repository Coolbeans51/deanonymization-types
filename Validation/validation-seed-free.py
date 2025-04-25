import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment
import os
import glob
import time

def load_graph(filename):
    """Load graph from edgelist file."""
    return nx.read_edgelist(filename, nodetype=str)

def extract_node_features(graph, max_nodes=None):
    """
    Extract structural features for each node in the graph.
    Uses simpler/faster metrics for large graphs.
    """
    features = {}
    
    # For very large graphs, limit the computation or use simpler metrics
    is_large_graph = len(graph.nodes()) > 1000 if max_nodes is None else len(graph.nodes()) > max_nodes
    
    print("Computing degree...")
    degree = dict(graph.degree())
    degree_cent = {n: d / (len(graph) - 1) for n, d in degree.items()}
    
    if is_large_graph:
        print("Large graph detected. Using simplified metrics...")
        
        # For large graphs, use only degree and local clustering
        print("Computing clustering...")
        clustering = nx.clustering(graph)
        
        # Skip expensive centrality calculations for large graphs
        for node in graph.nodes():
            features[node] = [
                degree_cent[node],                # Degree centrality
                clustering.get(node, 0),          # Clustering coefficient
                degree[node],                     # Raw degree
                len(list(graph.neighbors(node)))  # Number of neighbors (equivalent to degree for simple graphs)
            ]
    else:
        print("Computing closeness centrality...")
        close_cent = nx.closeness_centrality(graph)
        
        print("Computing clustering...")
        clustering = nx.clustering(graph)
        
        # Compute efficient approximation of betweenness using sampling
        print("Computing approximate betweenness centrality...")
        if len(graph) > 500:
            # For medium-sized graphs, use sampling
            between_cent = nx.betweenness_centrality(graph, k=min(500, len(graph)//2))
        else:
            between_cent = nx.betweenness_centrality(graph)
        
        print("Computing PageRank...")
        pagerank = nx.pagerank(graph, max_iter=100, tol=1e-4)
        
        # Combine features
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
    
    # For larger graphs, use batch processing to avoid memory issues
    batch_size = 1000
    use_batching = n1 * n2 > 1000000  # 10M threshold
    
    if use_batching:
        print(f"Using batch processing for large similarity matrix ({n1}x{n2})...")
        sim_matrix = np.zeros((n1, n2))
        
        # Process in batches to avoid memory issues
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
                        
                    # Cosine similarity
                    sim = np.dot(vec1, vec2) / (norm1 * norm2)
                    sim_matrix[i, j] = sim
            
            print(f"Processed batch {i_start//batch_size + 1}/{(n1-1)//batch_size + 1} " 
                  f"in {time.time() - batch_time:.2f}s")
    else:
        # For smaller matrices, vectorize the computation
        # Convert feature dictionaries to matrices for faster computation
        g1_matrix = np.array([g1_features[node] for node in g1_nodes])
        g2_matrix = np.array([g2_features[node] for node in g2_nodes])
        
        # Normalize feature vectors for cosine similarity
        g1_norms = np.linalg.norm(g1_matrix, axis=1, keepdims=True)
        g2_norms = np.linalg.norm(g2_matrix, axis=1, keepdims=True)
        
        # Replace zero norms with a small value to avoid division by zero
        g1_norms[g1_norms == 0] = 1e-10
        g2_norms[g2_norms == 0] = 1e-10
        
        g1_normalized = g1_matrix / g1_norms
        g2_normalized = g2_matrix / g2_norms
        
        # Compute similarity matrix
        sim_matrix = np.dot(g1_normalized, g2_normalized.T)
    
    return sim_matrix, g1_nodes, g2_nodes

def match_nodes(g1_features, g2_features):
    """Match nodes between two graphs based on features."""
    # Convert similarity matrix to cost matrix for linear assignment
    sim_matrix, g1_nodes, g2_nodes = similarity_matrix(g1_features, g2_features)
    
    print("Converting to cost matrix...")
    cost_matrix = 1 - sim_matrix  # Convert similarity to cost
    
    # Use Hungarian algorithm to find optimal assignment
    print("Running Hungarian algorithm for optimal assignment...")
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Create mapping
    mapping = {}
    for i, j in zip(row_ind, col_ind):
        mapping[g1_nodes[i]] = g2_nodes[j]
    
    return mapping

def save_mapping(mapping, output_file):
    """Save node mapping to file."""
    with open(output_file, 'w') as f:
        for node1, node2 in mapping.items():
            f.write(f"{node1} {node2}\n")

def find_graph_files():
    """Find edgelist files in the current directory."""
    # Look for files with pattern validation_G*.edgelist
    files = sorted(glob.glob("validation_G*.edgelist"))
    
    if len(files) < 2:
        raise FileNotFoundError("Could not find at least two validation_G*.edgelist files in the current directory")
    
    return files[0], files[1]  # Return the first two matching files

def main():
    start_time = time.time()
    
    # Find graph files in the current directory
    try:
        g1_file, g2_file = find_graph_files()
        print(f"Found graph files: {g1_file} and {g2_file}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Output file
    output_file = "node_mapping.txt"
    
    # Load graphs
    print("Loading graphs...")
    load_time = time.time()
    G1 = load_graph(g1_file)
    G2 = load_graph(g2_file)
    print(f"Graphs loaded in {time.time() - load_time:.2f}s")
    
    print(f"Graph 1: {len(G1.nodes())} nodes, {len(G1.edges())} edges")
    print(f"Graph 2: {len(G2.nodes())} nodes, {len(G2.edges())} edges")
    
    # Determine max_nodes threshold based on graph sizes
    max_nodes = max(len(G1.nodes()), len(G2.nodes()))
    threshold = 10000  # Threshold for simplified metrics
    
    # Extract node features
    print("Extracting node features...")
    feature_time = time.time()
    g1_features = extract_node_features(G1, max_nodes=threshold)
    g2_features = extract_node_features(G2, max_nodes=threshold)
    print(f"Features extracted in {time.time() - feature_time:.2f}s")
    
    # Match nodes
    print("Matching nodes...")
    match_time = time.time()
    mapping = match_nodes(g1_features, g2_features)
    print(f"Nodes matched in {time.time() - match_time:.2f}s")
    
    # Save mapping
    print("Saving mapping...")
    save_mapping(mapping, output_file)
    
    print(f"De-anonymization complete in {time.time() - start_time:.2f}s")
    print(f"Mapping saved to {output_file}")

if __name__ == "__main__":
    main()