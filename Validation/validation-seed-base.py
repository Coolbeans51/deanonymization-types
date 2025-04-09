import networkx as nx
import numpy as np
import time

def load_edgelist(filename):
    """Load an edgelist file into a networkx graph."""
    G = nx.Graph()
    with open(filename, 'r') as f:
        for line in f:
            src, dst = map(int, line.strip().split())
            G.add_edge(src, dst)
    return G

def load_seed_mapping(filename):
    """Load seed mapping from file."""
    mapping = {}
    with open(filename, 'r') as f:
        for line in f:
            node_g1, node_g2 = map(int, line.strip().split())
            mapping[node_g1] = node_g2
    return mapping

def get_node_signature(G, node, mapped_nodes, mapped_nodes_other):
    """
    Compute a signature for a node based on its connections to already-mapped nodes.
    Returns a binary vector where each position corresponds to whether the node 
    is connected to the mapped node at that position.
    """
    # More efficient implementation using direct lookup
    return [1 if G.has_edge(node, n) else 0 for n in mapped_nodes]

def deanonymize_graph(G1, G2, seed_mapping, max_iterations=10, time_limit=60):
    """
    Extend seed mapping to a complete mapping between G1 and G2 graphs.
    
    Parameters:
    - G1, G2: The two graphs to match
    - seed_mapping: Dictionary of known node mappings from G1 to G2
    - max_iterations: Maximum number of matching iterations
    - time_limit: Maximum execution time in seconds
    
    Returns:
    - Complete mapping dictionary
    """
    print(f"Starting de-anonymization with {len(seed_mapping)} seed nodes")
    start_time = time.time()
    
    # Initialize mappings
    mapping = seed_mapping.copy()
    reverse_mapping = {v: k for k, v in mapping.items()}
    
    # Get unmapped nodes
    unmapped_g1 = list(set(G1.nodes()) - set(mapping.keys()))
    unmapped_g2 = list(set(G2.nodes()) - set(mapping.values()))
    
    print(f"Nodes in G1: {len(G1.nodes())}, Nodes in G2: {len(G2.nodes())}")
    print(f"Unmapped in G1: {len(unmapped_g1)}, Unmapped in G2: {len(unmapped_g2)}")
    
    # Track iterations and mapped nodes to check convergence
    iteration = 0
    last_mapping_size = len(mapping)
    
    # Continue until convergence or max iterations reached
    while (unmapped_g1 and unmapped_g2 and 
           iteration < max_iterations and 
           time.time() - start_time < time_limit):
        
        iteration += 1
        print(f"\nIteration {iteration}:")
        
        # Get current mapped nodes (list to maintain consistent order)
        mapped_g1_nodes = list(mapping.keys())

        # Calculate batch size - process at most 1000 nodes at a time to avoid memory issues
        batch_size = min(1000, len(unmapped_g1), len(unmapped_g2))
        
        # Take a batch of unmapped nodes
        batch_g1 = unmapped_g1[:batch_size]
        batch_g2 = unmapped_g2[:batch_size]
        
        # Create node signatures for G1 batch
        signatures_g1 = {}
        for node_g1 in batch_g1:
            signatures_g1[node_g1] = get_node_signature(G1, node_g1, mapped_g1_nodes, None)
        
        # Create node signatures for G2 batch
        signatures_g2 = {}
        for node_g2 in batch_g2:
            # Map through the reverse mapping to get G1 equivalent positions
            mapped_g2_nodes = [mapping[n] for n in mapped_g1_nodes]
            signatures_g2[node_g2] = get_node_signature(G2, node_g2, mapped_g2_nodes, None)
        
        # Find the best matches using similarity scoring
        new_matches = 0
        
        # Score threshold - can be adjusted
        threshold = 0.7 if iteration <= 2 else 0.5
        
        # Match nodes with highest similarity
        for node_g1 in batch_g1:
            sig1 = signatures_g1[node_g1]
            
            # Skip if signature is empty (no connections to mapped nodes)
            if not any(sig1):
                continue
                
            best_match = None
            best_score = threshold  # Only accept matches above threshold
            
            for node_g2 in batch_g2:
                # Skip already matched nodes
                if node_g2 in reverse_mapping:
                    continue
                    
                sig2 = signatures_g2[node_g2]
                
                # Calculate similarity (Jaccard similarity)
                matches = sum(1 for a, b in zip(sig1, sig2) if a == b and (a == 1 or b == 1))
                total = sum(1 for a, b in zip(sig1, sig2) if a == 1 or b == 1)
                
                score = matches / total if total > 0 else 0
                
                if score > best_score:
                    best_score = score
                    best_match = node_g2
            
                # Add the best match if found
                if best_match is not None:
                    mapping[node_g1] = best_match
                    revMatches = sum(1 for a, b in zip(sig2, sig1) if a == b and (a == 1 or b == 1))
                    revTotal = sum(1 for a, b in zip(sig2, sig1) if a == 1 or b == 1)
                    revScore = revMatches / revTotal if revTotal > 0 else 0
                    if revScore > threshold:
                        reverse_mapping[best_match] = node_g1
                    new_matches += 1
                    
                    # Remove matched nodes from unmapped lists
                    if node_g1 in unmapped_g1:
                        unmapped_g1.remove(node_g1)
                    if best_match in unmapped_g2:
                        unmapped_g2.remove(best_match)
        
        print(f"  Added {new_matches} new mappings")
        print(f"  Total mappings: {len(mapping)}")
        print(f"  Remaining unmapped in G1: {len(unmapped_g1)}")
        print(f"  Remaining unmapped in G2: {len(unmapped_g2)}")
        
        # Check if we're making progress
        if len(mapping) == last_mapping_size:
            print("No new mappings found, stopping iterations")
            break
            
        last_mapping_size = len(mapping)
    
    # Handle remaining unmapped nodes if needed
    if unmapped_g1 and unmapped_g2:
        print("\nMatching remaining nodes based on degree...")
        
        # Sort remaining nodes by degree
        degrees_g1 = {node: G1.degree(node) for node in unmapped_g1}
        degrees_g2 = {node: G2.degree(node) for node in unmapped_g2}
        
        sorted_g1 = sorted(degrees_g1.items(), key=lambda x: x[1])
        sorted_g2 = sorted(degrees_g2.items(), key=lambda x: x[1])
        
        # Match remaining nodes based on degree similarity
        for (node_g1, _), (node_g2, _) in zip(sorted_g1, sorted_g2):
            mapping[node_g1] = node_g2
    
    print(f"\nDe-anonymization complete in {time.time() - start_time:.2f} seconds")
    print(f"Final mapping size: {len(mapping)} nodes")
    
    return mapping

def save_mapping(mapping, filename):
    """Save the mapping to a file."""
    with open(filename, 'w') as f:
        for node_g1, node_g2 in sorted(mapping.items()):
            f.write(f"{node_g1} {node_g2}\n")

def main():
    # Load the graph edgelists
    print("Loading graphs...")
    G1 = load_edgelist("validation_G1.edgelist")
    G2 = load_edgelist("validation_G2.edgelist")
    
    print(f"G1: {len(G1.nodes())} nodes, {len(G1.edges())} edges")
    print(f"G2: {len(G2.nodes())} nodes, {len(G2.edges())} edges")
    
    # Load the seed mapping
    seed_mapping = load_seed_mapping("other_seed_mapping.txt")
    print(f"Loaded {len(seed_mapping)} seed mappings")
    
    # Perform de-anonymization with time limit
    complete_mapping = deanonymize_graph(G1, G2, seed_mapping, max_iterations=5, time_limit=300)
    
    # Save the complete mapping
    output_file = "complete_mapping.txt"
    save_mapping(complete_mapping, output_file)
    print(f"Complete mapping saved to {output_file}")

if __name__ == "__main__":
    main()