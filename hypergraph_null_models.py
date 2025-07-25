import numpy as np
import networkx as nx
import xgi
import random
from math import comb
from itertools import combinations
from collections import defaultdict


                              # --------- PROJECTION-PRESERVING NULL MODELS --------- #


def flag_hypergraph(Gr, ns, seed=None):
    
    """
    Generate a flag (or clique) hypergraph from a
    NetworkX graph by replacing cliques with hyperedges.

    Parameters
    ----------
    Gr : Networkx Graph.

    ns: can be either list of int where the s-th entry is the number of (s+3)-cliques 
        promoted to hyperedges; or list of floats, where each entry is the fraction of
        total cliques to be promoted at that order.

    Returns
    -------
    H : the clique-promoted Hypergraph in xgi format.

    """
    G = Gr.copy()
    
    if seed is not None:
        np.random.seed(seed)

    cliques = []
    for clq in nx.enumerate_all_cliques(G):  # sorted by size
        if len(clq) <= 2: continue           # don't add singletons nor pairs
        elif len(clq) <= len(ns)+2:
            cliques.append(clq)
        else: break

    # store cliques per order
    cliques_s = defaultdict(list)
    for x in cliques:
        cliques_s[len(x)].append(x)
    
    if isinstance(ns[0], float):
        for i in range(len(ns)):
            ns[i] = int(ns[i]*len(cliques_s[i+3]))
            
    hedges_to_add = []       # list of cliques to promote
    edges_to_remove = set()  # when promoting a clique, remove the original (pairwise) edges            

    for i, nc in enumerate(ns):        
        if nc==0: continue
        s = i + 3  # clique size
        # select cliques to promote
        idxs = np.random.choice(len(cliques_s[s]), nc, replace=False)
        to_add = [cliques_s[s][idx] for idx in idxs]
        hedges_to_add += to_add
        edges_to_remove |= set([e for j in to_add for e in combinations(j, 2)])
            
    nodes = G.nodes()
    G.remove_edges_from(list(edges_to_remove))
    edges_G = G.edges()
    H = xgi.Hypergraph()
    H.add_nodes_from(nodes)
    H.add_edges_from(edges_G)
    H.add_edges_from(hedges_to_add)
    
    return H




def project_hedges (H_orig, f, seed=None):
    
    """""""""
    Returns a copy of the given hypergraph, where a fraction f the
    hyperedges is projected to pairwise links.
        
    Parameters:
    -----------
    H_orig : the original hypergraph.
    
    f : the fraction of hyperedges to be down-projected.
    
    seed (optional) : the seed for the random selection of hyperedges to be projected.
    
    """""""""
    
    if seed is not None:
        random.seed(seed)
        
    H = H_orig.copy()
        
    hedge_dic = H.edges.members(dtype=dict)    
    hedges = [i for i in hedge_dic.values() if len(i)>2]    

    num = round(f*len(hedges))
    to_project = random.sample(hedges, num)
    to_add = set([i for j in to_project for i in combinations(j, 2)])
    id_to_remove = {i for i in hedge_dic if hedge_dic[i] in to_project}
                
    H.remove_edges_from(id_to_remove)
    H.add_edges_from(list(to_add))
    # avoid multiedges 
    H.cleanup(isolates=True, singletons=True, connected=False, relabel=False)
                 
    return H       



                                   # ---------RESHUFFLING METHODS--------- #


def edge_shuffled_hypergraph(H, degree_prop=False, seed=None):
    
    """""""""
    Returns a reshuffled version of the hypergraph, keeping only the number of nodes,
    the size of each hyper-edge and the number of hyper-edges.
    
    Parameters:
    ------------
    
    H : the hypergraph to reshuffle. Can be either an instance of  or of NetworkX.Graph.
    
    degree_prop (bool) : if True, the probability of choosing one node when creating a hyper-edge
                         is proportional to its hyper-degree in the original hypergraph. This 
                         reproduce (on average) the hyper-degree of the original system 
                         in the reshuffled hypergraph.
                         
    seed (int) : The seed for the random number generator (default None).
               
    ------------
    
    Returns : the reshuffled hypergraph (graph), in the same format of the given input (xgi/NetworkX).
    
    ------------
    
    Note: the reshuffled hypergraph might not be connected. To manage this cases, 
    see the H.cleanup() function of xgi.
    
    """""""""
    if seed is not None:
        np.random.seed(seed)
        
    nodes = list(H.nodes)
        
    if isinstance(H, nx.Graph):
        edges = list(H.edges)
        H1 = xgi.Hypergraph()
        H1.add_nodes_from(nodes)
        H1.add_edges_from(edges)
    else:
        H1 = H.copy()
        
    e_sizes = [len(e) for e in H1.edges.members()]
    interactions = []
    i = 0            
    p_deg = None
    if degree_prop: 
        degs = [H1.degree(n) for n in nodes]
        p_deg = list(np.array(degs)/sum(degs))
       
    while i < len(e_sizes):
        e = list(np.random.choice(nodes, size=e_sizes[i], replace=False, p=p_deg))
        e.sort()
        # avoid double edges
        if e not in interactions:
            interactions.append(e)
            i+=1
            
    H_r = xgi.Hypergraph()
    H_r.add_nodes_from(nodes)
    H_r.add_edges_from(interactions)
    
    if isinstance(H, nx.Graph):
        H_r = xgi.to_graph(H_r)
    
    return(H_r)



def dp_edge_shuffled_hypergraph(H, seed=None):
    """""""""
    Auxiliary functions equal to edge_shuffled_hypergraph(H, degree_prop=True, seed=seed)
    (see edge_shuffled_hypergraph docstring).
    """""""""
    return edge_shuffled_hypergraph(H, degree_prop=True, seed=seed)



def configuration_model_hypergraph (H, n_shuff=1e4, n_min_he=4, f_max_he=0.9, seed=None):
    
    """""""""""
    Generates a reshuffled hypergraph while keeping: the number of hyper-edges at
    each order; the degree of every node at each order. 
    
    Parameters:
    ---------------
    H (xgi.Hypergraph) : the hypergraph to reshuffle.
              
    n_shuff (int) : number of succesful reshuffling iterations (default 10000) to
                    perform at each size of hyperedges.
                         
    n_min_he (int) : minimum number of hyperedges necessary to perform the reshuffling.
                     If there are less than n_min_e hyperedges of size s, all these
                     s-hyperedges are simply copied to the returned hypergraph without
                     reshuffling (default=4).
              
    f_max_he (int): maximum fraction of possible s-hyperedges over which the reshuffling
                    is not performed at size s (default=0.9). Example (for size s=2): if
                    there are more than max_he*N(N-1)/2 edges of size 2, reshuffling is
                    not performed at s=2 and the 2-edges are just copied from the old to
                    the new hypergraph.
              
    seed (int) : seed for random numbers generator (default None).

    ---------------
    Note: This implementation does not assure the connectivity of the resulting hypergraph:
          an originally connected H may be reshuffled in a disconnected H_{null}.
    
    """""""""""    
    
    if seed is not None:
        np.random.seed(seed)

    EdgesBySize = defaultdict(list)        
    for e in H.edges.members():
        m = len(e)
        EdgesBySize[m].append(list(e))

    for s in EdgesBySize.keys():
        
        # skip size if there are too few or too many h-edges
        Es = len(EdgesBySize[s])
        Es_max = comb(len(H.nodes), s)
        if Es<n_min_he or Es>f_max_he*Es_max:
            continue
            
        itr = 0
        while itr<n_shuff:
            # select pair of h-edges
            id1, id2 = np.random.choice(len(EdgesBySize[s]), 2, replace=False)
            e1 = EdgesBySize[s][id1]
            e2 = EdgesBySize[s][id2]
            # select one node per h-edge
            n1 = np.random.choice(e1)
            n2 = np.random.choice(e2)
            # skip iteration if one node is part of both edges
            if n1 in e2 or n2 in e1:
                continue
            # swap nodes
            e1new = e1.copy()
            e2new = e2.copy()
            e1new.remove(n1)
            e2new.remove(n2)
            e1new.append(n2)
            e2new.append(n1)
            e1new.sort()
            e2new.sort()
            
            # check that we do not create double h-edges 
            if e1new not in EdgesBySize[s] and e2new not in EdgesBySize[s]:
                EdgesBySize[s].remove(e1)
                EdgesBySize[s].remove(e2)
                EdgesBySize[s].append(e1new)
                EdgesBySize[s].append(e2new)
                itr+=1

    hedges = [e for s_he in EdgesBySize.values() for e in s_he]
    Hn = xgi.Hypergraph(hedges)
    
    return Hn
