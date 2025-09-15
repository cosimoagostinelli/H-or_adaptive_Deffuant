import numpy as np
import networkx as nx
import xgi


def stratified_er_hypergraph (N, ps, p_type='prob', seed=None):
    
    """""""""""
    Generates a hypergraph by overlapping multiple layers. Each layer is a higher-order 
    Erdos-Renyi model for a specific hyper-edge size. 
    
    Parameters:
    ---------------
    N (int) : Number of nodes.
   
    ps (list) : Each entry p[s] si the probability of an s-hyperedge if p_type=”prob” 
                and mean expected degree if p_type=”degree”.
                
    p_type (str, optional) : Determines the way ps is interpreted (see ps for detail). 
                             Valid options are “prob” or “degree”, by default “prob”.
  
    seed : seed for random numbers generator.
    ---------------
    
    Notes: len(ps) = m-1, where m is the maximum hyperedge size.
    """""""""""
    
    nodes = range(N)
    H = xgi.Hypergraph()
    H.add_nodes_from(nodes)
    m_max = len(ps)    
    
    for s in range(m_max):
        if ps[s]>0:
            H_s = xgi.uniform_erdos_renyi_hypergraph(N, s+2, ps[s], p_type=p_type, seed=seed)
            edges = H_s.edges.members()
            H.add_edges_from(edges)
            
    return(H)



def powerlaw_degree_distribution(N, gamma, seed, cutoff=np.inf):
    """
    Draw N samples from a power-law distribution of the form
    p(k) = Ak^{-gamma} , in the domain [1,inf]. gamma is 
    specified by the user and A=gamma-1 is a normalization constant.
    the samples are rounded to the nearest integer.

    Parameters:
    --------------
    N (int) : number of samples.

    gamma (float) : exponent of the degree distribution.

    seed (int) : seed for the random number generator.

    cutoff (int) : maximum admissible value for the samples
                   (default none). If given, the returned samples
                   will be a list of min(k,cutoff).
    -------------
    Returns:
        ks (list of int) : list of degrees sorted by increasing size.
    
    """
    np.random.seed(seed)
    xs = np.random.random(N)
    ks = []
    for x in xs:
        k = ( 1. - x ) ** ( 1. / (1.-gamma) )
        k_ = np.min( [k,cutoff] )
        if k_ <1:
            k_=1 
        ks.append( int(np.round(k_)) )
        
    return sorted(ks)
    


def stratified_cm_hypergraph (order_deg_list, seed=None):
    
    """""""""""
    Generates a hypergraph where each node has the given degree at the specified order. 
    
    Parameters:
    ---------------
    order_deg_list : contains the nodes degrees at each order, in form of list of lists.
                     order_deg_list[x] is the list of degrees when looking only at 
                     hyperedges of size x+2; 
                     
    seed: seed for random numbers generator.
    ---------------
    Notes: each list contained in order_deg_list must have the same number of elements
          (that is the number of nodes). The maximum hyperedge size will be equal to 
          len(order_deg_list)+1.
          If sum(order_deg_list[i]) is not divisible by i+2 (for every i) the algorithm 
          still runs, but raises a warning and adds an additional connection to random 
          nodes to satisfy this condition (see xgi.uniform_hypergraph_configuration_model).
    
    """""""""""
    
    H = xgi.Hypergraph()
    nodes = range(len(order_deg_list[0]))
    H.add_nodes_from(nodes)
    m_max = len(order_deg_list)
    
    for i in range(m_max):
        k = { n: deg for n,deg in enumerate(order_deg_list[i]) }
        H_i = xgi.uniform_hypergraph_configuration_model(k, i+2, seed=seed)
        edges = H_i.edges.members()
        H.add_edges_from(edges)
        
    return(H)




def stratified_ws_hypergraph (N, ps, seed=None):
    
    """""""""""
    Generates a hypergraph by overlapping multiple layers, each layer being a higher-order 
    Watts-Strogatz-like model for a specific hyper-edge size s. For example, the layer for
    s=3 is created as follow: generate a ring lattice of overlapping 3-hyperedges of the 
    form: [0,1,2], [1,2,3], [2,3,4], ... Then, rewiring: each hyperedge is selected with
    probability ps[s-2] and the first node is kept within the hyperedge, while the others 
    are randomply replaced.
    
    Parameters:
    ---------------
    N (int) : Number of nodes.
    
    ps (list) : Rewiring probability for each hyper-edge size. ps[i] is the rewiring
                probability for hyperedges of size i+2.
                     
    seed (int) : seed for random numbers generator.
    
    ---------------
    Notes: The sizes of the hyperedges are deduced from the lenght of ps. This means that
           if len(ps)==3, there will be hyperedges of size 2,3, and 4.
    
    """""""""""
    
    nodes = range(N)
    H = xgi.Hypergraph()
    H.add_nodes_from(nodes)   
    m_max = len(ps)
    
    for s in range(m_max):              
        H_s = xgi.watts_strogatz_hypergraph(N, s+2, 2, 0, ps[s], seed=seed)
        edges = H_s.edges.members()
        H.add_edges_from(edges)        
        
    return(H)
