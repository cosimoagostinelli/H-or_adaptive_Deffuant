import numpy as np
import random


class HAD_model:

    def __init__(self, epsilon, groups, opinions=None):
        """
        Class for the higher-order adaptive Deffuant model.

        Parameters:
        -----------
        epsilon (float):
                threshold for convergence of opinions (must be in [0,1]).
        
        groups (list of sets):
                hyperedges containing the nodes.
                Each hyperedge must be a set containing the nodes' IDs.

        opinions (dict):
                dictionary containing the opinions (floats), keyed by nodes' IDs.
                If None (default), the opinions are randomly drawn from a uniform distribution.
        
        """
        self.epsilon = epsilon
        self.groups = groups
        self.nodes = set()
        for gr in groups:
            self.nodes |= gr
        if opinions is None:
            vals = np.random.random(len(self.nodes))
            self.opinions = {n:o for n,o in zip(self.nodes,vals)}
        else:
            self.opinions = opinions

    
    
    def split(self, group, split_overlap, split_seed):
        """
        Split the incoming group into multiple sub-groups. Modifies the
        interanl structure of the object (self.groups) and returns the new groups.
        
        Parameters:
        -----------
        group (set):
                group of nodes to split.
                
        split_overlap (bool):
                whether to admit overlap between splitting groups.

        split_seed (str):
                how to choose the seed for splitting the group. Can be either 'random'
                (default), 'degree', or 'extreme'.
        """
        new_groups = []
        seeds_left = group.copy()
        passive_nodes = group.copy()
        switch = 1
        nd = 1
        
        while nd<len(group):
            
            # first select the center (seed) of the new group
            if split_seed=='random':
                seed = random.choice(list(seeds_left))
                
            elif split_seed=='degree':
                d_max = 0
                for n in seeds_left:
                    # degree of node n
                    d_n = len([gn for gn in self.groups if n in gn])
                    if d_n > d_max:
                        d_max = d_n
                        seed = n
                        
            elif split_seed=='extreme':
                op_n = {self.opinions[n]: n for n in seeds_left}
                # alternate between maximum and minimum opinion
                if switch:
                    seed = op_n[min(op_n.keys())]
                else:
                    seed = op_n[max(op_n.keys())]
                switch = abs(switch-1)
                
            # take neighbors within epsilon distance
            ng = {i for i in passive_nodes if abs(self.opinions[seed] - self.opinions[i]) < self.epsilon}
            new_groups.append(ng)
            nodes_done = set([i for j in new_groups for i in j])
            nd = len(nodes_done)
            seeds_left -= nodes_done
            if not split_overlap:
                passive_nodes = seeds_left
        
        self.groups.remove(group)
        for ngp in new_groups:
            if ngp not in self.groups:
                self.groups.append(ngp)
                
        return new_groups


    
    def simulate(self, T=None, condition='std', mu=1., split_overlap=True, split_seed='random', alpha=1., adaptive=True):
        """
        Simulates the higher-order adaptive Deffuant dynamics.

        Parameters:
        -----------
        T (int):
                number of time steps. If None (default) the model runs until the steady
                state is reached. The convergence criterion is the one presented in
                https://www.nature.com/articles/s42005-022-00807-4 (see Methods).
                
        condition (str):
                rule to decide whether a group converges to a common opinion or not.
                Can be 'std' (default) of max_min.
        
        mu (float):
                parameter controlling the convergence of opinions within a group when
                the convergence rule is satisfied. Default is 1, meaning that all the
                opinions converge to the mean one. The convergence rule is:
                x_i (t+1)  =  mu * <x>  +  x_i (t) * (1-mu) ,
                where <x> is the average opinion of the group.
                
        split_overlap (bool):
                whether to allow overlap between splitting groups (default = True).

        split_seed (str):
                how to choose the seed for splitting the group. Can be either 'random'
                (default), 'degree', or 'extreme'.

        alpha (float):
                parameter controlling: (i) the probability of joining (or not) a group 
                after splitting and (ii) the probability to join a group of a given size.

        adaptive (bool):
                whether to simulate an adaptive process, i.e. with split and rewiring of
                groups (default), or just a Deffuant dynamics on a static hypergraph.
        -----------
        
        Return (dict):
                the results stored in a dictionary. results['groups'] is a list of length T 
                containing the list of groups (set) at each timestep. results['opinions'] is
                a dictionary containing the opinions of nodes across time, keyed by nodes' IDs.
        """
        op0 = self.opinions.copy()
        nodes = self.nodes.copy()
        results = {'groups': [self.groups.copy()],
                  'opinions': {n: [op0[n]] for n in nodes}
                  }
        if T is None:
            TT=1
        else:
            TT = T
        t = 0
        while t<TT:

            old_groups, to_rewire = [], []
            grps = self.groups.copy()
            random.shuffle(grps)
            # all groups discuss and either agree or split
            for group in grps:
                ops = [self.opinions[i] for i in group]
                if condition=='std':
                    val = np.std(ops, ddof=1)
                elif condition=='max_min':
                    val = max(ops) - min(ops)
                # agreement
                if val < self.epsilon:
                    mean = np.mean(ops)
                    for n in group:
                        self.opinions[n] = self.opinions[n] * (1-mu) + mu * mean
                # splitting
                else:
                    if adaptive:
                        #keep track of groups that have just split
                        old_groups.append(group)
                        # if the current group is equal to a subgroup resulting from previous
                        # splitting, I remove it from to_rewire because it's going to be split
                        # again and we have to rewire only the minimal subgroups.
                        if group in to_rewire:
                            to_rewire.remove(group)
                        splt = self.split(group, split_overlap, split_seed)
                        to_rewire += [i for i in splt if i not in to_rewire]

            if adaptive:
                # rewire groups that have just split
                while len(to_rewire)>0:
                    tr = random.choice(to_rewire)
                    p = 1. / ( len(tr) ** alpha)
                    # the group joins another one
                    if random.random() < p:
                        self.groups.remove(tr)
                        probs = [1. / (len(j|tr) ** alpha) for j in self.groups]
                        probs = np.array(probs) / sum(probs)
                        new_g = old_groups[0]
                        while new_g in old_groups or new_g in self.groups:
                            idx = np.random.choice(len(self.groups), p=probs)
                            target = self.groups[idx]
                            new_g = tr|target
                        # merge groups
                        self.groups.remove(target)
                        self.groups.append(new_g)
                        to_rewire.remove(tr)
                        if target in to_rewire:
                            to_rewire.remove(target)
                    
                    # the group does not rewire
                    else:
                        to_rewire.remove(tr)

            op_t = self.opinions.copy()
            for n in nodes:
                results['opinions'][n].append(op_t[n])
            results['groups'].append(self.groups.copy())
            
            t+=1
            if T is None:
                convergence = sum(
                    [ abs(op_t[n] - results['opinions'][n][-2]) for n in nodes ]
                )
                if convergence < 0.001:
                    break
                else:
                    TT+=1
                
        return results
        