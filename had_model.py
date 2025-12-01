import numpy as np
import random


class HAD_model:

    def __init__(self, epsilon, groups, opinions=None):
        """
        Class for the higher-order adaptive Deffuant model.

        Parameters:
        -----------
        epsilon (float):
                confidence threshold for agreement (must be in [0,1]).
        
        groups (list of sets):
                hyperedges containing the nodes.
                Each hyperedge must be a set containing the nodes' IDs.

        opinions (dict):
                dictionary containing the opinions (floats), keyed by nodes' IDs.
                If None (default), the opinions are randomly drawn from a uniform distribution.
        
        """
        self.epsilon = epsilon
        self.groups = groups.copy()
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


    
    def simulate(self, T=None, condition='max_min', mu=1., split_overlap=True, split_seed='random',
                 alpha=1., adaptive=True, sgbr=True):
        """
        Simulates the higher-order adaptive Deffuant dynamics.

        Parameters:
        -----------
        T (int):
                number of time steps. If None (default) the model runs until the 
                steady state is reached. The convergence criterion on the opinions 
                is the one presented in https://www.nature.com/articles/s42005-022-00807-4 
                (Methods); after it is satisfied, the process stil run for a number
                of time steps equal to the number of different opinions left. In this
                way, in case of small esplilon, we let the possibility for isolated
                nodes to find a group to converge with. The resolution to compare 
                opinions and count the number of unique ones is set to the 4th decimal.
                
        condition (str):
                rule to decide whether a group converges to a common opinion or not.
                Can be 'std' or 'max_min' (default).
        
        mu (float):
                parameter controlling the update of opinions within a group when
                the agreement rule is satisfied. Default is 1, meaning that all the
                opinions converge to the mean one. The update rule is:
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

        sgbr (bool):
                "Store Groups Before Rewiring", i.e., whether to store the hyperedges 
                at each time step before (default) or after the rewiring of hyperedges 
                coming from a split event.
        -----------
        
        Return (dict):
                the results stored in a dictionary. results['groups'] is a list of length T 
                containing the lists of groups (sets) at each timestep. results['opinions'] is
                a dictionary containing the opinions of nodes across time, keyed by nodes' IDs.
                results['n_events'] is a dictionary keyed by 'agree', 'split', 'merge', 
                containing the lists of number of events of each kind at each time step.
                For example, results['n_events']['split'] is a list of length T where the t-th 
                entry is the number of splits occurred at time t.
        """
        # store initial conditions
        op0 = self.opinions.copy()
        nodes = self.nodes.copy()
        results = {
            'groups': [self.groups.copy()],
            'opinions': {n: [op0[n]] for n in nodes}, 
            'n_events': {'agree': [],
                        'split': [],
                        'merge': []}
                  }
        
        flag = False   # variable managing the steady state
        t = 0
        if T is None:
            TT=2       # inizialize TT to start the process
        else:
            TT = T
        
        while t<TT:

            n_agree, n_split, n_merge = 0, 0, 0
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
                    n_agree += 1
                    mean = np.mean(ops)
                    for n in group:
                        self.opinions[n] = self.opinions[n] * (1-mu) + mu * mean
                # splitting
                else:
                    if adaptive:
                        n_split += 1
                        #keep track of groups that have just split
                        old_groups.append(group)
                        # if the current group is equal to a subgroup resulting from previous
                        # splitting, I remove it from to_rewire because it's going to be split
                        # again and we have to rewire only the minimal subgroups.
                        if group in to_rewire:
                            to_rewire.remove(group)
                        splt = self.split(group, split_overlap, split_seed)
                        to_rewire += [i for i in splt if i not in to_rewire]

            # If sgbr==True, the groups are stored before rewiring.
            if sgbr:
                results['groups'].append(self.groups.copy())            
                
            if adaptive:
                # rewire groups that have just split
                while len(to_rewire)>0:
                    tr = random.choice(to_rewire)
                    p = 1. / ( len(tr) ** alpha)
                    # the group might join another one
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
                        n_merge += 1
                        self.groups.remove(target)
                        self.groups.append(new_g)
                        if target in to_rewire:
                            to_rewire.remove(target)

                    to_rewire.remove(tr)               

            
            # store results for this time step
            op_t = self.opinions.copy()
            for n in nodes:
                results['opinions'][n].append(op_t[n])
            results['n_events']['agree'].append(n_agree)
            results['n_events']['split'].append(n_split)
            results['n_events']['merge'].append(n_merge)
            # store hyperedges if not done before
            if not sgbr:
                results['groups'].append(self.groups.copy())
                
            t+=1
            if T is None:
                # verify convergence condition on opinions
                conv = sum(
                    [ abs(op_t[n] - results['opinions'][n][-2]) for n in nodes ]
                )
                if conv < 0.001 and not flag:
                    # number of different opinions (cut to 4th decimal) at steady state
                    n_ops = len( {np.round(op_t[n], 4) for n in nodes} )
                    # if opinions converge, I let a bit of time for the structure to relax.
                    # In case of small epsilon, after convergence there will still be small
                    # groups of nodes searching for a group to agree with. By setting
                    # TT = t + n_ops, I give them the possibility to explore other groups
                    # with possibly different opinions. If in the meantime the opinions
                    # restart evolving, I keep the simulation running and discard the first 
                    # convergence (this is done via the 'flag' variable).
                    TT = t + n_ops
                    flag = True
                    #print('opinions converged at time', t-1, 'continuing for', n_ops, 'more time steps.')
                elif conv > 0.001:
                    TT = t+2   # keep the process running
                    flag = False
                
        return results
        