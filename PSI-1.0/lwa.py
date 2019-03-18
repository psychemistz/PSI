from numpy import *

class NearestNeighborProbeSelectionImputation:
    def __init__(self, use_random = False):
        self.use_random = use_random
        self.probes_to_save = None
        self.probes_to_include = []

    def set_probes_to_include(self, probes_to_include):
        self.probes_to_include = probes_to_include

    def set_probes_to_save(self, probes_to_save):
        self.probes_to_save = probes_to_save

    def train(self, data, max_n_genes, gene_subset = None):
        # do greedy selection
        (n_genes, n_samples) = data.shape
        self.n_genes = n_genes
        self.n_samples = n_samples
        dists = zeros([n_samples, n_samples])
        self.selected_genes = []
        self.selected_genes_set = set()
        max_dist = (maximum.reduce(maximum.reduce(data, 0), 0) - minimum.reduce(minimum.reduce(data, 0), 0)) ** 2
        dists0 = -eye(n_samples)
        #dists0 = max_dist * eye(n_samples)
        full_dists = sum(data**2,0)[newaxis,:] + sum(data**2,0)[:,newaxis] - 2 * dot(transpose(data), data)
        if gene_subset is None:
            gene_subset = range(self.n_genes)
        for gene_i in xrange(max_n_genes):
            if self.use_random:
                scores = random.random([len(gene_subset)])
            else:
                scores = zeros([len(gene_subset)])
                for sample_i in xrange(n_samples):
                    #new_dists = dists0[:,sample_i][newaxis,:] + dists[:,sample_i][newaxis,:] + (data[:,sample_i][:,newaxis] - data)**2
                    new_dists = dists0[:,sample_i][newaxis,:] + dists[:,sample_i][newaxis,:] + (data[gene_subset,sample_i][:,newaxis] - data[gene_subset,:])**2
                    NNs = argmin(new_dists, 1)
                    NNs = argsort(new_dists, 1)[:,1]
                    fd = full_dists[:,sample_i] - dists[:,sample_i]
                    scores = scores + take(fd, NNs) - (data[gene_subset,sample_i] - data[gene_subset, NNs])**2
            #new_gene = argmin(scores)
            new_gene = gene_subset[argmin(scores)]
            print(gene_i, new_gene)
            #new_gene = random.randint(0, n_genes)
            self.selected_genes.append(new_gene)
            self.selected_genes_set.add(new_gene)
            dists = dists + (data[new_gene,:][newaxis,:] - data[new_gene,:][:,newaxis])**2
        self.data = data
            
    def get_selected_genes(self, n_genes):
        #print self.selected_genes[:n_genes]
        return self.selected_genes[:n_genes]

    def impute(self, data):
        n_genes = data.shape[0]
        orig_data = self.data[self.selected_genes[:n_genes],:]
        dists = sum(orig_data**2, 0)[newaxis,:] + sum(data**2, 0)[:,newaxis] - 2 * dot(transpose(data), orig_data)
        NNs = argmin(dists, 1)
        res = zeros([self.n_genes, data.shape[1]])
        for i in xrange(data.shape[1]):
            res[:,i] = self.data[:,NNs[i]]
        return res

class LWAPSI:
    def __init__(self, sigmasqr_mult = 1.25, CV_K = 5):
        self.sigmasqr_mult = sigmasqr_mult
        self.probes_to_include = []
        self.CV_K = CV_K

    def set_probes_to_include(self, probes_to_include):
        self.probes_to_include = probes_to_include

    def compute_tstr_dists(self, tsdata, trdata, probes):
        dists = sum((tsdata[probes,:][:,:,newaxis] - trdata[probes,:][:,newaxis,:])**2,0)
        return dists

    def select_next_probe(self):
        scores = zeros([len(self.possible_probes)])
        for cv_k in xrange(self.CV_K):
            trdata = self.data[:,self.training_inds[cv_k]]
            tsdata = self.data[:,self.test_inds[cv_k]]
            cur_dists = self.compute_tstr_dists(tsdata, trdata, self.selected_genes)
            best_score = None
            for new_probe_i in xrange(len(self.possible_probes)):
                new_probe = self.possible_probes[new_probe_i]
                rem = [x for x in self.remaining_genes if x != new_probe]
                dists = (tsdata[new_probe,:][:,newaxis] - trdata[new_probe,:][newaxis,:])**2
                new_dists = cur_dists + dists
                logw = -0.5*new_dists/self.kernel_widths[-1]
                mw = maximum.reduce(logw, 1)
                unnw = exp(logw - mw[:,newaxis])
                nw = unnw / sum(unnw, 1)[:,newaxis]
                pr = dot(trdata[rem,:], transpose(nw))
                sse = sum((pr-tsdata[rem,:])**2)
                scores[new_probe_i] += sse
        new_probe_i = argmin(scores)
        new_probe = self.possible_probes[new_probe_i]
        self.selected_genes.append(new_probe)
        self.update_remaining_probes()
        self.select_next_kernel_width()

    def select_first_probe(self):
        # to pick the first probe use NN since we don't have a kernel width yet
        scores = zeros(len(self.gene_subset))
        for cv_k in xrange(self.CV_K):
            trdata = self.data[:,self.training_inds[cv_k]]
            tsdata = self.data[:,self.test_inds[cv_k]]
            for probe_i in xrange(len(self.gene_subset)):
                probe = self.gene_subset[probe_i]
                dists = (tsdata[probe,:][:,newaxis] - trdata[probe,:][newaxis,:])**2
                NN_inds = argmin(dists, 1)
                prdata = transpose(array([trdata[:,NN_i] for NN_i in NN_inds]))
                scores[probe_i] += sum((prdata-tsdata)**2)
        first_probe = self.gene_subset[argmin(scores)]
        self.selected_genes.append(first_probe)
        self.update_remaining_probes()
        self.select_next_kernel_width()

    def select_next_kernel_width(self):
        if len(self.kernel_widths) > 0:
            kw = self.kernel_widths[-1]
        else:
            kw = 1.0
        prv_err = self.score_kernel_width(kw)
        up_err = self.score_kernel_width(kw * self.sigmasqr_mult)
        if (up_err < prv_err):
            # going up!
            prv_err = up_err
            kw *= self.sigmasqr_mult
            while True:
                up_err = self.score_kernel_width(kw * self.sigmasqr_mult)
                if up_err >= prv_err:
                    break
                prv_err = up_err
                kw *= self.sigmasqr_mult
        else:
            # going down or staying in place
            down_err = self.score_kernel_width(kw / self.sigmasqr_mult)
            if (down_err < prv_err):
                # going down!
                prv_err = down_err
                kw /= self.sigmasqr_mult
                while True:
                    down_err = self.score_kernel_width(kw / self.sigmasqr_mult)
                    if down_err >= prv_err:
                        break
                    prv_err = down_err
                    kw /= self.sigmasqr_mult
            else:
                # staying in place
                pass
        self.kernel_widths.append(kw)
        return kw

    def score_kernel_width(self, kernel_width):
        sse = 0.0
        for cv_k in xrange(self.CV_K):
            trdata = self.data[:,self.training_inds[cv_k]]
            tsdata = self.data[:,self.test_inds[cv_k]]
            dists = self.compute_tstr_dists(tsdata, trdata, self.selected_genes)
            logw = -0.5*dists/kernel_width
            mw = maximum.reduce(logw, 1)
            unnw = exp(logw - mw[:,newaxis])
            nw = unnw / sum(unnw, 1)[:,newaxis] # shape (n_test, n_training)
            pr = dot(trdata[self.remaining_genes,:], transpose(nw)) # shape (n_remaining, n_test)
            sse += sum((tsdata[self.remaining_genes,:] - pr)**2)
        return sse
 
    def update_remaining_probes(self):
        self.selected_genes_set = set(self.selected_genes)
        self.remaining_genes = list(set(range(self.n_genes)).difference(self.selected_genes_set))
        self.remaining_genes.sort()
        possible_probes = list(set(self.gene_subset).difference(self.selected_genes_set))
        possible_probes.sort()
        self.possible_probes = possible_probes

    def make_cv_splits(self):
        n = self.data.shape[1]
        all_inds = arange(n)
        random.shuffle(all_inds)
        cut_points = concatenate([zeros([1],dtype='int32'),
                                  (arange(1,self.CV_K) * n / self.CV_K).astype('int32'),
                                  ones([1],dtype='int32')])
        self.test_inds = [all_inds[cut_points[split_i]:cut_points[split_i+1]] for split_i in range(self.CV_K)]
        self.training_inds = [list(set(all_inds).difference(test_inds)) for test_inds in self.test_inds]
        for i in xrange(len(self.training_inds)):
            self.training_inds[i].sort()
            self.test_inds[i].sort()

    def train(self, data, n_genes, gene_subset = None):
        self.n_genes, self.n_samples = data.shape
        if gene_subset is None:
            gene_subset = range(data.shape[0])
        self.gene_subset = gene_subset
        self.selected_genes = []
        self.remaining_genes = list(self.gene_subset)
        self.kernel_widths = []
        self.data = data
        self.make_cv_splits()
        if len(self.probes_to_include) > 0:
            self.selected_genes = list(self.probes_to_include)
            self.update_remaining_probes()
            self.kernel_widths = [None] * (len(self.selected_genes)-1)
            self.select_next_kernel_width()
        else:
            # a hack to choose the first probe
            self.select_first_probe()
        for gene_i in xrange(n_genes-len(self.selected_genes)):
            self.select_next_probe()
            print("%d probes selected."%(len(self.selected_genes)))
            
    def get_selected_genes(self, n_genes):
        return self.selected_genes[:n_genes]

    def get_imputation_params(self, n_genes):
        return self.kernel_widths[n_genes-1]

    def impute_from_params(self, params, data):
        n_genes, n_samples = data.shape
        (kw,self.selected_genes,trdata) = params
        use_genes = self.get_selected_genes(n_genes)
        dists = sum((data[:,:,newaxis] - trdata[use_genes,:][:,newaxis,:])**2,0)
        logw = -0.5*dists/kw
        mw = maximum.reduce(logw, 1)
        unnw = exp(logw - mw[:,newaxis])
        nw = unnw / sum(unnw, 1)[:,newaxis]
        imp = zeros([trdata.shape[0], n_samples])
        set_use_genes = set(use_genes)
        rem = [x for x in xrange(trdata.shape[0]) if x not in set_use_genes]
        imp[rem,:] = dot(trdata[rem,:], transpose(nw))
        imp[use_genes,:] = data
        return imp
        
    def impute(self, data):
        n_genes, n_samples = data.shape
        kw = self.kernel_widths[n_genes-1]
        return self.impute_from_params(kw, data)

