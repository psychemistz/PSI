from numpy import *
import bisect
import l2imputer

class L1InfPSI:
    def __init__(self, lmb_params = (0.97, 1.0, 1.25) ):
        self.lmb1_mult, self.lmb2_0, self.lmb2_mult = lmb_params
        self.l2i = l2imputer.L2Imputer([])
        self.probes_to_save = None
        self.probes_to_include = []

    def set_probes_to_save(self, probes_to_save):
        self.probes_to_save = probes_to_save
        
    def set_probes_to_include(self, probes_to_include):
        self.probes_to_include = probes_to_include

    def optimize_row2(self, j1):
        if self.W[j1] is None:
            u1 = self.resiX
        else:
            u1 = self.resiX + self.W[j1] * self.Xsq[j1]
        #u1[j1] = 0.0
        nrm = self.Xsq[j1]
        opts2 = u1 / nrm
        opts2[j1] = 0.0
        asi = argsort(abs(opts2))
        sopts2 = take(abs(opts2), asi)
        m_opts = maximum(0.0,((add.accumulate(sopts2[::-1]) - self.lmb1 / nrm) / arange(1, sopts2.shape[0] + 1))[::-1])
        m_i = nonzero(m_opts <= sopts2)[0][0]
        M_opt = m_opts[m_i]
        if (m_i == 0) or (M_opt == 0.0):
            if self.W[j1] is None:
                return
            self.W[j1] = None
            self.resiX = u1
            return
        self.W[j1] = zeros([self.n_genes])
        self.W[j1][asi[:m_i]] = opts2[asi[:m_i]]
        self.W[j1][opts2 >= M_opt] = M_opt
        self.W[j1][opts2 <= -M_opt] = -M_opt
        self.resiX = u1 - self.W[j1] * self.Xsq[j1]
        #self.resi = resi1 - self.W[j1][:,newaxis] * self.X[j1,:][newaxis,:]
        return        
        
    def optimize_row(self, j1):
        if sum(self.X[j1,:]**2) == 0.0:
            return
        if self.W[j1] is None:
            resi1 = self.resi
        else:
            resi1 = self.resi + self.W[j1][:,newaxis] * self.X[j1,:][newaxis,:]
        #C001 resi1 = self.resi + self.W[j1,:][:,newaxis] * self.X[j1,:][newaxis,:]
        u1 = sum(self.X[j1,:][newaxis,:] * resi1, 1)
        u1[j1] = 0.0
        nrm = sum(self.X[j1,:]**2)
        opts2 = u1 / nrm
        asi = argsort(abs(opts2))
        sopts2 = take(abs(opts2), asi)
        my_lmb1 = self.lmb1
        if self.must_include[j1]:
            my_lmb1 = 0.0
        m_opts = maximum(0.0,((add.accumulate(sopts2[::-1]) - my_lmb1 / nrm) / arange(1, sopts2.shape[0] + 1))[::-1])
        m_i = nonzero(m_opts <= sopts2)[0][0]
        M_opt = m_opts[m_i]
        if (m_i == 0) or (M_opt == 0.0):
            if self.W[j1] is None:
                return
            self.W[j1] = None
            self.resi = resi1
            return
        self.W[j1] = zeros([self.n_genes])
        self.W[j1][asi[:m_i]] = opts2[asi[:m_i]]
        self.W[j1][opts2 >= M_opt] = M_opt
        self.W[j1][opts2 <= -M_opt] = -M_opt
        self.resi = resi1 - self.W[j1][:,newaxis] * self.X[j1,:][newaxis,:]
        return
        
    def l1inf_coordinate_descent2(self, X, lmb1, W0 = None):
        (self.n_genes, self.n_samples) = X.shape
        self.resi = X.copy()
        if W0 is None:
            self.W = [None] * self.n_genes
        else:
            self.W = list(W0)
            nzg = [x for x in xrange(len(self.W)) if self.W[x] is not None]
            W1 = array([self.W[x] for x in nzg])
            self.resi = self.resi - dot(transpose(W1), X[nzg,:])
        j1 = 0
        self.lmb1 = lmb1
        self.X = X
        prv_score = None
        prv_nzi = None
        while True:
            # optimize W[j1,:] fixing all others
            self.optimize_row(j1)
            j1 = j1 + 1
            if (j1 == self.n_genes):
                j1 = 0
                ll = 0.5 * sum(self.resi**2)
                score = ll + self.lmb1 * sum([max(abs(x)) for x in self.W if x is not None])
                nzi = [x for x in xrange(len(self.W)) if self.W[x] is not None]
                if (prv_nzi is None) or (nzi != prv_nzi):
                    prv_nzi = nzi
                    nzi_count = 0
                else:
                    nzi_count += 1
                    if (nzi_count >= 10):
                        break
                if (prv_score is None) or (prv_score - score > 1e-6):
                    prv_score = score
                else:
                    break
        return self.W

    def l1inf_coordinate_descent(self, X, lmb1, W0 = None):
        (n_genes, n_samples) = X.shape
        if W0 is None:
            W = zeros([n_genes, n_genes])
        else:
            W = W0.copy()
        last_changed = (n_genes - 1, n_genes - 1)
        j1 = 0
        j2 = 0
        der_eps = 1e-6
        proberrs = sum(X**2,1)
        verify_new_weight = False
        prv_score = None
        while True:
            if (j1 == 0) and (j2 == 0):
                preds = dot(transpose(W), X)
                score = sum((preds - X)**2) + lmb1 * sum(maximum.reduce(abs(W), 1))
                print("New iteration. Score:", score)
                if (prv_score is None) or (prv_score - score > 0.01):
                    prv_score = score
                else:
                    print("Stopping.")
                    break
            if (j1 == j2):
                is_opt = True
            else:
                # optimize weight at (j1, j2)
                resi = X[j2,:] - dot(W[:,j2], X)
                der1 = -sum(X[j1,:] * resi)
                if any(abs(W[j1,:]) > abs(W[j1,j2])):
                    is_opt = (abs(der1) < der_eps)
                else:
                    if W[j1,j2] == 0.0:
                        is_opt = (abs(der1) <= lmb1)
                    elif W[j1,j2] > 0.0:
                        is_opt = (der1 >= -lmb1)
                    else:
                        is_opt = (der1 <= lmb1)
            if not is_opt:
                last_changed = (j1, j2)
                resi2 = resi + W[j1,j2] * X[j1,:]
                W[j1,j2] = 0.0
                max_others = max(abs(W[j1,:]))
                w_opt = sum(X[j1,:] * resi2) / sum(X[j1,:]**2)
                if abs(w_opt) <= max_others:
                    W[j1,j2] = w_opt
                elif w_opt > max_others:
                    resi3 = resi2 - max_others * X[j1,:]
                    der2 = -sum(X[j1,:] * resi3)
                    if der2 >= -lmb1:
                        w_opt = max_others
                    else:
                        w_opt = (sum(X[j1,:] * resi2) - lmb1) / sum(X[j1,:]**2)
                    W[j1,j2] = w_opt
                elif w_opt < -max_others:
                    resi3 = resi2 + max_others * X[j1,:]
                    der2 = -sum(X[j1,:] * resi3)
                    if der2 <= lmb1:
                        w_opt = -max_others
                    else:
                        w_opt = (sum(X[j1,:] * resi2) + lmb1) / sum(X[j1,:]**2)
                    W[j1,j2] = w_opt
                else:
                    print("Error! Should never get here.")
                    raise
                if verify_new_weight:
                    resi = X[j2,:] - dot(W[:,j2], X)
                    proberrs[j2] = sum(resi**2)
                    print("New score:",sum(proberrs) + lmb1 * sum(maximum.reduce(abs(W), 1)))
                    der1 = sum(X[j1,:] * resi)
                    if any(abs(W[j1,:]) > abs(W[j1,j2])):
                        is_opt = (abs(der1) < der_eps)
                    else:
                        if W[j1,j2] == 0.0:
                            is_opt = (abs(der1) <= lmb1)
                        elif W[j1,j2] > 0.0:
                            is_opt = (der1 >= -lmb1)
                        else:
                            is_opt = (der1 <= lmb1)
                    if not is_opt:
                        print("ERROR! New weight not optimal. What now?")
                        raise
            else:
                if (j1,j2) == last_changed:
                    print("Coordinate descent finished.")
                    break            
            j2 = j2 + 1
            if (j2 == n_genes):
                j2 = 0
                j1 = j1 + 1
                if (j1 == n_genes):
                    j1 = 0
        return W
    
    def solve_path(self, data, max_n_genes):
        W = None
        selections = [None] * (max_n_genes + 1)
        selections[0] = []
        X = data - mean(data, 1)[:,newaxis]
        #lmb0 = max(sum(abs(dot(X, transpose(X))),1)-sum(X**2,1))
        lmb0 = max([sum(abs(dot(X, x))) - sum(x**2) for x in X])
        lmb = lmb0
        min_n_genes = len(self.probes_to_include)
        for i in xrange(min_n_genes):
            selections[i+1] = self.probes_to_include[:i+1]
        for i in xrange(min_n_genes, max_n_genes):
            while True:
                prv_lmb = lmb
                while True:
                    lmb = lmb * self.lmb1_mult
                    W = self.l1inf_coordinate_descent2(X, lmb, W)
                    #print lmb, len([x for x in W if x is not None])
                    #C001 print lmb, sum(maximum.reduce(abs(W), 1) > 0.0)
                    #C001 if sum(maximum.reduce(abs(W),1) > 0.0) > i:
                    if len([x for x in W if x is not None]) > i:
                        break
                    prv_lmb = lmb
                #C001 while sum(maximum.reduce(abs(W),1) > 0.0) > (i + 1):
                lmb_search_iters = 0
                while len([x for x in W if x is not None]) > (i + 1):
                    lmb_search_iters += 1
                    if (lmb_search_iters > 10):
                        while len([x for x in W if x is not None]) > (i + 1):
                            nzg = [x for x in xrange(len(W)) if W[x] is not None]
                            norms = [max(abs(W[x])) for x in nzg]
                            min_norm_i = argmin(norms)
                            W[nzg[min_norm_i]] = None
                        break
                    lmb = exp(0.5 * (log(lmb) + log(prv_lmb)))
                    W = self.l1inf_coordinate_descent2(X, lmb, W)
                    #C001 print lmb, sum(maximum.reduce(abs(W), 1) > 0.0)
                    #print lmb, len([x for x in W if x is not None])
                #C001 nzg = list(nonzero(maximum.reduce(abs(W),1) > 0.0)[0])
                nzg = [x for x in xrange(len(W)) if W[x] is not None]
                if len(nzg) == (i + 1):
                    break
            selections[i+1] = nzg
            print("%d probes with lambda=%.4f"%(i+1, lmb))
        self.selections = selections
        return

    def train(self, data, max_n_genes):
        self.must_include = [False for x in xrange(data.shape[0])]
        for i in self.probes_to_include:
            self.must_include[i] = True
        self.solve_path(data, max_n_genes)
        self.mu = mean(data, 1)
        #if self.imp_lmbs is not None:
        print("Resolving using L2...")
        self.solutions2 = [None] * len(self.selections)
        self.solutions2[0] = (zeros([data.shape[0], 0]), zeros([0]), self.mu)
        lmb2_0 = self.lmb2_0
        for sol_i in xrange(1,len(self.selections)):
            if (self.probes_to_save is None) or (sol_i in self.probes_to_save):
                l2i = l2imputer.L2Imputer2((lmb2_0, self.lmb2_mult))
                W, mu0, mu1 = l2i.train_probeset(data, self.selections[sol_i])
                self.solutions2[sol_i] = W, mu0, mu1
                lmb2_0 = l2i.prv_lmb

    def get_selected_genes(self, n_genes):
        return self.selections[n_genes]

    def get_imputation_params(self, n_genes):
        selected_genes = self.selections[n_genes]
        remaining_genes = list(set(range(self.n_genes)).difference(set(selected_genes)))
        remaining_genes.sort()
        W, mu0, mu1 = self.solutions2[n_genes]
        res = zeros([self.n_genes, n_genes+1])
        res[remaining_genes,-1] = mu1 - dot(W, mu0)
        res[remaining_genes,:-1] = W
        res[selected_genes,range(n_genes)] = 1.0
        return res

    def impute_from_params(self, params, expr):
        imputation = params[:,-1][:,newaxis] + dot(params[:,:-1], expr)
        return imputation
        
    def impute(self, data):
        n_genes = data.shape[0]
        probeset = self.selections[n_genes]
        imp = self.l2i.impute(data, self.solutions2[n_genes])
        res = zeros([self.mu.shape[0], data.shape[1]])
        res[probeset,:] = data
        remaining = list(set(range(self.mu.shape[0])).difference(set(probeset)))
        remaining.sort()
        res[remaining,:] = imp
        return res

