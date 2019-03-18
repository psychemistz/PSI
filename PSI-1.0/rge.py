from numpy import *
import time
import bisect

class GaussianProbeSelectionImputation:
    def __init__(self):
        self.probes_to_save = None
        self.probes_to_include = []

    def set_probes_to_include(self, probes_to_include):
        self.probes_to_include = probes_to_include

    def set_probes_to_save(self, probes_to_save):
        self.probes_to_save = probes_to_save

    def train(self, data, max_n_genes):
        (n_genes, n_samples) = data.shape
        self.n_genes = n_genes
        C_11 = zeros([0,0])
        C_22 = self.compute_full_covariance(data)
        self.mu = mean(data, 1)
        C_11i = zeros([0,0])
        C_12 = zeros([0, n_genes])
        self.selected_genes = []
        remaining_genes = range(n_genes)
        self.C_11i_s = []
        self.C_12s = []
        for gene_i in xrange(max_n_genes):
            add_this_probe = False
            if (self.probes_to_save is None) or (gene_i+1 in self.probes_to_save):
                add_this_probe = True
            condC22 = C_22 - dot(dot(transpose(C_12), C_11i), C_12)
            scores = sum(condC22**2, 1) / diag(condC22)
            if gene_i >= len(self.probes_to_include):
                new_gene_i = argmax(scores)
            else:
                new_gene_i = remaining_genes.index(self.probes_to_include[gene_i])
            new_gene = remaining_genes[new_gene_i]
            print("RGE: Added probe %d with score %.3f"%(new_gene, scores[new_gene_i]))
            self.selected_genes.append(new_gene)
            # update matrices
            C_j_S = C_12[:,new_gene_i]
            C_j_j = C_22[new_gene_i, new_gene_i]
            C_j_S_C_11i = dot(C_j_S, C_11i)
            denom = C_j_j - sum(C_j_S * C_j_S_C_11i)
            A_11 = C_j_S_C_11i[newaxis,:] * C_j_S_C_11i[:,newaxis] / denom + C_11i
            A_12 = -C_j_S_C_11i / denom
            A_22 = 1.0 / denom
            C_11i = concatenate([concatenate([A_11, A_12[:,newaxis]], 1), concatenate([A_12, ones([1])*A_22])[newaxis,:]], 0)
            if add_this_probe:
                self.C_11i_s.append(C_11i)
            else:
                self.C_11i_s.append(None)
            v = concatenate([C_12[:,new_gene_i], ones([1]) * C_22[new_gene_i, new_gene_i]])
            C_11 = concatenate([concatenate([C_11, C_12[:,new_gene_i][:,newaxis]], 1), v[newaxis,:]], 0)
            C_12 = concatenate([C_12, C_22[new_gene_i,:][newaxis,:]], 0)
            C_12 = C_12[:,range(new_gene_i)+range(new_gene_i+1,len(remaining_genes))]
            C_22 = C_22[:,range(new_gene_i)+range(new_gene_i+1,len(remaining_genes))][range(new_gene_i)+range(new_gene_i+1,len(remaining_genes)),:]
            remaining_genes = remaining_genes[:new_gene_i] + remaining_genes[new_gene_i+1:]
            if add_this_probe:
                self.C_12s.append(C_12)
            else:
                self.C_12s.append(None)
            #self.C_12 = C_12
            # test imputation
            #X1 = data[self.selected_genes,:]
            #X2 = data[remaining_genes,:]
            #impX2 = self.mu[remaining_genes][:,newaxis] + dot(transpose(C_12), dot(C_11i, X1 - self.mu[self.selected_genes][:,newaxis]))
            #print "Error:",sum((X2-impX2)**2)/(len(remaining_genes) * X2.shape[1])
            #print "|c_11i| =",log(linalg.det(C_11)),log(linalg.det(C_11i))

    def get_selected_genes(self, n_genes):
        return self.selected_genes[:n_genes]

    def get_imputation_params(self, n_genes):
        selected_genes = self.selected_genes[:n_genes]
        remaining_genes = list(set(range(self.n_genes)).difference(set(selected_genes)))
        remaining_genes.sort()
        C_11i = self.C_11i_s[n_genes-1]
        C_21 = transpose(self.C_12s[n_genes-1])
        res = zeros([self.n_genes, n_genes+1])
        M = dot(C_21, C_11i)
        res[remaining_genes,-1] = self.mu[remaining_genes] - dot(M, self.mu[selected_genes])
        res[remaining_genes,:-1] = M
        res[selected_genes,range(n_genes)] = 1.0
        return res

    def impute_from_params(self, params, expr):
        imputation = params[:,-1][:,newaxis] + dot(params[:,:-1], expr)
        return imputation
        
    def impute(self, data):
        n_genes = data.shape[0]
        selected_genes = self.selected_genes[:n_genes]
        remaining_genes = list(set(range(self.n_genes)).difference(set(selected_genes)))
        remaining_genes.sort()
        C_11i = self.C_11i_s[n_genes-1]
        C_21 = transpose(self.C_12s[n_genes-1])
        res = zeros([self.n_genes, data.shape[1]])
        res[remaining_genes,:] = self.mu[remaining_genes][:,newaxis] + dot(C_21, dot(C_11i, data - self.mu[selected_genes][:,newaxis]))
        res[selected_genes,:] = data
        return res

class PCAProbeSelectionImputation(GaussianProbeSelectionImputation):
    def __init__(self, pct_variance = None):
        self.pct_variance = pct_variance
        GaussianProbeSelectionImputation.__init__(self)

    def choose_k_loocv(self, data):
        (n_genes, n_samples) = data.shape
        max_pcs = min(n_samples - 2, n_genes)
        scores = zeros([max_pcs])
        for i in xrange(n_samples):
            trdata = data[:,range(i)+range(i+1,n_samples)]
            mn = mean(trdata, 1)
            trdata = trdata - mn[:,newaxis]
            tsdata = data[:,i] - mn
            U, s, Vh = linalg.svd(trdata, full_matrices=0)
            s = s**2 / n_samples
            rem_evs = sum(s)
            J0_tsdata = zeros([n_genes])
            C1_tsdata = tsdata
            for k in xrange(max_pcs):
                n_rem = max_pcs - k
                mre = rem_evs / n_rem
                scores[k] += sum(log(s[:k])) + n_rem * log(mre) + sum(tsdata * (J0_tsdata + C1_tsdata / mre))
                v = U[:,k] * sum(U[:,k] * tsdata)
                J0_tsdata = J0_tsdata + v / s[k]
                C1_tsdata = C1_tsdata - v
                rem_evs = rem_evs - s[k]
        return argmin(scores)
        
    def learn_model(self, data):
        self.mu = mean(data, 1)
        data0 = data - self.mu[:,newaxis]
        svdres = linalg.svd(data0, full_matrices = 0)
        self.U = svdres[0]
        self.lmb = svdres[1] / sqrt(data.shape[1])
        pctvar = concatenate([zeros([1]), add.accumulate(self.lmb**2) / sum(self.lmb**2)])
        if self.pct_variance is None:
            self.k = self.choose_k_loocv(data)
        else:
            self.k = bisect.bisect(pctvar, self.pct_variance)
        self.pctvar = pctvar[self.k]
        print("Chosen %d components explaining %.4f of the variance"%(self.k, pctvar[self.k]))

    def compute_full_covariance(self, data):
        self.learn_model(data)
        s = self.lmb**2
        mev = mean(s[self.k:])
        C0 = dot(self.U[:,:self.k] * s[:self.k][newaxis,:], transpose(self.U[:,:self.k]))
        C1 = mev * (eye(self.U.shape[0]) - dot(self.U[:,:self.k], transpose(self.U[:,:self.k])))
        C = C0 + C1
        return C

    def compute_full_covariance_old(self, data):
        self.learn_model(data)
        C = dot(self.U[:,:self.k] * self.lmb[:self.k][newaxis,:]**2, transpose(self.U[:,:self.k]))
        D = zeros([C.shape[0], C.shape[0]])
        D[range(C.shape[0]),range(C.shape[0])] = C[range(C.shape[0]),range(C.shape[0])]
        C1 = self.pctvar * C + (1 - self.pctvar) * D
        return C1
        
class ShrinkageCovEstSelectionImputation(GaussianProbeSelectionImputation):
    def __init__(self, alpha_method = 0):
        GaussianProbeSelectionImputation.__init__(self)
        self.alpha_method = alpha_method

    def choose_alpha_loocv(self, data):
        sumX = sum(data, 1)
        mus = (sumX[:,newaxis] - data) * 1.0 / (data.shape[1] - 1)
        baseC = dot(data, transpose(data))
        V = zeros(data.shape)
        Lmb = zeros(data.shape)
        sumlogD = 0.0
        for i in xrange(data.shape[1]):
            print(i)
            C = (baseC - outer(data[:,i], data[:,i])) / (data.shape[1] - 1) - outer(mus[:,i], mus[:,i])
            D = diag(C)
            isD = D**(-0.5)
            M = (C - diag(D)) * outer(isD, isD)
            S, U = linalg.eigh(M)
            V[:,i] = dot(isD * (data[:,i] - mus[:,i]), U)
            Lmb[:,i] = S
            sumlogD += sum(log(D))
        alphas_reso = 1000.0
        alphas = arange(1,alphas_reso) / alphas_reso
        #alphas_scores = zeros(alphas.shape)
        alphas_scores = data.shape[0] * data.shape[1] * log(2 * pi) + sumlogD
        for i in xrange(data.shape[1]):
            alphas_scores += sum(log(1.0 + Lmb[:,i][newaxis,:] * alphas[:,newaxis]), 1)
            alphas_scores += sum(V[:,i][newaxis,:]**2 / (1.0 + Lmb[:,i][newaxis,:] * alphas[:,newaxis]), 1)
        alphas_scores = -0.5 * alphas_scores
        alpha_i = argmax(alphas_scores)
        alpha = alphas[alpha_i]
        print("Chosen alpha =",alpha,"with score =",alphas_scores[alpha_i])
        C = dot(data, transpose(data)) * 1.0 / (data.shape[1]) - outer(sumX, sumX) * 1.0 / (data.shape[1]**2)
        D = diag(C)
        covar_est = alpha * C + (1 - alpha) * diag(D)
        return covar_est

    def shrinked_covar(self, X):
        n = X.shape[1]
        mean_X = mean(X, 1)
        X_zero_mean = X - mean_X[:,newaxis]
        Wbar = dot(X_zero_mean, transpose(X_zero_mean)) * 1.0 / n
        S = n / (n - 1.0) * Wbar
        V = dot(X_zero_mean**2, transpose(X_zero_mean**2)) - n*Wbar**2
        V = V * n / (n-1.0)**3
        Vr = V / (diag(S)[:,newaxis]*diag(S)[newaxis,:])
        Sr = S / sqrt(diag(S)[:,newaxis]*diag(S)[newaxis,:])
        lambdastar = (sum(Vr) - sum(diag(Vr))) / (sum(Sr**2) - sum(diag(Sr**2)))
        lambdastar = max(0, min(1, lambdastar))
        T = diag(diag(S))
        shc = lambdastar * T + (1 - lambdastar) * S
        return shc

    def compute_full_covariance(self, data):
        if self.alpha_method == 0:
            C = shrinked_covar(data)
        else:
            C = self.choose_alpha_loocv(data)
        return C
        
class L2CovEstSelectionImputation(GaussianProbeSelectionImputation):
    def __init__(self, sigmasqrs, CV_K = None):
        GaussianProbeSelectionImputation.__init__(self)
        self.sigmasqrs = sigmasqrs
        self.CV_K = CV_K # (None for Leave-One-Out)

    def cv_choose_sigmasqr(self, sigmasqrs, data, CV_K):
        rndst = random.get_state()
        n_genes, n_samples = data.shape
        sinds = range(n_samples)
        random.shuffle(sinds)
        cut_points = [0] + list(arange(1, CV_K) * n_samples / CV_K) + [n_samples]
        sigmasqr_scores = zeros([len(sigmasqrs)])
        for i in xrange(CV_K):
            tsinds = sinds[cut_points[i]:cut_points[i+1]]
            tsinds.sort()
            trinds = list(set(range(n_samples)).difference(set(tsinds)))
            trinds.sort()
            trdata = data[:,trinds]
            m = mean(trdata, 1)
            trdata = trdata - m[:,newaxis]
            tsdata = data[:,tsinds] - m[:,newaxis]
            U, s, Vh = linalg.svd(trdata, full_matrices = 1)
            #s = s / sqrt(n_samples)
            s2 = s**2 / n_samples
            s2 = concatenate([s2, zeros([data.shape[0] - s2.shape[0]])])
            s3 = 0.5 * (s2[:,newaxis] + sqrt(s2[:,newaxis]**2 + 2.0 / n_samples / sigmasqrs[newaxis,:]))
            s4 = s3**(-1.0)
            for tsi in xrange(len(tsinds)):
                v = dot(tsdata[:,tsi], U)
                sigmasqr_scores += sum(s4 * v[:,newaxis]**2, 0) - sum(log(s4), 0)
        #print "Scores:",sigmasqr_scores
        sigmasqr = sigmasqrs[argmin(sigmasqr_scores)]
        #print "Choosing sigmasqr =",sigmasqr
        self.sigmasqr = sigmasqr
        random.set_state(rndst)
        
    def loocv_choose_sigmasqr(self, sigmasqrs, data):
        if len(sigmasqrs) == 1:
            self.sigmasqr = sigmasqrs[0]
            return
        if self.CV_K is not None:
            return self.cv_choose_sigmasqr(sigmasqrs, data, self.CV_K)
        n_genes, n_samples = data.shape
        sigmasqr_scores = zeros([len(sigmasqrs)])
        for i in xrange(n_samples):
            inds = range(i) + range(i+1,n_samples)
            trdata = data[:,inds]
            m = mean(trdata, 1)
            trdata = trdata - m[:,newaxis]
            tsdata = data[:,i] - m
            try:
                U, s, Vh = linalg.svd(trdata, full_matrices = 1)
            #s = s / sqrt(n_samples)
                s2 = s**2 / n_samples
                s2 = concatenate([s2, zeros([data.shape[0] - s2.shape[0]])])
                s3 = 0.5 * (s2[:,newaxis] + sqrt(s2[:,newaxis]**2 + 2.0 / n_samples / sigmasqrs[newaxis,:]))
                s4 = s3**(-1.0)
                v = dot(tsdata, U)
                sigmasqr_scores += sum(s4 * v[:,newaxis]**2, 0) - sum(log(s4), 0)
            except:
                print("SVD failed for sample %d, skipping"%i)
        #print "Scores:",sigmasqr_scores
        sigmasqr = sigmasqrs[argmin(sigmasqr_scores)]
        #print "Choosing sigmasqr =",sigmasqr
        self.sigmasqr = sigmasqr

    def compute_full_covariance(self, data):
        n_genes, n_samples = data.shape
        self.loocv_choose_sigmasqr(self.sigmasqrs, data)
        mu = mean(data, 1)
        data0 = (data - mu[:,newaxis])
        U, s, Vh = linalg.svd(data0, full_matrices = 1)
        s = s**2 / n_samples
        s = concatenate([s, zeros([data.shape[0] - s.shape[0]])])
        S_tag = 0.5 * s + 0.5 * sqrt(s**2 + 2.0 / n_samples / self.sigmasqr)
        C = dot(U * S_tag[newaxis,:], transpose(U))
        return C
