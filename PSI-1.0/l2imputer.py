from numpy import *

class L2Imputer:
    def __init__(self, lmbs):
        self.lmbs = lmbs

    def train_probeset(self, data, probeset):
        (n_genes, n_samples) = data.shape
        remaining_genes = list(set(range(n_genes)).difference(set(probeset)))
        remaining_genes.sort()
        data_selected = data[probeset,:]
        data_remaining = data[remaining_genes,:]
        
        U_X, s_X, Vh_X = linalg.svd(data_selected, full_matrices = 0)
        sumX = sum(data_selected, 1)[:,newaxis] - data_selected
        sumXrem = sum(data_remaining, 1)[:,newaxis] - data_remaining
        scores = zeros([len(self.lmbs)])
        YXt = dot(data_remaining, transpose(data_selected))
        for sample_i in xrange(n_samples):
            mu_Xmi = sumX[:,sample_i] / (n_samples - 1.0)
            mu_Ymi = sumXrem[:,sample_i] / (n_samples - 1.0)
            yrem = data_remaining[:,sample_i]
            v1 = data_selected[:,sample_i]
            v2 = sumX[:,sample_i]
            y = v1 - v2 / (n_samples - 1.0)
            z1 = dot(v1, U_X)
            z2 = dot(v2, U_X)
            z3 = z1 - z2 / (n_samples - 1.0)
            M0 = YXt - outer(yrem, v1) - outer(sumXrem[:,sample_i], mu_Xmi) - outer(mu_Ymi, sumX[:,sample_i]) + outer(mu_Ymi, mu_Xmi)
            for lmb_i in xrange(len(self.lmbs)):
                lmb = self.lmbs[lmb_i]
                new_eigs = (s_X**2 + lmb)**(-1.0)
                w1 = dot(U_X, new_eigs * z1)
                w2 = dot(U_X, new_eigs * z2) + w1 * sum(w1 * v2) / (1.0 - sum(w1 * v1))
                XXt_i_y = dot(U_X, new_eigs * z3) + w1 * sum(w1 * y) / (1.0 - sum(w1 * v1)) + w2 * sum(w2 * y) / (n_samples - 1.0 - sum(w2 * v2))
                pr = dot(M0, XXt_i_y)
                scores[lmb_i] += sum((pr - yrem)**2)
                #A1 = dot(U_X * ((s_X**2 + lmb)**(-1.0))[newaxis,:], transpose(U_X))
                #w1 = dot(A1, v1)
                #A2 = A1 + outer(w1, w1) / (1.0 - sum(w1 * v1))
                #w2 = dot(A2, v2)
                #A3 = A2 + outer(w2, w2) / (n_sample - 1.0 - sum(w2 * v2))
        #print "Scores:",scores
        lmbi = argmin(scores)
        lmb = self.lmbs[lmbi]
        print("Chosen lambda =",lmb)
        data0_selected = data_selected - mean(data_selected, 1)[:,newaxis]
        data0_remaining = data_remaining - mean(data_remaining, 1)[:,newaxis]
        U, s, Vh = linalg.svd(data0_selected, full_matrices = 0)
        new_eigs = (s**2 + lmb)**(-1.0)
        W0 = dot(U * new_eigs[newaxis,:], transpose(U))
        Wt = dot(W0, dot(data0_selected, transpose(data0_remaining)))
        W = transpose(Wt)
        mu0 = mean(data_selected, 1)
        mu1 = mean(data_remaining, 1)
        return W, mu0, mu1

    def impute(self, data, params):
        W, mu0, mu1 = params
        return dot(W, data - mu0[:,newaxis]) + mu1[:,newaxis]


class L2Imputer2:
    def __init__(self, lmb2_params):
        (self.lmb_0, self.lmb_mult) = lmb2_params
        self.prv_lmb = self.lmb_0

    def train_probeset(self, data, probeset):
        data0 = data - mean(data, 1)[:,newaxis]
        obs_i = probeset
        rem_i = list(set(range(data.shape[0])).difference(set(obs_i)))
        rem_i.sort()
        sumX = sum(data0, 1)
        mnsX = (sumX[:,newaxis] - data0) / (data.shape[1] - 1.0)
        U,s,Vh = linalg.svd(data0[obs_i,:], full_matrices=0)
        lmb = self.prv_lmb
        ssft = 0

        while True:
            s2 = s**2 + lmb
            XXti = dot(U * (s2**(-1))[newaxis,:], transpose(U))
            V = dot(U * (s2**(-1))[newaxis,:], dot(transpose(U), data0[obs_i,:]))
            score = 0.0
            for i in xrange(data.shape[1]):
                R = XXti + outer(V[:,i], V[:,i]) / (1.0 - sum(data0[:,i][obs_i] * V[:,i]))
                # verify that R is the inverse of XXt - outer(data0[:,i], data0[:,i])
                v1 = dot(R, mnsX[:,i][obs_i])
                M = R + outer(v1, v1) / (data.shape[1] - 1.0) / (1.0 - sum(mnsX[:,i][obs_i] * v1) / (data.shape[1] - 1.0))
                inds = range(i) + range(i+1, data.shape[1])
                trX = data0[obs_i,:][:,inds] - mnsX[:,i][obs_i][:,newaxis]
                trY = data0[rem_i,:][:,inds] - mnsX[:,i][rem_i][:,newaxis]
                tsX = data0[obs_i,i] - mnsX[:,i][obs_i]
                tsY = data0[rem_i,i] - mnsX[:,i][rem_i]
                W = dot(M, dot(trX, transpose(trY)))
                prY = dot(transpose(W), tsX)
                score += sum((tsY - prY)**2)
            cur_score = score
            #print lmb, cur_score
            if ssft == 0:
                prv_score = cur_score
                prv_lmb = lmb
                lmb = lmb * self.lmb_mult
                ssft = 1
            elif ssft == 1:
                if cur_score < prv_score:
                    ssdir = 'down'
                    prv_lmb = lmb
                    prv_score = cur_score
                    lmb = lmb * self.lmb_mult
                else:
                    ssdir = 'up'
                    lmb = lmb / self.lmb_mult**2
                ssft = 2
            elif ssft == 2:
                if cur_score >= prv_score:
                    lmb = prv_lmb
                    break
                prv_score = cur_score
                prv_lmb = lmb
                if ssdir == 'down':
                    lmb = lmb * self.lmb_mult
                elif ssdir == 'up':
                    lmb = lmb / self.lmb_mult
        s2 = (s**2 + lmb)**(-1)
        XXti = dot(U * s2[newaxis,:], transpose(U))
        XYt = dot(data0[obs_i,:], transpose(data0[rem_i,:]))
        W = dot(XXti, XYt)
        self.prv_lmb = lmb
        mu0 = mean(data[obs_i,:], 1)
        mu1 = mean(data[rem_i,:], 1)
        return transpose(W), mu0, mu1
        
##    def train_probeset(self, data, probeset):
##        (n_genes, n_samples) = data.shape
##        remaining_genes = list(set(range(n_genes)).difference(set(probeset)))
##        remaining_genes.sort()
##        data_selected = data[probeset,:]
##        data_remaining = data[remaining_genes,:]
        
##        U_X, s_X, Vh_X = linalg.svd(data_selected, full_matrices = 0)
##        sumX = sum(data_selected, 1)[:,newaxis] - data_selected
##        sumXrem = sum(data_remaining, 1)[:,newaxis] - data_remaining
##        YXt = dot(data_remaining, transpose(data_selected))
##        vyz = [None] * n_samples
##        decomps = [None] * n_samples
##        for sample_i in xrange(n_samples):
##            inds = range(sample_i) + range(sample_i + 1, n_samples)
##            muX = mean(data_selected[:,inds], 1)
##            trX = data_selected[:,inds] - muX[:,newaxis]
##            tsX = data_selected[:,sample_i] - muX
##            muY = mean(data_remaining[:,inds], 1)
##            trY = data_remaining[:,inds] - muY[:,newaxis]
##            tsY = data_remaining[:,sample_i] - muY
##            U,s,Vh = linalg.svd(trX, full_matrices = 0)
##            M = dot(transpose(U), dot(trX, transpose(trY)) / n_samples)
##            M2 = dot(tsX, U)
##            decomps[sample_i] = (U,s,Vh,trX,tsX,trY,tsY,muX,muY,M,M2)
####            mu_Xmi = sumX[:,sample_i] / (n_samples - 1.0)
####            mu_Ymi = sumXrem[:,sample_i] / (n_samples - 1.0)
####            yrem = data_remaining[:,sample_i]
####            v1 = data_selected[:,sample_i]
####            v2 = sumX[:,sample_i]
####            y = v1 - v2 / (n_samples - 1.0)
####            z1 = dot(v1, U_X)
####            z2 = dot(v2, U_X)
####            z3 = z1 - z2 / (n_samples - 1.0)
####            M0 = YXt - outer(yrem, v1) - outer(sumXrem[:,sample_i], mu_Xmi) - outer(mu_Ymi, sumX[:,sample_i]) + outer(mu_Ymi, mu_Xmi)
####            vyz[sample_i] = (v1,v2,z1,z2,z3,y,yrem)

##        ssft = 0
##        lmb = self.lmb2_0

##        while True:
##            #new_eigs = (s_X**2 + lmb)**(-1.0)
##            score = 0.0
##            for sample_i in xrange(n_samples):
##                (U,s,Vh,trX,tsX,trY,tsY,muX,muY,M,M2) = decomps[sample_i]
##                s2 = (s**2 / n_samples + lmb)**(-1.0)
##                W = dot(U * s2[newaxis,:], M)
##                pr = dot(M2, W)
##                score += sum((pr - tsY)**2)
####                (v1,v2,z1,z2,z3,y,yrem) = vyz[sample_i]
####                w1 = dot(U_X, new_eigs * z1)
####                w2 = dot(U_X, new_eigs * z2) + w1 * sum(w1 * v2) / (1.0 - sum(w1 * v1))
####                XXt_i_y = dot(U_X, new_eigs * z3) + w1 * sum(w1 * y) / (1.0 - sum(w1 * v1)) + w2 * sum(w2 * y) / (n_samples - 1.0 - sum(w2 * v2))
####                pr = dot(M0, XXt_i_y)
####                score += sum((pr - yrem)**2)
##            cur_score = score
##            print(lmb, cur_score)
##            if ssft == 0:
##                prv_score = cur_score
##                prv_lmb = lmb
##                lmb = lmb * self.lmb2_mult
##                ssft = 1
##            elif ssft == 1:
##                if cur_score < prv_score:
##                    ssdir = 'down'
##                    prv_lmb = lmb
##                    prv_score = cur_score
##                    lmb = lmb * self.lmb2_mult
##                else:
##                    ssdir = 'up'
##                    lmb = lmb / self.lmb2_mult**2
##                ssft = 2
##            elif ssft == 2:
##                if cur_score >= prv_score:
##                    lmb = prv_lmb
##                    break
##                prv_score = cur_score
##                prv_lmb = lmb
##                if ssdir == 'down':
##                    lmb = lmb * self.lmb2_mult
##                elif ssdir == 'up':
##                    lmb = lmb / self.lmb2_mult
##        print("Chosen lambda =",lmb)
##        self.lmb = lmb
##        data0_selected = data_selected - mean(data_selected, 1)[:,newaxis]
##        data0_remaining = data_remaining - mean(data_remaining, 1)[:,newaxis]
##        U, s, Vh = linalg.svd(data0_selected, full_matrices = 0)
##        new_eigs = (s**2 + lmb)**(-1.0)
##        W0 = dot(U * new_eigs[newaxis,:], transpose(U))
##        Wt = dot(W0, dot(data0_selected, transpose(data0_remaining)))
##        W = transpose(Wt)
##        mu0 = mean(data_selected, 1)
##        mu1 = mean(data_remaining, 1)
##        return W, mu0, mu1

    def impute(self, data, params):
        W, mu0, mu1 = params
        return dot(W, data - mu0[:,newaxis]) + mu1[:,newaxis]


