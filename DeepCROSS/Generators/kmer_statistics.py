import collections
import numpy as np

class kmer_statistics(object):
    def __init__(self, n, samples):
        self._n = n
        self._samples = samples
        self._kmer_counts = collections.defaultdict(int)
        self._total_kmers = 0
        for kmers in self.kmers():
            self._kmer_counts[kmers] += 1
            self._total_kmers += 1

    def kmers(self):
        n = self._n
        for sample in self._samples:
            for i in range(len(sample)-n+1):
                yield sample[i:i+n]

    def unique_kmers(self):
        return set(self._kmer_counts.keys())

    def log_likelihood(self, kmer):
        if kmer not in self._kmer_counts:
            return -np.inf
        else:
            return np.log(self._kmer_counts[kmer]) - np.log(self._total_kmers)

    def kl_to(self, p):
        # p is another kmer_statistics
        log_likelihood_ratios = []
        for kmer in p.kmers():
            log_likelihood_ratios.append(p.log_likelihood(kmer) - self.log_likelihood(kmer))
        return np.mean(log_likelihood_ratios)

    def cosine_sim_with(self, p):
        p_dot_q = 0.
        p_norm = 0.
        q_norm = 0.
        for kmer in p.unique_kmers():
            p_i = np.exp(p.log_likelihood(kmer))
            q_i = np.exp(self.log_likelihood(kmer))
            p_dot_q += p_i * q_i
            p_norm += p_i**2
        for kmer in self.unique_kmers():
            q_i = np.exp(self.log_likelihood(kmer))
            q_norm += q_i**2
        return p_dot_q / (np.sqrt(p_norm) * np.sqrt(q_norm))

    def precision_wrt(self, p):
        num = 0.
        denom = 0
        p_kmers = p.unique_kmers()
        for kmer in self.unique_kmers():
            if kmer in p_kmers:
                num += self._kmer_counts[kmer]
            denom += self._kmer_counts[kmer]
        return float(num) / denom

    def recall_wrt(self, p):
        return p.precision_wrt(self)

    def js_with(self, p):
        log_p = np.array([p.log_likelihood(kmer) for kmer in p.unique_kmers()])
        log_q = np.array([self.log_likelihood(kmer) for kmer in p.unique_kmers()])
        log_m = np.logaddexp(log_p - np.log(2), log_q - np.log(2))
        kl_p_m = np.sum(np.exp(log_p) * (log_p - log_m))

        log_p = np.array([p.log_likelihood(kmer) for kmer in self.unique_kmers()])
        log_q = np.array([self.log_likelihood(kmer) for kmer in self.unique_kmers()])
        log_m = np.logaddexp(log_p - np.log(2), log_q - np.log(2))
        kl_q_m = np.sum(np.exp(log_q) * (log_q - log_m))

        return 0.5*(kl_p_m + kl_q_m) / np.log(2)
    
    def R_with(self, p):
        X = np.array([p._kmer_counts[kmer]/p._total_kmers for kmer in p.unique_kmers()])
        Y = np.array([self._kmer_counts[kmer]/self._total_kmers for kmer in self.unique_kmers()])
        XY = np.array([self._kmer_counts[kmer]*p._kmer_counts[kmer]/p._total_kmers/self._total_kmers for kmer in p.unique_kmers()])
        EX = 1/np.power(4,self._n)
        EY = EX
        