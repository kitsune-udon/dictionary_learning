import numpy as np

def omp(Y, A, k, stopping_criteria):
    def index_of_max_correlation(A, active_set, r):
        def correlation(v0, v1):
            d = np.linalg.norm(v0) * np.linalg.norm(v1)
            if d == 0.0:
                raise "zero vector exception"
            else:
                return np.dot(v0, v1) / d

        max_v = float("-inf")
        max_i = -1
        for i in xrange(A.shape[1]):
            if i in active_set:
                continue
            v = abs(correlation(A[:,i], r))
            if max_v < v:
                max_i, max_v = i, v
        return max_i

    def subset_of_atoms(A, active_set):
        m, n = A.shape[0], len(active_set)
        A0 = np.zeros((m, n))
        for i in xrange(len(active_set)):
            A0[:, i] = A[:, active_set[i]]
        return A0

    def solve_equation(A, Y):
        return np.dot(np.linalg.pinv(A), Y)

    def adjust_x(n, x, active_set):
        x0 = np.zeros((n,1))
        for i in xrange(len(active_set)):
            j = active_set[i]
            x0[j,0] = x[i]
        return x0

    m, n = A.shape
    r = Y
    active_set = []
    max_active_set_len = min(k, n)

    while len(active_set) < max_active_set_len and not(stopping_criteria(r)):
        i = index_of_max_correlation(A, active_set, r)
        active_set.append(i)
        A0 = subset_of_atoms(A, active_set)
        x = solve_equation(A0, Y)
        r = Y - np.dot(A0, x)

    if len(active_set) > 0:
        A0 = subset_of_atoms(A, active_set)
        x = solve_equation(A0, Y)
        return adjust_x(n, x, active_set)
    else:
        raise "invalid parameter"

def dictionary_learning(Y, iter_num, sparsity, criteria, A0):
    def sparse_coding(Y, A, k, criteria):
        X = np.zeros((A.shape[1], Y.shape[1]))
        for i in xrange(Y.shape[1]):
            crit = criteria(i)(Y[:, i])
            X[:, i] = omp(Y[:, i], A, k, crit).ravel()
        return X

    def update_dictionary(Y, A, X):
        E_pre = Y - np.dot(A, X)
        R = np.zeros(A.shape)
        for j in xrange(A.shape[1]):
            E = E_pre + np.dot(np.asmatrix(A[:,j]).transpose(), np.asmatrix(X[j]))
            U,s,V = np.linalg.svd(E, full_matrices=False)
            atom = np.asarray(U)[:, 0]
            R[:, j] = atom
        return R

    A = np.array(A0)
    for i in xrange(iter_num):
        X = sparse_coding(Y, A, sparsity, criteria)
        A = update_dictionary(Y, A, X)
    return A

def omp_test():
    Y = np.array([2,3,4]).reshape((3,1)) # data
    A = np.array([[1,1,1],[1,-1,2],[1,1,3]]) # bases
    k = 3 # sparsity
    Y_norm = np.linalg.norm(Y)
    stopping_criteria = lambda r: np.linalg.norm(r) <= 0.01 * Y_norm
    X = omp(Y, A, k, stopping_criteria)
    print X
    print np.dot(A, X)

def dictionary_learning_test():
    def make_init_dict(n_dim, n_atoms):
        A = np.random.rand(n_dim, n_atoms) - 0.5
        #A = np.ones((n_dim, n_atoms))
        for i in xrange(n_atoms):
            norm = np.linalg.norm(A[:, i])
            if norm == 0.0:
                raise "invalid atom"
            mult = 1.0 / norm
            A[:, i] = mult * A[:, i]
        return A
    def make_data(n_dim, n_samples):
        Y = np.random.rand(n_dim, n_samples) - 0.5
        return Y
    def display_remain(Y, A):
        for i in xrange(Y.shape[1]):
            x = omp(Y[:,i].reshape((n_dim,1)), A, sparsity, crit(0)(Y[:,i]))
            orig_norm = np.linalg.norm(Y[:,i])
            diff = Y[:,i] - np.dot(A, x).ravel()
            diff_norm = np.linalg.norm(diff)
            print diff_norm / orig_norm
    n_samples = 9
    n_atoms = 7
    n_dim = 5
    n_iter = 10
    sparsity = 2
    A0 = make_init_dict(n_dim, n_atoms)
    Y = make_data(n_dim, n_samples)
    crit = lambda idx: lambda Y: (lambda Y_norm: lambda r: np.linalg.norm(r) <= 0.1 * Y_norm)(np.linalg.norm(Y))
    A = dictionary_learning(Y, n_iter, sparsity, crit, A0)
    display_remain(Y, A)

#omp_test()
np.random.seed(0)
dictionary_learning_test()
