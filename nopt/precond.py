"""

 Preconditioners

"""

import numpy as np
from scipy.sparse import diags, csr_matrix, linalg
import jax.numpy as jnp
from jax.ops import index, index_update, index_add
from nopt import LinearOperator


def P1(problem, x, *args):
    """Constraint preconditioner

    M = inv([[I, B^T],
             [B,  0]])
    
    See: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.140.1151&rep=rep1&type=pdf
    """

    def SAINV(B):
        """
        Computes inv(A), where A = BB^T
        """

        Bt = B.T
        
        m = B.shape[0]
        Z = jnp.eye(m)
        p = jnp.zeros(m)

        def np_matvec(v):
            return np.array(Bt @ v)

        for i in range(m):
            for j in range(i, m):
                #res = np_matvec(Z[:,i]).dot(np_matvec(Z[:,j]))
                #p[j] = res
                #p[j] = res
                p = index_update(p, j, (Bt @ Z[:,i]).dot(Bt @ Z[:,j]))
            
            for j in range(i+1,m):
                Z = index_add(Z, index[:,j], -p[j]/p[i]*Z[:,i])
                #Z[:,j] -= p[j]/p[i]*Z[:,i]

            #Z = index_update(Z, Z < 1e-5, 0)

        #Z = csr_matrix(Z)
        #Dinv = diags(1/p)
        Dinv = jnp.diag(1/p)
        
                                 
        def matvec(v):
            return Z @ (Dinv @ (Z.T @ v))

        return LinearOperator((m, m), matvec=matvec)
    
    n, m = problem.nvars, problem.nconstraints
    shape = (m+n, m+n)
    
    G = problem.G(x)
    
    def matvec1(v):
        v1, v2 = v[:n], v[n:]
        res1 = v1 - G.T @ v2
        res2 = v2
        return jnp.concatenate([res1, res2])
    
    def matvec2(v):
        v1, v2 = v[:n], v[n:]
        res1 = v1
        res2 = -SAINV(G) @ v2
        return jnp.concatenate([res1, res2])
    
    def matvec3(v):
        v1, v2 = v[:n], v[n:]
        res1 = v1
        res2 = -G @ v1 + v2
        return jnp.concatenate([res1, res2])
    
    R = LinearOperator(shape, matvec=matvec1, dtype=jnp.float32)
    D = LinearOperator(shape, matvec=matvec2, dtype=jnp.float32)
    L = LinearOperator(shape, matvec=matvec3, dtype=jnp.float32)
        
    def matvec(v):
        return R @ (D @ (L @ v))

    return LinearOperator(shape, matvec=matvec, dtype=jnp.float32)
        
            
        
        
