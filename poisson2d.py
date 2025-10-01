import numpy as np
import sympy as sp
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve

x, y = sp.symbols('x,y')

class Poisson2D:
    r"""Solve Poisson's equation in 2D::

        \nabla^2 u(x, y) = f(x, y), in [0, L]^2

    where L is the length of the domain in both x and y directions.
    Dirichlet boundary conditions are used for the entire boundary.
    The Dirichlet values depend on the chosen manufactured solution.

    """

    def __init__(self, L, ue):
        """Initialize Poisson solver for the method of manufactured solutions

        Parameters
        ----------
        L : number
            The length of the domain in both x and y directions
        ue : Sympy function
            The analytical solution used with the method of manufactured solutions.
            ue is used to compute the right hand side function f.
        """
        self.L = L
        self.ue = ue
        self.f = sp.diff(self.ue, x, 2)+sp.diff(self.ue, y, 2)

    def create_mesh(self, N):
        """Create 2D mesh and store in self.xij and self.yij"""
        self.N = N
        self.h = self.L/N
        self.x = np.linspace(0, self.L, N+1)
        self.y = np.linspace(0, self.L, N+1)
        self.xij, self.yij = np.meshgrid(self.x, self.y, indexing='ij')
        return self.xij, self.yij


    def D2(self):
        """Return second order differentiation matrix"""
        D2 = sparse.diags([np.ones(self.N), np.full(self.N+1, -2), np.ones(self.N)], np.array([-1, 0, 1]), (self.N+1, self.N+1), 'lil')
        D2.toarray()
        D2[0, :4] = 2, -5, 4, -1
        D2[-1, -4:] = -1, 4, -5, 2
        return D2
    

    def laplace(self):
        """Return vectorized Laplace operator"""
        D2x = (1./self.h**2)*self.D2()
        D2y = (1./self.h**2)*self.D2()
        A = (sparse.kron(D2x, sparse.eye(self.N+1)) + sparse.kron(sparse.eye(self.N+1), D2y))
        return A

    def get_boundary_indices(self):
        """Return indices of vectorized matrix that belongs to the boundary"""
        B = np.ones((self.N+1, self.N+1), dtype=bool)
        B[1:-1, 1:-1] = 0
        self.bnds = np.where(B.ravel() == 1)[0]
        return self.bnds

    def assemble(self):
        """Return assembled matrix A and right-hand side vector b."""
        # Evaluate source term and exact boundary

        F = sp.lambdify((x, y), self.f)(self.xij, self.yij)

        A = self.laplace()
        A = A.tolil()
        for i in self.bnds:
            A[i] = 0
            A[i, i] = 1
        A = A.tocsr()

        b = F.ravel()
        b[self.bnds] = sp.lambdify((x, y), self.ue)(self.xij, self.yij).ravel()[self.bnds]

        return A, b



    def l2_error(self, u):
        """Return l2-error norm"""
 
        ue_func = sp.lambdify((x, y), self.ue, 'numpy')
        ue_exact = ue_func(self.xij, self.yij)
        return np.sqrt((self.h*self.h) * np.sum((ue_exact - u) ** 2))

    
    def __call__(self, N):
        """Solve Poisson's equation.

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction

        Returns
        -------
        The solution as a Numpy array

        """
        self.create_mesh(N)
        self.get_boundary_indices()
        A, b = self.assemble()         
        self.U = sparse.linalg.spsolve(A, b.flatten()).reshape((N+1, N+1))
    
        return self.U


    def convergence_rates(self, m=6):
        E = []
        h = []
        N0 = 8
        for m in range(m):
            u = self(N0)
            E.append(self.l2_error(u))
            h.append(self.h)
            N0 *= 2
        r = [np.log(E[i-1]/E[i])/np.log(h[i-1]/h[i]) for i in range(1, m+1, 1)]
        return r, np.array(E), np.array(h)

    

    def eval(self, x, y):
        """Return u(x, y)

        Parameters
        ----------
        x, y : numbers
            The coordinates for evaluation

        Returns
        -------
        The value of u(x, y)

        """
        # create a function for interpolation, consider four nearest points 
        if x<0 or x>self.L or y<0 or y>self.L:
            raise ValueError("x and y must be in [0, L]")
        i = int(x/self.h)
        j = int(y/self.h)

        if i==self.N: i -= 1
        if j==self.N: j -= 1

        x1 = self.x[i]
        x2 = self.x[i+1]
        y1 = self.y[j]
        y2 = self.y[j+1]
        Q11 = self.U[i, j]
        Q12 = self.U[i, j+1]
        Q21 = self.U[i+1, j]
        Q22 = self.U[i+1, j+1]
        return (Q11 * (x2 - x) * (y2 - y) +
                Q21 * (x - x1) * (y2 - y) +
                Q12 * (x2 - x) * (y - y1) +
                Q22 * (x - x1) * (y - y1)) / ((x2 - x1) * (y2 - y1))


def test_convergence_poisson2d():
    # This exact solution is NOT zero on the entire boundary
    ue = sp.exp(sp.cos(4*sp.pi*x)*sp.sin(2*sp.pi*y))
    sol = Poisson2D(1, ue)
    r, E, h = sol.convergence_rates()
    assert abs(r[-1]-2) < 1e-2

def test_interpolation():
    ue = sp.exp(sp.cos(4*sp.pi*x)*sp.sin(2*sp.pi*y))
    sol = Poisson2D(1, ue)
    U = sol(100)
    assert abs(sol.eval(0.52, 0.63) - ue.subs({x: 0.52, y: 0.63}).n()) < 1e-3
    assert abs(sol.eval(sol.h/2, 1-sol.h/2) - ue.subs({x: sol.h/2, y: 1-sol.h/2}).n()) < 1e-3

if __name__ == '__main__':
    try:
        print("Running test for convergence:")
        test_convergence_poisson2d()
        print("Convergence test passed!")
    except AssertionError as e:
        print("Convergence test failed.")
        raise e
    try:
        print("Running test for interpolation:")
        test_interpolation()
        print("Interpolation test passed!")
    except AssertionError as e:
        print("Interpolation test failed.")
        raise e