

import numpy as np
import sympy as sp
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation



x, y, t = sp.symbols('x, y, t')

class Wave2D:
    def __init__(self, L=1.0):
        """Initialize with a default domain size L."""
        self.L = L  
        self.U = None
        self.Um1 = None


    def create_mesh(self, N):
        """Create 2D mesh and store in self.xij and self.yij."""

        self.N = N
        self.h = self.L / self.N  # Grid spacing
        self.x = np.linspace(0, self.L, self.N + 1)
        self.y = np.linspace(0, self.L, self.N + 1)
        self.xij, self.yij = np.meshgrid(self.x, self.y, indexing='ij')  # Meshgrid
        return self.xij, self.yij

    def D2(self, N):
        """Return second order differentiation matrix"""
        # See lecture notes week 5
        D2 = sparse.diags([np.ones(self.N), np.full(self.N+1, -2), np.ones(self.N)], np.array([-1, 0, 1]), (self.N+1, self.N+1), 'lil')
        D2.toarray()
        D2[0, :4] = 2, -5, 4, -1
        D2[-1, -4:] = -1, 4, -5, 2
        return D2

    @property
    def w(self):
        """Return the dispersion coefficient."""

        return self.c * sp.pi * sp.sqrt(self.mx**2 + self.my**2)


    def ue(self, mx, my):
        """Return the exact standing wave"""
        return sp.sin(mx*sp.pi*x)*sp.sin(my*sp.pi*y)*sp.cos(self.w*t)

    def initialize(self, N, mx, my):
        r"""Initialize the solution at $U^{n}$ and $U^{n-1}$

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        mx, my : int
            Parameters for the standing wave
        """
        self.create_mesh(N)
        self.Un = np.zeros((N + 1, N + 1))
        self.Um1 = np.zeros((N + 1, N + 1))
        ue = self.ue(mx, my)
        ue_func = sp.lambdify((x, y, t), ue, 'numpy')
        xij, yij = self.xij, self.yij
        self.Um1[:] = ue_func(xij, yij, 0)

        self.D = self.D2(N)/self.h**2
        self.Un[:] = self.Um1[:] + 0.5*(self.c*self.dt)**2*(self.D @ self.Um1 + self.Um1 @ self.D.T)

        return self.Un, self.Um1
    


    @property
    def dt(self):
        """Return the time step."""
        return self.cfl * self.h / self.c

    def l2_error(self, u, t0):
        """Return l2-error norm

        Parameters
        ----------
        u : array
            The solution mesh function
        t0 : number
            The time of the comparison
        """
        ue = self.ue(self.mx, self.my)
        ue_func = sp.lambdify((x, y, t), ue, 'numpy')
        ue_exact = ue_func(self.xij, self.yij, t0)
        return np.sqrt((self.h * self.h) * np.sum((ue_exact - u)**2))

    def apply_bcs(self):
        """Applying Dirichlet boundary conditions."""
     
        self.Unp1[0, :] = 0  # Bottom boundary
        self.Unp1[-1, :] = 0  # Top boundary
        self.Unp1[:, 0] = 0  # Left boundary
        self.Unp1[:, -1] = 0  # Right boundary
        return self.Unp1
    
    def __call__(self, N, Nt, cfl=0.5, c=1.0, mx=3, my=3, store_data=-1):
        """Solve the 2D wave equation with explicit 2nd-order time stepping."""
        self.cfl, self.c, self.mx, self.my = cfl, c, mx, my
        self.initialize(N, mx, my)   # sets self.Um1 = U^0, self.Un = U^1

        dt = self.dt
        self.Unp1 = np.zeros_like(self.Un)
        results = {} if store_data > 0 else None
        errors = []

        # time-stepping: start at step index = 1 (we have U^0 (Um1) and U^1 (Un))
        for n in range(1, Nt):
            # compute interior update for U^{n+1}
            self.Unp1[:] = 2.0 * self.Un - self.Um1 + (self.c * dt)**2 * (self.D @ self.Un + self.Un @ self.D.T)

            # apply Dirichlet BCs at this new time level
            self.apply_bcs()

            # store intermediate solution if requested
            if store_data > 0 and n % store_data == 0:
                results[n] = self.Unp1.copy()

            # update error list (if asked to collect errors at each time step)
            if store_data == -1:
                # physical time of current U^{n+1} is (n+1)*dt; adjust if you want
                errors.append(self.l2_error(self.Unp1, (n+1) * dt))

            # shift time levels: Um1 <- Un, Un <- Unp1 (copy to avoid aliasing)
            self.Um1, self.Un = self.Un, self.Unp1.copy()

        # After stepping end: return final h and errors (if store_data == -1)
        return (self.h, errors) if store_data == -1 else results

    def convergence_rates(self, m=4, cfl=0.1, Nt=10, mx=3, my=3):
        """Compute convergence rates for varying discretizations."""
        E = []
        h = []
        N0 = 8  # Start with 8 intervals

        for _ in range(m):
            dx, err = self(N0, Nt, cfl=cfl, mx=mx, my=my, store_data=-1)
            E.append(err[-1])
            h.append(dx)
            N0 *= 2  # Refine spatial mesh
            Nt *= 2  # Refine temporal resolution

        r = [np.log(E[i - 1] / E[i]) / np.log(h[i - 1] / h[i]) for i in range(1, m)]
        return r, np.array(E), np.array(h)

class Wave2D_Neumann(Wave2D):

    def D2(self, N):
        """Return second order differentiation matrix with Neumann BCs"""
        D2 = sparse.diags([np.ones(self.N), np.full(self.N+1, -2), np.ones(self.N)], np.array([-1, 0, 1]), (self.N+1, self.N+1), 'lil')
        D2.toarray()
        D2[0, :3] = -2, 2, 0
        D2[-1, -3:] = 0, 2, -2
        return D2


    def ue(self, mx, my):
        """Return the exact standing wave with Neumann BCs."""
        return sp.cos(mx * sp.pi * x) * sp.cos(my * sp.pi * y) * sp.cos(self.w * t)
        

    def apply_bcs(self):
        """Apply Neumann boundary conditions (do nothing)."""
        return self.Unp1

def test_convergence_wave2d():
    sol = Wave2D(L=1.0)  # Central initialization of domain size L=1
    r, E, h = sol.convergence_rates(mx=2, my=3)  # mx and my for exact solution
    
    print("Convergence rates:", r)
    assert abs(r[-1] - 2) < 1e-2  # Expect second-order accuracy

def test_convergence_wave2d_neumann():
    solN = Wave2D_Neumann()
    r, E, h = solN.convergence_rates(mx=2, my=3)
    print("Convergence rates:", r)
    assert abs(r[-1]-2) < 0.05

def test_exact_wave2d():
    # find the errors when mx=my and cfl= 1/sqrt(2)
    mx = my = 1
    cfl = 1 / np.sqrt(2)
    sol = Wave2D()
    r, E, h = sol.convergence_rates(mx=mx,my=my,cfl=cfl)
    print(E)
    solN = Wave2D_Neumann()
    rN, EN, hN = solN.convergence_rates(mx=mx,my=my,cfl=cfl)
    print(EN)
    assert EN[-1] < 1e-10
    assert E[-1] < 1e-10
  

def make_animation():
    from IPython.display import HTML
    from IPython.display import display
    solN = Wave2D_Neumann()
    results = solN(50, 50, cfl=1/sp.sqrt(2), mx=2, my=2, store_data=5)
    fig = plt.figure()

    #also get out xij and yij
    xij, yij = solN.xij, solN.yij
    data = list(results.values())

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(xij, yij, data[0], cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
    


    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    frames = []
    for n, val in results.items():
        frame = ax.plot_wireframe(xij, yij, val, rstride=2, cstride=2);
        #frame = ax.plot_surface(xij, yij, val, vmin=-0.5*data[0].max(),
        #                        vmax=data[0].max(), cmap=cm.coolwarm,
        #                        linewidth=0, antialiased=False)
        frames.append([frame])

    ani = animation.ArtistAnimation(fig, frames, interval=400, blit=True,
                                    repeat_delay=1000)
    #save animation as gif in the folder called report
    ani.save('neumannwave.gif', writer='pillow', fps=5)
    return ani
if __name__ == "__main__":
    from IPython.display import HTML
    from IPython.display import display
    # Run the convergence test for Wave2D
    test_convergence_wave2d()
    print("Test passed: Convergence for Wave2D works as expected.")
    test_convergence_wave2d_neumann()
    print("Test passed: Convergence for Wave2D_Neumann works as expected.")
    #test_exact_wave2d()
    #print("Test passed: Exact solution for Wave2D works as expected.")
    ani = make_animation()
    display(HTML(ani.to_jshtml()))
