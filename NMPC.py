import numpy as np 
import casadi as ca
import matplotlib.pyplot as plt

class NMPC:
    def __init__(self, x_goal, x_start, horizon, dt, obstacles, radius, x_range):
        """
        Parameters:
          x_goal   : Goal
          x_start  : Initial
          horizon  : Prediction horizon
          dt       : Discrete time step
          obstacles: Obstacle centers
          radius   : Obstacle radii
          x_range  : Bounds
        """
        self.x_goal = np.array(x_goal)
        self.x_start = np.array(x_start)
        self.horizon = horizon
        self.dt = dt
        self.state_dim = 2
        self.obstacles = [np.array(obs) for obs in obstacles]
        self.radius = radius
        self.range = x_range  
    
    def optimize(self):

        X = ca.SX.sym('X', self.state_dim, self.horizon + 1)  
        U = ca.SX.sym('U', self.state_dim, self.horizon)     
        
        # Construct cost function
        cost = 0
        for k in range(self.horizon):
            cost += ca.sumsqr(X[:, k] - self.x_goal) + ca.sumsqr(U[:, k])
        # Terminal cost
        cost += ca.sumsqr(X[:, self.horizon] - self.x_goal)
        
        # Constraint list and their lower/upper bounds
        g = []
        lbg = []
        ubg = []
        
        # Initial state constraint
        g.append(X[:, 0] - self.x_start)
        lbg.extend([0] * self.state_dim)
        ubg.extend([0] * self.state_dim)
        
        # Dynamic and obstacle avoidance constraints
        for k in range(self.horizon):
            dyn = X[:, k + 1] - (X[:, k] + self.dt * (0.05 * (X[:, k] - self.x_goal) + U[:, k]))
            g.append(dyn)
            lbg.extend([0] * self.state_dim)
            ubg.extend([0] * self.state_dim)
            

            for center, rad in zip(self.obstacles, self.radius):
                obs_constr = ca.sumsqr(X[:, k] - center) - (rad + 0.3)**2
                g.append(obs_constr)
                lbg.append(0)        
                ubg.append(ca.inf)    
                
        opt_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        
        x_max = self.range[0]
        x_min = self.range[1]
        nX = int(X.size1() * X.size2())  
        nU = int(U.size1() * U.size2())  
        
        lbx = []
        ubx = []
       
        for i in range(nX):
            lbx.append(x_min)
            ubx.append(x_max)
      
        for i in range(nU):
            lbx.append(-ca.inf)
            ubx.append(ca.inf)
            
        # Define the nonlinear optimization problem
        nlp = {'x': opt_vars, 'f': cost, 'g': ca.vertcat(*g)}
        opts = {'ipopt.print_level': 0, 'print_time': 0}
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        

        x0 = np.zeros((int(opt_vars.size()[0]),))
        

        sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        sol_x = sol['x'].full().flatten()
        
        U_opt = sol_x[nX:]
        
        first_u = U_opt[:self.state_dim]
        return first_u

if __name__ == "__main__":

    x_goal = [0, 0]             # Goal
    x_start = [4.5, 4.5]        # Start
    horizon = 50                # Prediction horizon
    dt = 0.05                   # Time step
    obstacles = [[2, 2], [3, 3]] # Obstacle centers
    radius = [0.5, 0.5]         # Obstacle radii
    x_range = [20, -20]         # bounds


    tol = 0.1                   # Termination tolerance
    max_iters = 100             # Maximum iter
    

    current_state = np.array(x_start)
    trajectory = [current_state.copy()]
    
    for i in range(max_iters):
        nmpc_controller = NMPC(x_goal, current_state, horizon, dt, obstacles, radius, x_range)
        u = nmpc_controller.optimize()
        next_state = current_state + dt * (0.05 * (current_state - np.array(x_goal)) + u)
        trajectory.append(next_state.copy())
        current_state = next_state
        
        if np.linalg.norm(current_state - np.array(x_goal)) < tol:
            print(f"Goal reached in {i+1} iterations.")
            break
    else:
        print("Reached maximum iterations without reaching goal.")
    
    trajectory = np.array(trajectory)
    
    plt.plot(trajectory[:, 0], trajectory[:, 1])
    plt.plot(x_start[0], x_start[1], 'go')  
    plt.plot(x_goal[0], x_goal[1], 'go')  
    
    for center, r in zip(obstacles, radius):
        circle = plt.Circle(center, r, color='b', fill=False)
        plt.gca().add_artist(circle)
    
    plt.axis('equal')
    plt.show()
