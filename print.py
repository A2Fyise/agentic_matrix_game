import numpy as np

def print_grid(self):
    grid = np.full((self.size, self.size), "*", dtype=str)
    ax, ay = self.agent_location
    tx, ty = self.target_location

    if np.array_equal(self.agent_location, self.target_location):
        grid[ax, ay] = "G"
    else:    
        grid[ax, ay] = "A"  
        grid[tx, ty] = "T" 

    print("\n".join(" ".join(row) for row in grid))
    print("\n" + "="*20 + "\n")
