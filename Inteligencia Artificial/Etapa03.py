import random
from collections import deque

class AgenteCaminho:
    def __init__(self, grid_size, obstacle_ratio=0.2):
        self.grid_size = grid_size
        self.grid = self.generate_grid(obstacle_ratio)
        self.start = self.random_position()
        self.end = self.random_position()
        while self.start == self.end:
            self.end = self.random_position()

    def generate_grid(self, obstacle_ratio):
        grid = [[0 for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        num_obstacles = int(self.grid_size * self.grid_size * obstacle_ratio)
        for _ in range(num_obstacles):
            x, y = random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1)
            grid[y][x] = 1  
        return grid

    def random_position(self):
        return (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1))

    def is_valid(self, x, y):
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size and self.grid[y][x] == 0

    def find_path(self):
        directions = [(0,1), (0,-1), (1,0), (-1,0)]  # N, S, E, O
        queue = deque([(self.start, [self.start])])
        visited = set()
        visited.add(self.start)

        while queue:
            (x, y), path = queue.popleft()
            if (x, y) == self.end:
                return path
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if self.is_valid(nx, ny) and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append(((nx, ny), path + [(nx, ny)]))
        return None  # sem caminho possível

    def print_grid(self, path=None):
        for y in range(self.grid_size):
            row = ""
            for x in range(self.grid_size):
                if path and (x, y) in path:
                    if (x, y) == self.start:
                        row += "S "
                    elif (x, y) == self.end:
                        row += "F "
                    else:
                        row += "* "
                elif self.grid[y][x] == 1:
                    row += "X "
                else:
                    row += ". "
            print(row)
        print()


if __name__ == "__main__":
    print("Ambiente Livre (sem obstáculos):")
    agente_livre = AgenteCaminho(grid_size=10, obstacle_ratio=0)
    path_livre = agente_livre.find_path()
    if path_livre:
        path_livre = path_livre[::-1]  
    agente_livre.print_grid(path_livre)
    print("Caminho encontrado:", path_livre)

    print("Ambiente com Obstáculos:")
    agente_obstaculos = AgenteCaminho(grid_size=10, obstacle_ratio=0.2)
    path_obstaculos = agente_obstaculos.find_path()
    if path_obstaculos:
        path_obstaculos = path_obstaculos[::-1] 
    agente_obstaculos.print_grid(path_obstaculos)
    print("Caminho encontrado:", path_obstaculos)
