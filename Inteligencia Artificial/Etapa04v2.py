"""
Etapa 1: Agente Reativo Simples

· Descrição: nesta fase, o agente não tem memória (estado interno) das posições visitadas. Sua decisão de movimento é baseada apenas na sua percepção atual (sua posição e se há uma parede nos limites do grid). O ambiente é um grid vazio, sem obstáculos.

· Objetivo do Agente: o objetivo do agente é explorar o ambiente até ter colidido com as quatro paredes limites (norte, sul, leste e oeste).

· Métricas de Avaliação:

Detecção Completa do Perímetro: o agente conseguiu determinar corretamente os limites do grid (sim/não).
"""
import random


class AgenteReativoSimples:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.position = (0, 5)
        self.finish = (9, 5)
        self.cost_matrix = self.random_cost_matrix()
        self.print_cost_matrix(self.cost_matrix)
        self.total_cost = 0

    def random_cost_matrix(self):
        cost_matrix = [
            [random.choices([1, 2, 3], weights=[3, 2, 1], k=1)[0] for j in range(10)]
            for i in range(10)
        ]
        cost_matrix[self.finish[0]][self.finish[1]] = 0
        cost_matrix[self.position[0]][self.position[1]] = 0
        return cost_matrix

    @staticmethod
    def print_cost_matrix(matrix):
        print("======================================")
        for row in matrix:
            print(" ".join(str(x) for x in row))

    def explore(self):

        while not self.has_reached_end():

            neighbors_cost = self.get_neighbor_costs()
            direction = min(neighbors_cost, key=neighbors_cost.get)
            self.move(direction)
            x, y = self.position
            self.total_cost += self.cost_matrix[x][y]
            self.cost_matrix[x][y] = 0
            self.print_cost_matrix(self.cost_matrix)
            print("Moveus para a direção", direction, "\nCusto total: ", self.total_cost)

        print("Chegou no fim!")

    def has_reached_end(self):
        return self.position == self.finish

    def get_neighbor_costs(self):
        """
        Retorna um dicionário com as coordenadas e os custos calculados de cada direção possível
        :return:
        """
        x, y = self.position
        neighbors = {}
        n = len(self.cost_matrix)  # number of rows
        m = len(self.cost_matrix[0])  # number of cols

        try:
            if y < 5:
                if y - 1 >= 0:
                    neighbors['W'] = self.cost_matrix[x][y - 1] + 1
                if y + 1 < n:
                    neighbors['E'] = self.cost_matrix[x][y + 1]
            elif y > 5:
                if y - 1 >= 0:
                    neighbors['W'] = self.cost_matrix[x][y - 1]
                if y + 1 < n:
                    neighbors['E'] = self.cost_matrix[x][y + 1] + 1
            else:
                if y - 1 >= 0:
                    neighbors['W'] = self.cost_matrix[x][y - 1] + 1
                if y + 1 < n:
                    neighbors['E'] = self.cost_matrix[x][y + 1] + 1

            if x + 1 >= 0:
                neighbors['S'] = self.cost_matrix[x + 1][y]

        except IndexError:
            pass

        # Removendo células visitadas
        if self.cost_matrix[x][y - 1] == 0:
            neighbors.pop('W')
        elif self.cost_matrix[x][y + 1] == 0:
            neighbors.pop('E')

        neighbors = self.draw_cost(neighbors)

        return neighbors

    def draw_cost(self, neighbors):
        x, y = self.position
        x_finish, y_finish = self.finish

        x_dist = abs(x - x_finish)
        y_dist = abs(y - y_finish)

        adjusted = neighbors.copy()

        if y_dist > x_dist:
            # Favorecendo WEST/EAST
            for d in ("E", "W"):
                if d in adjusted:
                    adjusted[d] -= 0.5
        elif x_dist > y_dist:
            # Favorecendo SOUTH
            if "S" in adjusted:
                adjusted["S"] -= 0.5

        return adjusted

    def move(self, direction):
        x, y = self.position

        if direction == "E":
            self.position = (x, y + 1)
        elif direction == "W":
            self.position = (x, y - 1)
        else:
            self.position = (x + 1, y)


if __name__ == "__main__":
    print("Iniciando exploração do grid...")
    agente = AgenteReativoSimples(grid_size=10)
    agente.explore()

        