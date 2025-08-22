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
        self.position = self.random_initial_position()
        self.perimeter_detected = set() 

    def random_initial_position(self):
        x = random.randint(0, self.grid_size - 1)
        y = random.randint(0, self.grid_size - 1)
        print(f"Posição inicial aleatória: ({x}, {y})")
        return (x, y)

    def move(self, direction):
        x, y = self.position
        if direction == 'N' and y < self.grid_size - 1:  
            self.position = (x, y + 1)
        elif direction == 'S' and y > 0:  
            self.position = (x, y - 1)
        elif direction == 'E' and x < self.grid_size - 1:  
            self.position = (x + 1, y)
        elif direction == 'O' and x > 0:  
            self.position = (x - 1, y)
        print(self.position)

    def perceive(self):
        x, y = self.position
        if x == 0:
            self.perimeter_detected.add('O')
            print("Cheguei no limite oeste!")
        if x == self.grid_size - 1:
            self.perimeter_detected.add('E')
            print("Cheguei no limite leste!")
        if y == 0:
            self.perimeter_detected.add('S')
            print("Cheguei no limite sul!")
        if y == self.grid_size - 1:
            self.perimeter_detected.add('N')
            print("Cheguei no limite norte!")

    def has_detected_perimeter(self):
        return len(self.perimeter_detected) == 4
    
    def explore(self):
        directions = ['N', 'S', 'E', 'O']
        while not self.has_detected_perimeter():
            for direction in directions:
                while direction not in self.perimeter_detected:
                    self.move(direction)
                    self.perceive()
                
                if self.has_detected_perimeter():
                    break

        return self.perimeter_detected
        

if __name__ == "__main__":
    print("Iniciando exploração do grid...")
    agente = AgenteReativoSimples(grid_size=10)
    resultado = agente.explore()
    print("Perímetro detectado:", resultado)
    print("Detecção completa do perímetro:", agente.has_detected_perimeter())

        