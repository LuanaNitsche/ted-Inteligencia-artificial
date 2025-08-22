"""
Descrição: o agente agora é atualizado com uma memória (um estado interno), que inicia zerada. Ele deve manter um "mapa" de quais células já visitou.

Inicialmente, o ambiente é livre. Depois, introduzimos obstáculos estáticos em posições pré-definidas.

· Objetivo do Agente: visitar o maior número possível de células do grid, evitando repetir células já visitadas e contornando obstáculos (quando presentes).

Obstáculos: introduzir obstáculos (exemplo abaixo):

· Métricas de Avaliação:

Completude da Exploração: qual a porcentagem de células acessíveis que o agente visitou? (Meta: 100%).

Eficiência de Exploração: qual o número de passos redundantes (visitas a uma mesma célula mais de uma vez)? (Meta: o mais próximo de zero possível).

Sucesso no Desvio: o agente consegue explorar todas as áreas acessíveis mesmo na presença de obstáculos complexos? (Sim/Não).

Generalização do agente: como ele se comportaria com obstáculos diferentes?
"""
import random
from typing import List

from apoio import Obstaculos


class AgenteReativoSimples:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.position = self.random_initial_position()
        self.perimeter_detected = set()
        self.visited = set()
        self.visited.add(self.position) 
        self.caminho_percorrido:List[tuple[int, int]] = []
        self.revisitou:int = 0
        self.directions = Obstaculos.DIRECTIONS 
        self.obstaculos = Obstaculos.OBSTACULOS 

    def random_initial_position(self):
        x = random.randint(0, self.grid_size - 1)
        y = random.randint(0, self.grid_size - 1)
        print(f"Posição inicial aleatória: ({x}, {y})")
        return (x, y)

    def move(self, direction):
        x, y = self.position
        coor_movimento = None

        if direction == 'N' and y < self.grid_size - 1: coor_movimento = (x, y + 1)
        elif direction == 'S' and y > 0: coor_movimento = (x, y - 1)
        elif direction == 'E' and x < self.grid_size - 1: coor_movimento = (x + 1, y)
        elif direction == 'O' and x > 0: coor_movimento = (x - 1, y)
        
        if coor_movimento is None:
            print("Movimento inválido")
            return False
        
        if coor_movimento in self.obstaculos:
            print("Movimento bloqueado por obstáculo")
            return False
        
        if coor_movimento not in self.visited:
            self.caminho_percorrido.append(self.position)
        else:
            self.revisitou +=1

        self.position = coor_movimento
        self.visited.add(self.position)
        print(self.position)

        return True

    def vizinhos_livres(self, x, y):
        deslocamentos = {'N': (0, 1), 'S': (0, -1), 'E': (1, 0), 'O': (-1, 0)}
        for direcao, (desloc_x, desloc_y) in deslocamentos.items():
            novo_x, novo_y = x + desloc_x, y + desloc_y

            dentro_dos_limites = 0 <= novo_x < self.grid_size and 0 <= novo_y < self.grid_size
            
            if dentro_dos_limites:
                nova_posicao = (novo_x, novo_y)
                
                if nova_posicao not in self.obstaculos:
                    yield direcao, nova_posicao

    def proximo_na_exploracao(self):
        x_atual, y_atual = self.position

        for direcao, (x_novo, y_novo) in self.vizinhos_livres(x_atual, y_atual):
            if (x_novo, y_novo) not in self.visited:
                return ('avanco', direcao)

        if getattr(self, 'caminho_percorrido', []):
            x_ant, y_ant = self.caminho_percorrido.pop()

            if x_ant == x_atual - 1 and y_ant == y_atual:
                return ('retrocesso', 'O')  
            if x_ant == x_atual + 1 and y_ant == y_atual:
                return ('retrocesso', 'E')  
            if y_ant == y_atual - 1 and x_ant == x_atual:
                return ('retrocesso', 'S')  
            if y_ant == y_atual + 1 and x_ant == x_atual:
                return ('retrocesso', 'N')  

        return (None, None) 

    def perceive(self):
        x, y = self.position
        if x == 0:
            self.perimeter_detected.add('O')
        if x == self.grid_size - 1:
            self.perimeter_detected.add('E')
        if y == 0:
            self.perimeter_detected.add('S')
        if y == self.grid_size - 1:
            self.perimeter_detected.add('N')

    def has_detected_perimeter(self):
        return len(self.perimeter_detected) == 4  

    def explore(self):
        while True:
            modo, direcao = self.proximo_na_exploracao()
            if modo is None:
                print("Exploração completa ou sem movimentos válidos.")
                break
            
            self.move(direcao)
            self.perceive()

        return self.visited


if __name__ == "__main__":
    print("Iniciando exploração do grid...")
    agente = AgenteReativoSimples(grid_size=10) #alterar para aleatorio - nao necessariamente vai se quadrado
    visitadas = agente.explore()

    print("Células visitadas:", len(visitadas))
    print("Perímetro detectado:", agente.perimeter_detected)
    print("Detecção completa do perímetro?:", agente.has_detected_perimeter())
    print("Passos redundantes (revisitas):", agente.revisitou)
        