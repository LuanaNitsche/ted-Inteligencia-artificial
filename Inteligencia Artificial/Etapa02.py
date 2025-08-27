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

import matplotlib.pyplot as plt

from apoio import Obstaculos
from metricas import calcular_metricas


def visualizar_tempo_real(agente, step_delay=0.05, max_steps=100000, salvar_em=None):
    """Anima a exploração do agente em tempo real no grid.

    Resumo:
        Exibe a malha do grid, plota os obstáculos e, a cada passo, atualiza a
        posição do agente e a trilha percorrida. A execução segue até o agente
        sinalizar término interno ou atingir `max_steps`.

    Args:
        agente (AgenteReativoSimples):
            Instância do agente com memória que explorará o grid.
        step_delay (float, opcional):
            Pausa (em segundos) entre frames/atualizações. Padrão: 0.05.
        max_steps (int, opcional):
            Número máximo de passos da animação (failsafe). Padrão: 100000.
        salvar_em (str | None, opcional):
            Caminho de arquivo para salvar a figura final. Se None, não salva.

    Returns:
        None: A função exibe a animação; não retorna valor.
    """
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))

    n = agente.grid_size

    for i in range(n + 1):
        ax.plot([-.5, n-.5], [i-.5, i-.5])
        ax.plot([i-.5, i-.5], [-.5, n-.5])

    if agente.obstaculos:
        ox = [x for x, y in agente.obstaculos]
        oy = [y for x, y in agente.obstaculos]
        ax.scatter(ox, oy, marker='s', s=200, alpha=0.5, label='Obstáculos')

    px, py = [agente.position[0]], [agente.position[1]]
    (linha_path,) = ax.plot(px, py, marker='o', label='Caminho')
    scatter_agente = ax.scatter([agente.position[0]], [agente.position[1]], s=120, marker='*', label='Agente')

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_aspect('equal', adjustable='box')
    ax.invert_yaxis()
    ax.legend(loc='upper right')
    ax.set_title("Exploração em tempo real")

    def on_step(step_idx, pos_before, direc, pos_after, paredes_tocadas):
        """Atualiza a visualização a cada passo do agente.

        Resumo:
            Atualiza a linha do caminho, a posição do marcador do agente e o
            título com contadores úteis da exploração.

        Args:
            step_idx (int): Índice do passo atual.
            pos_before (tuple[int, int]): Posição (x, y) antes do movimento.
            direc (str): Direção do passo ('N', 'S', 'E' ou 'O').
            pos_after (tuple[int, int]): Posição (x, y) após o movimento.
            paredes_tocadas (frozenset[str]): Bords detectadas até o momento.

        Returns:
            None
        """
        px.append(pos_after[0])
        py.append(pos_after[1])
        linha_path.set_data(px, py)
        scatter_agente.set_offsets([[pos_after[0], pos_after[1]]])
        ax.set_title(
            f"Passos: {step_idx}  |  Visitadas: {len(agente.visited)}  |  Paredes: {''.join(sorted(paredes_tocadas))}"
        )
        plt.pause(step_delay)

    agente.explore(max_steps=max_steps, on_step=on_step)

    plt.ioff()
    if salvar_em:
        plt.savefig(salvar_em, bbox_inches='tight')
    plt.show()



class AgenteReativoSimples:
    """Agente reativo com memória para exploração sistemática de um grid.

    Resumo:
        Mantém:
        - `visited`: conjunto de células visitadas;
        - `caminho_percorrido`: pilha para retrocesso (backtracking);
        - contadores de passos (totais e redundantes).
        Em cada estado, tenta avançar para vizinhos não visitados.
        Se não houver, realiza backtracking pelo rastro salvo.

    Atributos públicos principais:
        grid_size (int): Dimensão do grid (grid_size x grid_size).
        obstaculos (set[tuple[int,int]]): Células bloqueadas.
        directions (list[str]): Ordem de varredura das direções (em apoio).
        position (tuple[int,int]): Posição corrente (x, y).
        posicao_inicial (tuple[int,int]): Posição inicial válida.
        perimeter_detected (set[str]): Bordas do grid detectadas {'N','S','E','O'}.
        visited (set[tuple[int,int]]): Células já visitadas.
        caminho_percorrido (list[tuple[int,int]]): Rastro para backtracking.
        passos_totais (int): Passos válidos executados.
        passos_backtracking (int): Passos que desfazem o último avanço.
        passos_redundantes_puros (int): Passos que retornam a células visitadas
            que não são exatamente o passo anterior (revisita "pura").
        passos_redundantes (int): Soma de redundantes puros + backtracking.

    Args:
        grid_size (int):
            Tamanho do grid.
        seed (int | None, opcional):
            Semente aleatória para reprodutibilidade. Padrão: None.
    """
    def __init__(self, grid_size, seed=None):
        """Construtor do agente reativo com memória.

        Resumo:
            Configura o grid, carrega obstáculos, sorteia posição inicial
            válida e inicializa estruturas de memória e contadores.

        Args:
            grid_size (int):
                Tamanho do grid (lado).
            seed (int | None, opcional):
                Semente aleatória para reprodutibilidade. Padrão: None.

        Returns:
            None
        """
        if seed is not None:
            random.seed(seed)

        self.grid_size = grid_size
        self.obstaculos = set(Obstaculos.OBSTACULOS)
        self.directions = Obstaculos.DIRECTIONS

        self.position = self.random_initial_position()
        while self.position in self.obstaculos:
            self.position = self.random_initial_position()
        self.posicao_inicial = self.position  

        self.perimeter_detected = set()
        self.visited = {self.position}
        self.caminho_percorrido: List[tuple[int, int]] = []

        self.passos_totais = 0
        self.passos_backtracking = 0
        self.passos_redundantes_puros = 0
        self.passos_redundantes = 0

    def random_initial_position(self):
        """Sorteia uma posição inicial válida no grid.

        Resumo:
            Seleciona coordenadas (x, y) uniformemente dentro do grid.
            A validação contra obstáculos é feita no __init__.

        Args:
            None

        Returns:
            tuple[int, int]: Posição inicial (x, y) sorteada.
        """
        x = random.randint(0, self.grid_size - 1)
        y = random.randint(0, self.grid_size - 1)
        print(f"Posição inicial aleatória: ({x}, {y})")
        return (x, y)


    def move(self, direction):
        """Realiza um movimento (se válido) e atualiza contadores/memória.

        Resumo:
            Tenta mover uma célula na direção informada respeitando limites
            e obstáculos. Ao mover:
              - incrementa `passos_totais`;
              - atualiza métricas de redundância;
              - atualiza `visited`, `caminho_percorrido` e `position`.

        Args:
            direction (str):
                Direção do movimento: 'N', 'S', 'E' ou 'O'.

        Returns:
            bool: True se o movimento foi realizado; False se inválido
            (limite/obstáculo) e nenhum estado foi alterado.
        """
        x, y = self.position
        coor_movimento = None

        if direction == 'N' and y < self.grid_size - 1:  coor_movimento = (x, y + 1)
        elif direction == 'S' and y > 0:                 coor_movimento = (x, y - 1)
        elif direction == 'E' and x < self.grid_size - 1: coor_movimento = (x + 1, y)
        elif direction == 'O' and x > 0:                 coor_movimento = (x - 1, y)

        if coor_movimento is None:
            return False
        if coor_movimento in self.obstaculos:
            return False

        self.passos_totais += 1
        if coor_movimento in self.visited:
            self.passos_redundantes += 1
            if self.caminho_percorrido and coor_movimento == self.caminho_percorrido[-1]:
                self.passos_backtracking += 1           
            else:
                self.passos_redundantes_puros += 1      
        else:
            self.caminho_percorrido.append(self.position)


        self.position = coor_movimento
        self.visited.add(self.position)
        return True

    def perceive(self):
        """Atualiza o conjunto das bordas do grid já detectadas.

        Resumo:
            Checa se a posição atual está em alguma borda e acrescenta
            a letra correspondente ao conjunto {'N','S','E','O'}.

        Args:
            None

        Returns:
            None
        """
        x, y = self.position
        if x == 0:                    self.perimeter_detected.add('O')
        if x == self.grid_size - 1:   self.perimeter_detected.add('E')
        if y == 0:                    self.perimeter_detected.add('S')
        if y == self.grid_size - 1:   self.perimeter_detected.add('N')

    def vizinhos_livres(self, x, y):
        """Gera vizinhos livres (não-obstáculo) na ordem de varredura.

        Resumo:
            Para a célula (x, y), varre as quatro direções conforme
            `self.directions`, e produz as posições adjacentes dentro
            do grid que não são obstáculos.

        Args:
            x (int): Coordenada x da célula.
            y (int): Coordenada y da célula.

        Yields:
            tuple[str, tuple[int, int]]:
                Par (direcao, posicao), onde `direcao` ∈ {'N','S','E','O'}
                e `posicao` é a célula (nx, ny) correspondente.
        """
        desloc = {'N': (0, 1), 'S': (0, -1), 'E': (1, 0), 'O': (-1, 0)}
        for direcao in self.directions:
            dx, dy = desloc[direcao]
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                pos = (nx, ny)
                if pos not in self.obstaculos:
                    yield direcao, pos

    def proximo_na_exploracao(self):
        """Decide a próxima ação: avançar para vizinho novo ou retroceder.

        Resumo:
            - Prioriza avançar para o primeiro vizinho livre ainda não visitado.
            - Se não houver, retira o topo de `caminho_percorrido` (pilha)
              para retroceder (backtracking) ao ponto anterior.
            - Se a pilha estiver vazia, indica término.

        Args:
            None

        Returns:
            tuple[str | None, str | None]:
                - ('avanco', direcao) quando há vizinho livre não visitado;
                - ('retrocesso', direcao) quando será feito backtracking;
                - (None, None) quando não há mais o que explorar.
        """
        x, y = self.position
        for direcao, (xn, yn) in self.vizinhos_livres(x, y):
            if (xn, yn) not in self.visited:
                return ('avanco', direcao)

        if self.caminho_percorrido:
            xa, ya = self.caminho_percorrido.pop()
            if xa == x - 1 and ya == y: return ('retrocesso', 'O')
            if xa == x + 1 and ya == y: return ('retrocesso', 'E')
            if ya == y - 1 and xa == x: return ('retrocesso', 'S')
            if ya == y + 1 and xa == x: return ('retrocesso', 'N')

        return (None, None)


    def has_detected_perimeter(self):
        """Indica se as quatro bordas do grid já foram detectadas.

        Resumo:
            Útil como métrica auxiliar para avaliar cobertura do perímetro.

        Args:
            None

        Returns:
            bool: True se {'N','S','E','O'} ⊆ `perimeter_detected`.
        """
        return len(self.perimeter_detected) == 4  

    def explore(self, max_steps=100000, on_step=None):
        """Executa a política de exploração até exaurir células acessíveis.

        Resumo:
            Laço principal:
            1) Decide entre 'avanco' a vizinho não visitado ou 'retrocesso';
            2) Tenta mover; se conseguir, chama `perceive` e o callback;
            3) Interrompe se não há mais o que fazer ou atingir `max_steps`.

        Args:
            max_steps (int, opcional):
                Limite superior de passos válidos. Padrão: 100000.
            on_step (callable | None, opcional):
                Callback de visualização/registro chamado a cada passo com
                assinatura:
                on_step(passos_totais, pos_antes, direcao, pos_depois, paredes_tocadas)

        Returns:
            set[tuple[int, int]]: Conjunto final de células visitadas.
        """
        while self.passos_totais < max_steps:
            modo, direcao = self.proximo_na_exploracao()
            if modo is None:
                break

            pos_antes = self.position
            if self.move(direcao):
                self.perceive()
                pos_depois = self.position

                if on_step is not None:
                    try:
                        on_step(self.passos_totais, pos_antes, direcao, pos_depois, frozenset(self.perimeter_detected))
                    except Exception as e:
                        print(f"on_step error: {e}")
            else:
                break
        return self.visited


if __name__ == "__main__":
    print("Iniciando exploração do grid...")
    agente = AgenteReativoSimples(grid_size=10, seed=42)

    visualizar_tempo_real(agente, step_delay=0.03, max_steps=100000, salvar_em=None)

    m = calcular_metricas(agente)
    print("Métricas:", m)
