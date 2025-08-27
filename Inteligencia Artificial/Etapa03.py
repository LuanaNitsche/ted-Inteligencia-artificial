import random
from collections import deque

import matplotlib.pyplot as plt
import numpy as np

from apoio import Obstaculos as ObstFixos


def visualizar_tempo_real_etapa3(agente, step_delay=0.03, percorrer=True, salvar_em=None):
    """Anima a busca do caminho (BFS) em tempo real no grid.

    Resumo:
        Desenha o grid, os obstáculos (se houver), marca início/fim e atualiza,
        a cada evento da BFS, os conjuntos de visitados, fronteira e o caminho
        parcial. Ao término, opcionalmente faz o agente “percorrer” o caminho
        encontrado passo a passo.

    Args:
        agente (AgenteCaminho):
            Instância configurada do agente e do ambiente (grid/obstáculos).
        step_delay (float, opcional):
            Pausa, em segundos, entre atualizações do gráfico. Padrão: 0.03.
        percorrer (bool, opcional):
            Se True, após encontrar o caminho, anima o deslocamento do agente
            ao longo do caminho. Padrão: True.
        salvar_em (str | None, opcional):
            Caminho para salvar a figura final. Se None, não salva.

    Returns:
        list[tuple[int, int]] | None:
            O caminho encontrado (lista de coordenadas (x, y)) ou None
            se não houver caminho.
    """
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))
    n = agente.grid_size

    for i in range(n + 1):
        ax.plot([-.5, n-.5], [i-.5, i-.5])
        ax.plot([i-.5, i-.5], [-.5, n-.5])

    # Obstáculos
    ox, oy = [], []
    for y in range(n):
        for x in range(n):
            if agente.grid[y][x] == 1:
                ox.append(x); oy.append(y)
    if ox:
        ax.scatter(ox, oy, marker='s', s=200, alpha=0.5, label='Obstáculos')

    # Marcadores de início e fim
    ax.scatter([agente.start[0]], [agente.start[1]], s=120, marker='s', label='Início')
    ax.scatter([agente.end[0]],   [agente.end[1]],   s=120, marker='^', label='Fim')

    # Elementos dinâmicos
    visited_sc  = ax.scatter([], [], s=20, marker='.', label='Visitados')
    frontier_sc = ax.scatter([], [], s=40, marker='o', label='Fronteira')
    visited_sc.set_offsets(np.empty((0, 2)))
    frontier_sc.set_offsets(np.empty((0, 2)))
    (path_ln,) = ax.plot([], [], marker='o', label='Caminho')
    agent_sc   = ax.scatter([agente.start[0]], [agente.start[1]], s=140, marker='*', label='Agente')

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_aspect('equal', adjustable='box')
    ax.invert_yaxis()
    ax.legend(loc='upper right')
    ax.set_title("BFS em tempo real")

    fig.canvas.draw()
    plt.pause(step_delay)

    def on_step(evt, node, path, fila_list, visitados_set):
        """Callback de visualização chamado pela BFS a cada evento.

        Resumo:
            Atualiza os conjuntos de nós visitados e fronteira, o caminho
            parcial e a posição do agente. Também altera o título com contadores
            úteis e sinaliza quando terminar/falhar.

        Args:
            evt (str): Evento ('init', 'visit', 'enqueue', 'done', 'fail').
            node (tuple[int, int] | None): Nó atual do evento.
            path (list[tuple[int, int]] | None): Caminho parcial até `node`.
            fila_list (list[tuple[tuple[int, int], list[tuple[int, int]]]]):
                Conteúdo atual da fila (posição e caminho associado).
            visitados_set (set[tuple[int, int]]):
                Conjunto de posições já visitadas pela BFS.

        Returns:
            None
        """
        vx = [p[0] for p in visitados_set]; vy = [p[1] for p in visitados_set]
        visited_sc.set_offsets(list(zip(vx, vy)) if vx else np.empty((0, 2)))

        fronteira = [pos for (pos, _p) in fila_list]
        fx = [p[0] for p in fronteira]; fy = [p[1] for p in fronteira]
        frontier_sc.set_offsets(list(zip(fx, fy)) if fx else np.empty((0, 2)))

        if path:
            px = [p[0] for p in path]; py = [p[1] for p in path]
            path_ln.set_data(px, py)
            agent_sc.set_offsets([path[-1]])

        title = f"BFS: evt={evt} | visitados={len(visitados_set)} | fronteira={len(fila_list)}"
        if evt == 'fail':
            title += " | SEM CAMINHO"

        if evt == 'done':
            frontier_sc.set_offsets(np.empty((0, 2)))

        ax.set_title(title)
        plt.pause(step_delay)

    path = agente.find_path_realtime(on_step=on_step)

    if percorrer and path:
        for p in path:
            agent_sc.set_offsets([p])
            plt.pause(step_delay)

    plt.ioff()
    if salvar_em:
        plt.savefig(salvar_em, bbox_inches='tight')
    plt.show()
    return path


class AgenteCaminho:
    """Agente de caminho com BFS (grid livre ou com obstáculos estáticos).

    Resumo:
        Mantém o grid (0=célula livre, 1=obstáculo), sorteia (ou recebe) início
        e fim e executa BFS para retornar um caminho mais curto, quando existir.
        Suporta visualização em tempo real via callback.

    Args:
        grid_size (int):
            Dimensão do grid (grid_size x grid_size).
        inicio (tuple[int,int] | None, opcional):
            Posição inicial (x, y). Se None, é sorteada uma célula livre.
        fim (tuple[int,int] | None, opcional):
            Posição de destino (x, y). Se None, é sorteada célula livre ≠ início.
        usar_obstaculos_fixos (bool, opcional):
            Se True, aplica `apoio.Obstaculos.OBSTACULOS` no grid. Padrão: False.
        seed (int | None, opcional):
            Semente para reprodutibilidade do sorteio. Padrão: None.
        garantir_caminho (bool, opcional):
            Se True, re-sorteia início/fim até que exista conexão alcançável
            (até `max_tentativas`). Padrão: False.
        max_tentativas (int, opcional):
            Limite de tentativas ao tentar garantir conectividade. Padrão: 1000.

    Raises:
        RuntimeError: Se não conseguir sortear par início/fim alcançáveis.
        ValueError: Se início/fim fora do grid, coincidem, ou sobre obstáculo.
    """

    def __init__(self, grid_size, *,
                 inicio=None, fim=None,
                 usar_obstaculos_fixos=False,
                 seed=None,
                 garantir_caminho=False,
                 max_tentativas=1000):
        """Constrói o agente e define grid, início e fim."""
        if seed is not None:
            random.seed(seed)

        self.grid_size = grid_size
        self.grid = self._gera_grid(usar_obstaculos_fixos)

        if inicio is None or fim is None:
            for _ in range(max_tentativas):
                s = self._random_posicao_livre()
                f = self._random_posicao_livre(exclude={s})
                if not garantir_caminho or self._alcancavel(s, f):
                    inicio, fim = s, f
                    break
            else:
                raise RuntimeError("Não consegui sortear início/fim alcançáveis em max_tentativas.")

        self._valida_no_grid(inicio); self._valida_no_grid(fim)
        if self.grid[inicio[1]][inicio[0]] == 1: raise ValueError(f"Início {inicio} em obstáculo.")
        if self.grid[fim[1]][fim[0]] == 1:       raise ValueError(f"Fim {fim} em obstáculo.")
        if inicio == fim:                         raise ValueError("Início e fim não podem ser iguais.")

        self.start, self.end = inicio, fim

    def _random_posicao_livre(self, exclude=None):
        """Sorteia aleatoriamente uma posição livre (não-obstáculo).

        Resumo:
            Gera amostras uniformes de células até encontrar uma livre
            que não pertença ao conjunto `exclude`.

        Args:
            exclude (set[tuple[int,int]] | None, opcional):
                Conjunto de posições a evitar. Padrão: set().

        Returns:
            tuple[int, int]: Uma posição (x, y) livre.
        """
        exclude = exclude or set()
        while True:
            x = random.randint(0, self.grid_size-1)
            y = random.randint(0, self.grid_size-1)
            if self.grid[y][x] == 0 and (x, y) not in exclude:
                return (x, y)

    def _alcancavel(self, s, f):
        """Testa conectividade entre s e f em grid (0=livre, 1=obstáculo).

        Resumo:
            BFS “seca” (sem construir caminho) apenas para verificar se existe
            um caminho entre `s` e `f` atravessando células livres.

        Args:
            s (tuple[int,int]): Posição origem.
            f (tuple[int,int]): Posição destino.

        Returns:
            bool: True se existe algum caminho; False caso contrário.
        """
        q = deque([s]); vis = {s}
        while q:
            x, y = q.popleft()
            if (x, y) == f: return True
            for dx, dy in ((0,1),(0,-1),(1,0),(-1,0)):
                nx, ny = x+dx, y+dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size \
                and self.grid[ny][nx] == 0 and (nx, ny) not in vis:
                    vis.add((nx, ny)); q.append((nx, ny))
        return False

    def _gera_grid(self, usar_obstaculos_fixos):
        """Gera a matriz do grid marcando obstáculos, se solicitado.

        Resumo:
            Inicializa uma matriz n x n com zeros (livre) e, se
            `usar_obstaculos_fixos=True`, marca 1 nas coordenadas presentes
            em `apoio.Obstaculos.OBSTACULOS`.

        Args:
            usar_obstaculos_fixos (bool): Se True, aplica obstáculos fixos.

        Returns:
            list[list[int]]: Matriz do grid com 0 (livre) e 1 (obstáculo).
        """
        n = self.grid_size
        grid = [[0 for _ in range(n)] for _ in range(n)]
        if usar_obstaculos_fixos and hasattr(ObstFixos, "OBSTACULOS"):
            for (x, y) in ObstFixos.OBSTACULOS:
                if 0 <= x < n and 0 <= y < n:
                    grid[y][x] = 1
        return grid

    def _valida_no_grid(self, pos):
        """Verifica se uma posição está dentro dos limites do grid.

        Args:
            pos (tuple[int,int]): Posição (x, y) a validar.

        Raises:
            ValueError: Se a posição estiver fora do grid.

        Returns:
            None
        """
        x, y = pos
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
            raise ValueError(f"Posição fora do grid: {pos}")

    def is_valid(self, x, y):
        """Indica se (x, y) é célula livre dentro do grid.

        Args:
            x (int): Coordenada x.
            y (int): Coordenada y.

        Returns:
            bool: True se está no grid e `grid[y][x] == 0`; False caso contrário.
        """
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size and self.grid[y][x] == 0

    def find_path(self):
        """Executa BFS e retorna um caminho mais curto entre início e fim.

        Resumo:
            Usa BFS padrão (4 vizinhos) para construir o caminho mínimo
            em número de passos entre `self.start` e `self.end`, caso exista.

        Args:
            None

        Returns:
            list[tuple[int, int]] | None:
                Caminho (lista de (x, y)) da origem ao destino ou None
                se não houver caminho.
        """
        dirs = [(0,1), (0,-1), (1,0), (-1,0)]
        q = deque([(self.start, [self.start])])
        visited = {self.start}

        while q:
            (x, y), path = q.popleft()
            if (x, y) == self.end:
                return path

            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if self.is_valid(nx, ny) and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    q.append(((nx, ny), path + [(nx, ny)]))
        return None

    def executar_acoes(self, path):
        """Converte um caminho (lista de células) em ações cardinais.

        Resumo:
            Dado um caminho [(x0,y0), (x1,y1), ...], converte cada transição
            em uma ação entre {'N','S','E','O'}.

        Args:
            path (list[tuple[int,int]]):
                Lista de posições consecutivas do caminho.

        Returns:
            list[str]:
                Lista de ações ('N','S','E','O'). Vazia se caminho inválido
                (None ou com menos de 2 posições).

        Raises:
            RuntimeError: Se houver salto inválido entre posições adjacentes.
        """
        if not path or len(path) < 2:
            return []
        acoes = []
        for (x1, y1), (x2, y2) in zip(path, path[1:]):
            dx, dy = x2 - x1, y2 - y1
            if   dx == 1 and dy == 0:  acoes.append('E')
            elif dx == -1 and dy == 0: acoes.append('O')
            elif dx == 0 and dy == 1:  acoes.append('N')
            elif dx == 0 and dy == -1: acoes.append('S')
            else:
                raise RuntimeError(f"Passo inválido no path: {(x1,y1)} -> {(x2,y2)}")
        return acoes

    def print_grid(self, path=None):
        """Imprime o grid no console com S/F, obstáculos e (opcional) caminho.

        Resumo:
            Representa:
            - 'S' para início, 'F' para fim,
            - 'X' para obstáculos,
            - '*' para células do caminho,
            - '.' para livres fora do caminho.

        Args:
            path (list[tuple[int,int]] | None, opcional):
                Caminho a destacar na impressão. Padrão: None.

        Returns:
            None
        """
        caminho_set = set(path) if path else set()
        for y in range(self.grid_size):
            row = []
            for x in range(self.grid_size):
                p = (x, y)
                if p == self.start:
                    row.append("S")
                elif p == self.end:
                    row.append("F")
                elif path and p in caminho_set:
                    row.append("*")
                elif self.grid[y][x] == 1:
                    row.append("X")
                else:
                    row.append(".")
            print(" ".join(row))
        print()

    def metricas(self, path):
        """Calcula métricas básicas do caminho encontrado.

        Resumo:
            Retorna:
            - sucesso: se existe caminho (True/False)
            - comprimento: número de passos (= len(path) - 1) ou None

        Args:
            path (list[tuple[int,int]] | None):
                Caminho retornado por `find_path`/`find_path_realtime`.

        Returns:
            dict: {'sucesso': bool, 'comprimento': int | None}
        """
        sucesso = path is not None
        comprimento = (len(path) - 1) if sucesso else None
        return {"sucesso": sucesso, "comprimento": comprimento}

    def find_path_realtime(self, on_step=None):
        """Executa BFS com callback de tempo real e para assim que achar o fim.

        Resumo:
            Igual à BFS de `find_path`, mas:
            - chama `on_step(evt, node, path, fila_list, visitados_set)` em
              eventos ('init','visit','enqueue','done','fail');
            - retorna imediatamente ao alcançar o destino, evitando explorações
              excedentes.

        Args:
            on_step (callable | None, opcional):
                Callback de visualização/telemetria. Assinatura:
                on_step(evt: str,
                        node: tuple[int,int] | None,
                        path: list[tuple[int,int]] | None,
                        fila_list: list[tuple[tuple[int,int], list[tuple[int,int]]]],
                        visitados_set: set[tuple[int,int]])

        Returns:
            list[tuple[int, int]] | None:
                Caminho (x, y) se encontrou; None se inalcançável.
        """
        dirs = [(0,1), (0,-1), (1,0), (-1,0)]
        q = deque([(self.start, [self.start])])
        visited = {self.start}

        if on_step:
            on_step('init', self.start, [self.start], list(q), set(visited))

        while q:
            (x, y), path = q.popleft()

            if (x, y) == self.end:
                if on_step:
                    on_step('done', (x, y), path, list(q), set(visited))
                return path

            if on_step:
                on_step('visit', (x, y), path, list(q), set(visited))

            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if self.is_valid(nx, ny) and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    new_path = path + [(nx, ny)]

                    if (nx, ny) == self.end:
                        if on_step:
                            on_step('done', (nx, ny), new_path, list(q), set(visited))
                        return new_path

                    q.append(((nx, ny), new_path))
                    if on_step:
                        on_step('enqueue', (nx, ny), new_path, list(q), set(visited))

        if on_step:
            on_step('fail', None, None, [], set(visited))
        return None


if __name__ == "__main__":
    # fase1 — ambiente livre
    print("Fase 1 — Ambiente livre:")
    agente_livre = AgenteCaminho(
        grid_size=10,
        usar_obstaculos_fixos=False,
        inicio=None, fim=None,
        seed=42,
        garantir_caminho=True
    )
    print("Par sorteado (livre):", agente_livre.start, "->", agente_livre.end)
    path1 = visualizar_tempo_real_etapa3(agente_livre, step_delay=0.03, percorrer=True)
    print("Métricas (Fase 1):", agente_livre.metricas(path1))

    # fase2 — com obstáculo
    print("\nFase 2 — Obstáculos fixos:")
    agente_obs = AgenteCaminho(
        grid_size=10,
        usar_obstaculos_fixos=True,
        inicio=None, fim=None,
        seed=43,
        garantir_caminho=True
    )
    print("Par sorteado (obstáculos):", agente_obs.start, "->", agente_obs.end)
    path2 = visualizar_tempo_real_etapa3(agente_obs, step_delay=0.03, percorrer=True)
    print("Métricas (Fase 2):", agente_obs.metricas(path2))
