"""
Etapa 4 — Agente Baseado em Utilidade (Variação 1: completamente observável)

Resumo:
    O ambiente é um grid n×n com custos de terreno {1 (normal), 2 (arenoso), 3 (rochoso)}.
    O agente deve, dado um início e um fim, encontrar o caminho de MENOR CUSTO TOTAL.
    Nesta variação, TODO o mapa de custos é conhecido desde o início.

    O algoritmo utilizado é Dijkstra (pesos não negativos), que minimiza a soma
    dos custos de entrada em cada célula ao longo do caminho.

Métricas:
    - sucesso (bool): alcançou o destino?
    - custo_total (float): soma dos custos das células no caminho encontrado.
    - comprimento (int): número de passos no caminho (len(path)-1).
"""

import heapq
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap


def visualizar_tempo_real_utilidade(agente, step_delay=0.03, percorrer=True, salvar_em=None):
    """Anima a execução do Dijkstra em tempo real sobre o grid de custos.

    Resumo:
        Renderiza o grid colorido de acordo com o custo (1/2/3), marca início/fim,
        e atualiza a cada iteração do Dijkstra os conjuntos de nós visitados,
        a fronteira (fila de prioridade), e o caminho parcial até o momento.
        Ao concluir, pode animar o deslocamento do agente ao longo do caminho ótimo.

    Args:
        agente (AgenteUtilidade):
            Instância configurada com grid, início, fim e matriz de custos.
        step_delay (float, opcional):
            Pausa (em segundos) entre atualizações do gráfico. Padrão: 0.03.
        percorrer (bool, opcional):
            Se True, anima o agente percorrendo o caminho final encontrado. Padrão: True.
        salvar_em (str | None, opcional):
            Caminho para salvar a figura final. Se None, não salva.

    Returns:
        tuple[list[tuple[int,int]] | None, float]:
            (path, custo_total), onde `path` é a lista de coordenadas (x,y)
            do caminho ótimo, ou None se inalcançável; `custo_total` é o custo
            acumulado calculado pelo Dijkstra (float; inf se sem caminho).
    """
    plt.ion()
    fig, ax = plt.subplots(figsize=(7, 7))
    n = agente.grid_size

    for i in range(n + 1):
        ax.plot([-.5, n-.5], [i-.5, i-.5], color="k", lw=0.5, alpha=0.4)
        ax.plot([i-.5, i-.5], [-.5, n-.5], color="k", lw=0.5, alpha=0.4)

    terrain = agente.terrain
    cmap = ListedColormap(["#2ecc71", "#f1c40f", "#e74c3c"]) 
    norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5], cmap.N)
    ax.imshow(
        terrain, cmap=cmap, norm=norm, origin="upper",
        extent=[-0.5, n-0.5, n-0.5, -0.5]
    )

    ax.scatter([agente.start[0]], [agente.start[1]], s=140, marker='s', c="white", edgecolors="k", label='Início')
    ax.scatter([agente.end[0]],   [agente.end[1]],   s=140, marker='^', c="white", edgecolors="k", label='Fim')

    visited_sc  = ax.scatter([], [], s=20, marker='.', c="#34495e", label='Visitados')
    frontier_sc = ax.scatter([], [], s=40, marker='o', c="#e67e22", label='Fronteira')
    (path_ln,)  = ax.plot([], [], marker='o', lw=2, c="white", label='Caminho')
    agent_sc    = ax.scatter([agente.start[0]], [agente.start[1]], s=150, marker='*', c="white", edgecolors="k", label='Agente')

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_aspect('equal', adjustable='box')
    ax.invert_yaxis()
    ax.legend(loc='upper right')
    ax.set_title("Dijkstra (menor custo) — inicializando...")

    fig.canvas.draw(); plt.pause(step_delay)

    def on_step(evt: str,
                node: Optional[Tuple[int, int]],
                cost_so_far: Dict[Tuple[int,int], float],
                prev: Dict[Tuple[int,int], Tuple[int,int]],
                frontier_heap,
                visited_set):
        """Callback interno para atualizar a visualização a cada evento do Dijkstra.

        Resumo:
            Atualiza pontos visitados, fronteira (conteúdo atual do heap),
            o caminho parcial (se já há predecessor até o fim) e o título
            com contadores básicos. Quando termina, mostra o custo final.

        Args:
            evt (str):
                Evento: "init", "visit", "enqueue", "done" ou "fail".
            node (tuple[int,int] | None):
                Nó atual do evento.
            cost_so_far (dict[tuple[int,int], float]):
                Distâncias acumuladas conhecidas até o momento (Dijkstra).
            prev (dict[tuple[int,int], tuple[int,int]]):
                Predecessores para reconstruir caminho.
            frontier_heap (list[tuple[float, tuple[int,int]]]):
                Conteúdo atual da fila de prioridade (heap).
            visited_set (set[tuple[int,int]]):
                Conjunto de nós já removidos do heap (visitados).

        Returns:
            None
        """
        vx = [p[0] for p in visited_set]
        vy = [p[1] for p in visited_set]
        visited_sc.set_offsets(list(zip(vx, vy)) if vx else np.empty((0, 2)))

        fronteira = [item[1] for item in frontier_heap]
        fx = [p[0] for p in fronteira]
        fy = [p[1] for p in fronteira]
        frontier_sc.set_offsets(list(zip(fx, fy)) if fx else np.empty((0, 2)))

        if agente.end in prev or agente.end == agente.start:
            path = agente._reconstruir(prev, agente.end)
            if path:
                px = [p[0] for p in path]; py = [p[1] for p in path]
                path_ln.set_data(px, py)
                agent_sc.set_offsets([path[-1]])

        title = f"Dijkstra: evt={evt} | visitados={len(visited_set)} | fronteira={len(fronteira)}"
        if evt == "done":
            title += f" | custo_total={cost_so_far.get(agente.end, float('inf')):.1f}"
            frontier_sc.set_offsets(np.empty((0, 2)))
        ax.set_title(title)
        plt.pause(step_delay)

    path, custo = agente.dijkstra_realtime(on_step=on_step)

    if percorrer and path:
        for p in path:
            agent_sc.set_offsets([p])
            plt.pause(step_delay)

    plt.ioff()
    if salvar_em:
        plt.savefig(salvar_em, bbox_inches='tight')
    plt.show()

    return path, custo


class AgenteUtilidade:
    """Agente de utilidade em grid totalmente observável (custos conhecidos).

    Resumo:
        Mantém um grid n×n com custos inteiros {1,2,3}. Dado início e fim,
        utiliza Dijkstra para obter o menor custo total de deslocamento,
        onde o custo de cada passo é o custo da célula de destino.

    Args:
        grid_size (int):
            Dimensão do grid (grid_size × grid_size).
        inicio (tuple[int,int]):
            Posição inicial (x, y).
        fim (tuple[int,int]):
            Posição destino (x, y).
        terreno (numpy.ndarray | None, opcional):
            Matriz (n, n) de custos {1,2,3}. Se None, gera um padrão radial
            com centro mais caro.

    Raises:
        ValueError: Se `terreno` existir e não for (grid_size, grid_size),
                    ou se início/fim estiverem fora do grid.
    """

    def __init__(self, grid_size: int, *,
                 inicio: Tuple[int,int],
                 fim: Tuple[int,int],
                 terreno: Optional[np.ndarray] = None):
        """
        Resumo:
            Inicializa o agente, valida início/fim e carrega (ou gera) a matriz
            de custos do terreno.

        Args:
            grid_size (int): Tamanho do lado do grid.
            inicio (tuple[int,int]): Coordenada inicial (x, y).
            fim (tuple[int,int]): Coordenada final (x, y).
            terreno (numpy.ndarray | None): Matriz de custos 1/2/3 ou None.

        Returns:
            None
        """
        self.grid_size = grid_size
        self.start = inicio
        self.end = fim

        if terreno is None:
            self.terrain = self._gera_terreno_central()
        else:
            self.terrain = np.array(terreno, dtype=int)
            if self.terrain.shape != (grid_size, grid_size):
                raise ValueError("Terreno deve ter shape (grid_size, grid_size).")

        self._valida_no_grid(self.start)
        self._valida_no_grid(self.end)

    def _gera_terreno_central(self) -> np.ndarray:
        """Gera um terreno com centro caro e periferia barata (1/2/3).

        Resumo:
            Cria uma matriz (n, n) com custo 1, exceto um miolo com custo 3
            e um anel ao redor com custo 2, a partir da distância Manhattan
            ao centro do grid.

        Args:
            None

        Returns:
            numpy.ndarray: Matriz (n, n) de inteiros {1,2,3}.
        """
        n = self.grid_size
        t = np.ones((n, n), dtype=int)
        cx, cy = n//2, n//2
        for y in range(n):
            for x in range(n):
                d = abs(x - cx) + abs(y - cy)
                if d <= 1:
                    t[y, x] = 3 
                elif d == 2 or d == 3:
                    t[y, x] = 2 
                else:
                    t[y, x] = 1 
        return t

    def _valida_no_grid(self, pos: Tuple[int,int]):
        """Valida se a posição está dentro do grid.

        Args:
            pos (tuple[int,int]): Coordenada (x, y) a validar.

        Raises:
            ValueError: Se (x, y) estiver fora [0, grid_size).

        Returns:
            None
        """
        x, y = pos
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
            raise ValueError(f"Fora do grid: {pos}")

    def _vizinhos(self, x: int, y: int):
        """Gera os 4-vizinhos dentro do grid.

        Args:
            x (int): Coordenada x.
            y (int): Coordenada y.

        Yields:
            tuple[int,int]: Coordenadas (nx, ny) vizinhas válidas.
        """
        for dx, dy in ((0,1),(0,-1),(1,0),(-1,0)):
            nx, ny = x+dx, y+dy
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                yield nx, ny

    def _custo(self, nx: int, ny: int) -> int:
        """Retorna o custo da célula de destino (nx, ny).

        Args:
            nx (int): Coordenada x do destino.
            ny (int): Coordenada y do destino.

        Returns:
            int: Custo inteiro {1,2,3} daquela célula.
        """
        return int(self.terrain[ny, nx])

    def _reconstruir(self, prev: Dict[Tuple[int,int], Tuple[int,int]], alvo: Tuple[int,int]):
        """Reconstrói o caminho do início até `alvo` a partir de `prev`.

        Resumo:
            Percorre o dicionário de predecessores para montar a lista de
            coordenadas do caminho. Se não existir predecessor e `alvo` não
            for o início, retorna None.

        Args:
            prev (dict[tuple[int,int], tuple[int,int]]): Predecessores.
            alvo (tuple[int,int]): Nó final (x, y).

        Returns:
            list[tuple[int,int]] | None:
                Caminho do início a `alvo` ou None, se não houver.
        """
        if alvo not in prev and alvo != self.start:
            return None
        path = [alvo]
        cur = alvo
        while cur != self.start:
            cur = prev[cur]
            path.append(cur)
        path.reverse()
        return path

    def dijkstra_realtime(self, on_step=None):
        """Executa Dijkstra e aciona um callback para visualização/telemetria.

        Resumo:
            Calcula o caminho de menor custo (pesos não negativos) entre
            `self.start` e `self.end`. A cada evento relevante, chama `on_step`
            com estado atual (visitados, heap, distâncias, predecessores).

        Args:
            on_step (callable | None, opcional):
                Callback com assinatura:
                    on_step(evt: str,
                            node: tuple[int,int] | None,
                            cost_so_far: dict[(int,int), float],
                            prev: dict[(int,int), (int,int)],
                            frontier_heap: list[tuple[float, (int,int)]],
                            visited_set: set[(int,int)])

        Returns:
            tuple[list[tuple[int,int]] | None, float]:
                (path, custo_total). `path` é o caminho ótimo ou None,
                `custo_total` é a distância final (inf se inalcançável).
        """
        start, goal = self.start, self.end
        dist: Dict[Tuple[int,int], float] = {start: 0.0}
        prev: Dict[Tuple[int,int], Tuple[int,int]] = {}
        visited = set()
        heap = [(0.0, start)]

        if on_step:
            on_step("init", start, dist, prev, list(heap), set(visited))

        while heap:
            custo, node = heapq.heappop(heap)
            if node in visited:
                continue
            visited.add(node)

            if node == goal:
                if on_step:
                    on_step("done", node, dist, prev, list(heap), set(visited))
                path = self._reconstruir(prev, goal)
                return path, dist.get(goal, float("inf"))

            if on_step:
                on_step("visit", node, dist, prev, list(heap), set(visited))

            x, y = node
            for nx, ny in self._vizinhos(x, y):
                step_cost = self._custo(nx, ny)
                novo = custo + step_cost
                if (nx, ny) not in dist or novo < dist[(nx, ny)]:
                    dist[(nx, ny)] = novo
                    prev[(nx, ny)] = node
                    heapq.heappush(heap, (novo, (nx, ny)))
                    if on_step:
                        on_step("enqueue", (nx, ny), dist, prev, list(heap), set(visited))

        if on_step:
            on_step("fail", None, dist, prev, [], set(visited))
        return None, float("inf")

    def metricas(self, path, custo_total):
        """Calcula métricas do trajeto encontrado.

        Resumo:
            Retorna sucesso, custo_total e comprimento (nº de passos) do
            caminho, caso exista.

        Args:
            path (list[tuple[int,int]] | None): Caminho retornado pelo Dijkstra.
            custo_total (float): Custo acumulado (ou inf se sem caminho).

        Returns:
            dict:
                {
                    "sucesso": bool,
                    "custo_total": float | None,
                    "comprimento": int | None
                }
        """
        sucesso = path is not None
        comprimento = (len(path) - 1) if sucesso else None
        return {
            "sucesso": sucesso,
            "custo_total": None if not sucesso else float(custo_total),
            "comprimento": comprimento,
        }

    def executar_acoes(self, path: List[Tuple[int,int]]):
        """Converte um caminho em ações cardinais.

        Resumo:
            Para cada par de posições consecutivas, infere a ação entre
            {'N','S','E','O'} assumindo movimentos 4-conectados.

        Args:
            path (list[tuple[int,int]]):
                Lista de coordenadas (x, y) consecutivas do caminho.

        Returns:
            list[str]:
                Sequência de ações. Vazio se path for None ou tiver < 2 posições.

        Raises:
            RuntimeError: Se detectar um salto não adjacente.
        """
        if not path or len(path) < 2:
            return []
        acoes = []
        for (x1, y1), (x2, y2) in zip(path, path[1:]):
            dx, dy = x2 - x1, y2 - y1
            if   dx == 1 and dy == 0:  acoes.append('E')
            elif dx == -1 and dy == 0: acoes.append('O')
            elif dx == 0 and dy == 1:  acoes.append('S')  
            elif dx == 0 and dy == -1: acoes.append('N')
            else:
                raise RuntimeError(f"Passo inválido: {(x1,y1)} -> {(x2,y2)}")
        return acoes


if __name__ == "__main__":
    agente = AgenteUtilidade(grid_size=11, inicio=(5, 1), fim=(5, 9))
    path, custo = visualizar_tempo_real_utilidade(agente, step_delay=0.01, percorrer=True)
    print("Métricas:", agente.metricas(path, custo))
    if path:
        print("Ações:", agente.executar_acoes(path))
