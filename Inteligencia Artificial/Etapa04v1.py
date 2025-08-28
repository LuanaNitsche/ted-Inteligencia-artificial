import heapq
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap


def visualizar_tempo_real_utilidade_conhecida(
    agente, step_delay=0.03, mostrar_valor=True, salvar_em=None
):
    """
    Anima a execução do agente quando o mapa é completamente conhecido.
    Em vez de mostrar a expansão do Dijkstra, mostramos:
      - Terreno (1/2/3)
      - (Opcional) mapa de custo-ao-objetivo (função de valor)
      - Movimento do agente seguindo a política ótima (greedy na utilidade)
    """
    agente.precompute_value_function()

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

    if mostrar_valor and agente.dist_to_goal is not None:
        d = agente.dist_to_goal.copy()
        finites = np.isfinite(d)
        if finites.any():
            d_norm = np.zeros_like(d, dtype=float)
            vals = d[finites]
            lo, hi = vals.min(), vals.max()
            if hi > lo:
                d_norm[finites] = (d[finites] - lo) / (hi - lo)
            ax.imshow(
                d_norm, cmap="gray", alpha=0.35, origin="upper",
                extent=[-0.5, n-0.5, n-0.5, -0.5]
            )

    ax.scatter([agente.start[0]], [agente.start[1]], s=140, marker='s', c="white", edgecolors="k", label='Início')
    ax.scatter([agente.end[0]],   [agente.end[1]],   s=140, marker='^', c="white", edgecolors="k", label='Fim')

    (path_ln,) = ax.plot([], [], marker='o', lw=2, c="white", label='Caminho')
    agent_sc = ax.scatter([agente.start[0]], [agente.start[1]], s=150, marker='*', c="white", edgecolors="k", label='Agente')

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_aspect('equal', adjustable='box')
    ax.invert_yaxis()
    ax.legend(loc='upper right')
    ax.set_title("Execução com mapa conhecido — planejando e iniciando...")

    fig.canvas.draw(); plt.pause(step_delay)

    path, custo_total = agente.executar_politica_otima()

    xs, ys = [], []
    for p in path:
        xs.append(p[0]); ys.append(p[1])
        path_ln.set_data(xs, ys)
        agent_sc.set_offsets([p])
        ax.set_title(f"Executando política ótima • pos={p} • custo_acum≈{agente.custo_acumulado(xs, ys):.1f}")
        plt.pause(step_delay)

    ax.set_title(f"Concluído • custo_total={custo_total:.1f} • passos={max(0, len(path)-1)}")

    plt.ioff()
    if salvar_em:
        plt.savefig(salvar_em, bbox_inches='tight')
    plt.show()

    return path, custo_total

import numpy as np

def terreno_figura_11x11() -> np.ndarray:
    """
    Retorna a matriz 11x11 de custos igual à figura:
    - 1 em todo o fundo (verde)
    - 3 na “mancha” central (vermelho)
    - 2 no anel/braços ao redor (amarelo)

    Observação:
    Coordenadas são (x,y) 0-based; esta função retorna array indexado [y, x].
    """
    n = 11
    t = np.ones((n, n), dtype=int)

    red = {
        (4,4),(5,4),(6,4),
        (3,5),(4,5),(5,5),(6,5),(7,5),
        (4,6),(5,6),(6,6),
        (5,3),(5,7),
        (5,4),(5,6)
    }

    for (x,y) in red:
        t[y, x] = 3

    yellow = {
        (3,4),(7,4),
        (4,3),(6,3),
        (4,7),(6,7),
        (3,6),(7,6),
        (5,2),(5,8),
        (2,5),(8,5)
    }
    for (x,y) in yellow:
        t[y, x] = 2

    return t


class AgenteUtilidade:
    """
    Mesmo cenário da sua Etapa 4 (custos 1/2/3), mas com comportamento adaptado
    ao conhecimento completo do mapa: calcula custo-ao-objetivo (função de valor)
    e segue a política ótima, sem "explorar" em tempo real.
    """

    def __init__(self, grid_size: int, *, inicio: Tuple[int,int], fim: Tuple[int,int], terreno: Optional[np.ndarray] = None):
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

        self.dist_to_goal: Optional[np.ndarray] = None

    def _gera_terreno_central(self) -> np.ndarray:
        n = self.grid_size
        t = np.ones((n, n), dtype=int)
        cx, cy = n//2, n//2
        for y in range(n):
            for x in range(n):
                d = abs(x - cx) + abs(y - cy)
                if d <= 1:       t[y, x] = 3
                elif d in (2, 3): t[y, x] = 2
                else:            t[y, x] = 1
        return t

    def _valida_no_grid(self, pos: Tuple[int,int]):
        x, y = pos
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
            raise ValueError(f"Fora do grid: {pos}")

    def _vizinhos(self, x: int, y: int):
        for dx, dy in ((0,1),(0,-1),(1,0),(-1,0)):
            nx, ny = x+dx, y+dy
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                yield nx, ny

    def _custo(self, nx: int, ny: int) -> int:
        return int(self.terrain[ny, nx])

    def precompute_value_function(self):
        """
        Dijkstra reverso a partir do objetivo:
        dist_to_goal[y, x] = custo mínimo para ir de (x,y) até self.end,
        considerando o custo de ENTRAR nas células seguintes.
        """
        n = self.grid_size
        INF = float("inf")
        dist = np.full((n, n), INF, dtype=float)
        prev: Dict[Tuple[int,int], Tuple[int,int]] = {}

        ex, ey = self.end
        dist[ey, ex] = 0.0

        heap: List[Tuple[float, Tuple[int,int]]] = [(0.0, (ex, ey))]
        visited = set()

        while heap:
            dcur, (x, y) = heapq.heappop(heap)
            if (x, y) in visited:
                continue
            visited.add((x, y))
            for nx, ny in self._vizinhos(x, y):
                new_cost = dcur + self._custo(x, y)
                if new_cost < dist[ny, nx]:
                    dist[ny, nx] = new_cost
                    prev[(nx, ny)] = (x, y)
                    heapq.heappush(heap, (new_cost, (nx, ny)))

        self.dist_to_goal = dist

    def melhor_vizinho_pela_utilidade(self, x: int, y: int) -> Optional[Tuple[int,int]]:
        """
        Dado (x,y), escolhe o vizinho n que minimiza:
            custo_entrar(n) + dist_to_goal[n]
        """
        assert self.dist_to_goal is not None, "Chame precompute_value_function() antes."
        best = None
        best_cost = float("inf")
        for nx, ny in self._vizinhos(x, y):
            step = self._custo(nx, ny)
            val = self.dist_to_goal[ny, nx]
            if not np.isfinite(val):
                continue
            score = step + val
            if score < best_cost:
                best_cost = score
                best = (nx, ny)
        return best

    def executar_politica_otima(self) -> Tuple[List[Tuple[int,int]], float]:
        """
        Segue a política greedy na utilidade usando dist_to_goal.
        Retorna (path, custo_total).
        """
        if self.start == self.end:
            return [self.start], 0.0

        assert self.dist_to_goal is not None, "Chame precompute_value_function() antes."
        if not np.isfinite(self.dist_to_goal[self.start[1], self.start[0]]):
            return [], float("inf")

        path = [self.start]
        x, y = self.start
        total = 0.0
        seen = {(x, y)}  

        for _ in range(self.grid_size * self.grid_size + 5):
            nxt = self.melhor_vizinho_pela_utilidade(x, y)
            if nxt is None:
                break
            nx, ny = nxt
            total += self._custo(nx, ny)
            path.append((nx, ny))
            x, y = nx, ny
            if (x, y) == self.end:
                return path, total
            if (x, y) in seen:
                break
            seen.add((x, y))

        return [], float("inf")

    def custo_acumulado(self, xs: List[int], ys: List[int]) -> float:
        """Apoio à visualização: soma custos ao longo dos pontos acumulados."""
        if not xs or not ys or len(xs) != len(ys):
            return 0.0
        total = 0.0
        for i in range(1, len(xs)):
            total += self._custo(xs[i], ys[i])
        return total

    def metricas(self, path, custo_total):
        sucesso = bool(path) and np.isfinite(custo_total)
        comprimento = (len(path) - 1) if sucesso else None
        return {
            "sucesso": sucesso,
            "custo_total": None if not sucesso else float(custo_total),
            "comprimento": comprimento,
        }

    def executar_acoes(self, path: List[Tuple[int,int]]):
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
    terreno = terreno_figura_11x11()
    agente = AgenteUtilidade(grid_size=11, inicio=(5, 1), fim=(5, 9), terreno=terreno)

    path, custo = visualizar_tempo_real_utilidade_conhecida(
        agente, step_delay=0.01, mostrar_valor=True
    )
    print("Métricas:", agente.metricas(path, custo))
    if path:
        print("Ações:", agente.executar_acoes(path))
