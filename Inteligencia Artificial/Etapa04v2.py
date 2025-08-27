import heapq, math
from typing import List, Optional, Tuple, Dict

import math
import heapq
import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

Coord = Tuple[int, int]


def gerar_matriz_custos(n: int, *, seed: Optional[int] = None, pesos=(3, 2, 1)) -> List[List[int]]:
    """Gera uma matriz n×n de custos inteiros {1,2,3}.

    Resumo:
        Cria um grid de custos onde cada célula recebe aleatoriamente 1, 2 ou 3,
        de acordo com as probabilidades relativas definidas por `pesos`.

    Args:
        n (int): Dimensão do grid (n × n).
        seed (int | None, opcional): Semente para reprodutibilidade. Padrão: None.
        pesos (tuple[int,int,int], opcional): Pesos para escolhas (1,2,3). Padrão: (3,2,1).

    Returns:
        list[list[int]]: Matriz (n, n) com valores 1, 2 ou 3.
    """
    if seed is not None:
        random.seed(seed)
    custos = []
    for _ in range(n):
        linha = [random.choices([1, 2, 3], weights=pesos, k=1)[0] for _ in range(n)]
        custos.append(linha)
    return custos


def mapa_coracao(n: int = 12) -> list[list[int]]:
    """Cria um mapa de custos com miolo caro (3) e anel (2) ao redor — estilo “coração/losango”.

    Resumo:
        Inicia tudo com 1; aplica custo 3 em um losango central (raio Manhattan 2),
        e custo 2 no anel com raio 3. Mantém (0,4) e (n-1,4) com custo 1 como exemplo
        de início/fim baratos.

    Args:
        n (int, opcional): Tamanho do grid. Padrão: 12.

    Returns:
        list[list[int]]: Matriz (n, n) com 1/2/3 configurados.
    """
    m = [[1 for _ in range(n)] for _ in range(n)]
    cx, cy = 4, 5
    for y in range(n):
        for x in range(n):
            d = abs(x - cx) + abs(y - cy)
            if d <= 2:
                m[y][x] = 3
            elif d == 3:
                m[y][x] = 2
    m[0][4]   = 1
    m[n-1][4] = 1
    return m


class AgenteUtilidadePO:
    """
    Etapa 4 – V2 (parcialmente observável) sem vieses:
    - Revela apenas a célula atual e os 4-vizinhos (raio=1).
    - Planeja com A* sobre o mapa CONHECIDO.
    - Para células DESCONHECIDAS no plano, usa custo neutro (ex.: 2.0).
    - Replaneja a cada passo.
    - No final, permite comparar o custo real percorrido com o ótimo pleno (Dijkstra no true_cost).
    """
    def __init__(self, grid_size: int, start: Coord, goal: Coord,
                 *, cost_map: List[List[int]],
                 unknown_planning_cost: float = 2.0,
                 reveal_radius: int = 1):
        self.n = grid_size
        self.start: Coord = start
        self.goal:  Coord = goal

        self.true_cost: List[List[int]] = cost_map
        self.known_costs: List[List[Optional[int]]] = [[None for _ in range(self.n)] for _ in range(self.n)]

        self.unknown_planning_cost = float(unknown_planning_cost)
        self.reveal_radius = int(reveal_radius)

        self.pos: Coord = start
        self.total_true_cost: float = 0.0
        self.steps: int = 0
        self.replans: int = 0
        self.path_last: List[Coord] = []
        self.trajeto: List[Coord] = [start]

        self._revelar_vizinhanca(self.pos, raio=self.reveal_radius)

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.n and 0 <= y < self.n

    def _viz4(self, x: int, y: int):
        for dx, dy in ((0,1),(0,-1),(1,0),(-1,0)):
            nx, ny = x+dx, y+dy
            if self._in_bounds(nx, ny):
                yield nx, ny

    def _revelar_vizinhanca(self, p: Coord, incluir_atual=True, raio=1):
        """Revela custos da célula atual e dos 4-vizinhos (raio=1)."""
        x0, y0 = p
        if incluir_atual and self.known_costs[y0][x0] is None:
            self.known_costs[y0][x0] = self.true_cost[y0][x0]

        if raio <= 0:
            return

        for nx, ny in self._viz4(x0, y0):
            if self.known_costs[ny][nx] is None:
                self.known_costs[ny][nx] = self.true_cost[ny][nx]

    def _planning_cost(self, c: Optional[int]) -> float:
        """Custo usado pelo planejador: conhecido = custo real; desconhecido = custo neutro."""
        return float(c) if c is not None else self.unknown_planning_cost

    def _a_star_known(self) -> Optional[List[Coord]]:
        sx, sy = self.pos
        gx, gy = self.goal

        def h(x: int, y: int) -> float:
            return abs(x - gx) + abs(y - gy)

        g: Dict[Coord, float] = {(sx, sy): 0.0}
        bad: Dict[Coord, int]  = {(sx, sy): 0}  
        came: Dict[Coord, Coord] = {}

        open_heap: List[Tuple[float, int, Coord]] = [(h(sx, sy), 0, (sx, sy))]

        while open_heap:
            f, bval, (x, y) = heapq.heappop(open_heap)
            if (x, y) == (gx, gy):
                path = [(x, y)]
                while (x, y) != (sx, sy):
                    x, y = came[(x, y)]
                    path.append((x, y))
                path.reverse()
                return path

            for nx, ny in self._viz4(x, y):
                c = self.known_costs[ny][nx]
                step_cost = self._planning_cost(c)
                ng = g[(x, y)] + step_cost
                nb = bad[(x, y)] + (1 if c == 3 else 0)

                prev_g = g.get((nx, ny), math.inf)
                prev_b = bad.get((nx, ny), math.inf)
                if (ng < prev_g) or (math.isclose(ng, prev_g) and nb < prev_b):
                    g[(nx, ny)] = ng
                    bad[(nx, ny)] = nb
                    came[(nx, ny)] = (x, y)
                    heapq.heappush(open_heap, (ng + h(nx, ny), nb, (nx, ny)))
        return None


    def step(self) -> bool:
        """Replaneja e anda 1 passo ao longo do melhor caminho atual. False = terminou/travou."""
        if self.pos == self.goal:
            return False

        path = self._a_star_known()
        self.replans += 1
        self.path_last = path if path else []

        if not path or len(path) < 2:
            return False

        nxt = path[1]
        x2, y2 = nxt
        self.total_true_cost += float(self.true_cost[y2][x2])
        self.pos = nxt
        self.trajeto.append(nxt)
        self.steps += 1

        self._revelar_vizinhanca(self.pos, raio=self.reveal_radius)
        return True

    def run(self, max_steps=10_000, on_step=None) -> bool:
        while self.steps < max_steps and self.pos != self.goal:
            moved = self.step()
            if on_step:
                on_step(self)
            if not moved:
                break
        return self.pos == self.goal

    def _dijkstra_full_true_cost(self) -> Tuple[Optional[List[Coord]], float]:
        """Ótimo global sob conhecimento pleno (NÃO mostrado ao agente)."""
        start, goal = self.start, self.goal
        dist: Dict[Coord, float] = {start: 0.0}
        prev: Dict[Coord, Coord] = {}
        heap: List[Tuple[float, Coord]] = [(0.0, start)]
        visited = set()

        while heap:
            d, (x, y) = heapq.heappop(heap)
            if (x, y) in visited: 
                continue
            visited.add((x, y))

            if (x, y) == goal:
                path = [(x, y)]
                while (x, y) != start:
                    x, y = prev[(x, y)]
                    path.append((x, y))
                path.reverse()
                return path, d

            for nx, ny in self._viz4(x, y):
                step = float(self.true_cost[ny][nx])
                nd = d + step
                if nd < dist.get((nx, ny), math.inf):
                    dist[(nx, ny)] = nd
                    prev[(nx, ny)] = (x, y)
                    heapq.heappush(heap, (nd, (nx, ny)))

        return None, math.inf

    def metricas(self) -> Dict[str, float]:
        """Métricas da execução parcial (sem comparação com o ótimo pleno)."""
        return {
            "sucesso": self.pos == self.goal,
            "custo_total_real": float(self.total_true_cost) if self.pos == self.goal else None,
            "passos": self.steps,
            "replanejamentos": self.replans,
        }

    def metricas_comparadas(self) -> Dict[str, float]:
        """
        Compara custo REAL percorrido (parcial) com o ÓTIMO PLENO (Dijkstra no true_cost).
        Retorna também o excesso_de_custo (real - ótimo).
        """
        _, custo_otimo = self._dijkstra_full_true_cost()
        custo_real = float(self.total_true_cost) if self.pos == self.goal else math.inf
        excesso = (custo_real - custo_otimo) if math.isfinite(custo_otimo) else math.inf
        return {
            "sucesso": self.pos == self.goal,
            "custo_total_real": None if not self.pos == self.goal else custo_real,
            "custo_otimo_pleno": None if not math.isfinite(custo_otimo) else float(custo_otimo),
            "excesso_de_custo": None if (not self.pos == self.goal or not math.isfinite(custo_otimo)) else float(excesso),
            "passos": self.steps,
            "replanejamentos": self.replans,
        }


def visualizar_tempo_real_etapa4(agent: AgenteUtilidadePO, step_delay: float = 0.05, salvar_em: Optional[str] = None) -> bool:
    import math
    n = agent.n

    def known_matrix():
        m = np.zeros((n, n), dtype=int)
        for y in range(n):
            for x in range(n):
                c = agent.known_costs[y][x]
                m[y, x] = 0 if c is None else c
        return m

    cmap = ListedColormap(["#9e9e9e", "#2e7d32", "#f9a825", "#b71c1c"])

    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(known_matrix(), cmap=cmap, vmin=0, vmax=3, origin="lower")

    for i in range(n + 1):
        ax.plot([-0.5, n - 0.5], [i - 0.5, i - 0.5], color="k", linewidth=0.4)
        ax.plot([i - 0.5, i - 0.5], [-0.5, n - 0.5], color="k", linewidth=0.4)

    sx, sy = agent.start
    gx, gy = agent.goal
    ax.scatter([sx], [sy], s=120, marker='s', color='white', edgecolor='k', label='Início')
    ax.scatter([gx], [gy], s=120, marker='^', color='white', edgecolor='k', label='Fim')

    agent_sc = ax.scatter([agent.pos[0]], [agent.pos[1]], s=160, marker='*',
                          color='cyan', edgecolor='k', label='Agente')

    (traj_ln,) = ax.plot([], [], '-',  color='deepskyblue', linewidth=2, label='Trajeto')
    (plan_ln,) = ax.plot([], [], '--', color='dodgerblue', linewidth=1.5, label='Plano atual')

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc='upper right')

    def on_step(_agent: AgenteUtilidadePO):
        im.set_data(known_matrix())

        if _agent.trajeto:
            xs = [p[0] for p in _agent.trajeto]
            ys = [p[1] for p in _agent.trajeto]
            traj_ln.set_data(xs, ys)

        if _agent.path_last:
            px = [p[0] for p in _agent.path_last]
            py = [p[1] for p in _agent.path_last]
            plan_ln.set_data(px, py)
        else:
            plan_ln.set_data([], [])

        agent_sc.set_offsets([_agent.pos])
        ax.set_title(f"Passos={_agent.steps}  CustoReal={_agent.total_true_cost:.1f}  Replans={_agent.replans}")
        plt.pause(step_delay)

    ok = agent.run(on_step=on_step)

    opt_path, opt_cost = agent._dijkstra_full_true_cost()
    if opt_path:
        opx = [p[0] for p in opt_path]; opy = [p[1] for p in opt_path]
        ax.plot(opx, opy, linestyle=':', linewidth=2.5, color='white', alpha=0.95, label='Ótimo (pleno)')
        ax.legend(loc='upper right')

    titulo1 = f"{'SUCESSO' if ok else 'FIM'} | Passos={agent.steps}  CustoReal={agent.total_true_cost:.1f}  Replans={agent.replans}"
    if opt_path:
        gap = agent.total_true_cost - opt_cost if ok else float('nan')
        gap_pct = (gap / opt_cost * 100.0) if ok and opt_cost > 0 else float('nan')
        titulo2 = f"Ótimo={opt_cost:.1f}  Gap={gap:.1f} ({gap_pct:.1f}%)  Ótimo? {'Sim' if ok and abs(gap) < 1e-9 else 'Não'}"
        ax.set_title(titulo1 + "\n" + titulo2)
    else:
        ax.set_title(titulo1)

    plt.pause(max(0.7, step_delay))
    plt.ioff()
    if salvar_em:
        plt.savefig(salvar_em, bbox_inches='tight')
    plt.show()
    return ok



if __name__ == "__main__":
    n = 12
    cost_map = mapa_coracao(n)  
    start = (4, 0)
    goal  = (4, 11)

    agente = AgenteUtilidadePO(
        n, start, goal,
        cost_map=cost_map,
        unknown_planning_cost=1.0, 
        reveal_radius=1
    )


    sucesso = visualizar_tempo_real_etapa4(agente, step_delay=0.04)

    print("Parcial:", agente.metricas())
    print("Comparado ao ótimo:", agente.metricas_comparadas())
