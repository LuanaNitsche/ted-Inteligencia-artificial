import heapq
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

Coord = Tuple[int, int]


def terreno_figura_11x11() -> np.ndarray:
    """
    Mapa 11×11 no formato da imagem (1/2/3).
    Índices: terreno[y, x]  (y = linha, x = coluna)
    """
    return np.array([
        [1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,3,1,1,1,1,1],
        [1,1,2,2,1,3,3,2,1,1,1],
        [1,1,2,1,3,3,3,2,1,1,1],
        [1,1,2,2,3,3,3,2,2,1,1],
        [1,1,1,2,3,1,2,2,1,1,1],
        [1,1,1,1,2,3,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1],
    ], dtype=int)


class AgenteUtilidadeV2:
    def __init__(self, grid_size: int, *, inicio: Coord, fim: Coord, terreno: np.ndarray,
                 raio_visao: int = 1):
        assert terreno.shape == (grid_size, grid_size), "Terreno deve ser (n, n)."
        self.grid_size = grid_size
        self.start: Coord = inicio
        self.end:   Coord = fim
        self.terrain = np.array(terreno, dtype=int)

        self._valida_no_grid(self.start)
        self._valida_no_grid(self.end)

        # estado de execução (parcial)
        self.pos: Coord = self.start
        self.total_true_cost: float = 0.0
        self.steps: int = 0
        self.trajeto: List[Coord] = [self.start]

        self.visited = {self.start}

        self.raio = int(raio_visao)
        self.known_costs: List[List[Optional[int]]] = [[None for _ in range(self.grid_size)]
                                                       for _ in range(self.grid_size)]
        self._revelar_vizinhanca(self.pos, raio=self.raio)

    def _valida_no_grid(self, pos: Coord):
        x, y = pos
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
            raise ValueError(f"Fora do grid: {pos}")

    def _vizinhos4(self, x: int, y: int):
        for dx, dy in ((0,1),(0,-1),(1,0),(-1,0)):
            nx, ny = x+dx, y+dy
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                yield nx, ny

    def _revelar_vizinhanca(self, p: Coord, raio: int = 1):
        x0, y0 = p
        for y in range(max(0, y0 - raio), min(self.grid_size, y0 + raio + 1)):
            for x in range(max(0, x0 - raio), min(self.grid_size, x0 + raio + 1)):
                if abs(x - x0) + abs(y - y0) <= raio:
                    if self.known_costs[y][x] is None:
                        self.known_costs[y][x] = int(self.terrain[y, x])

    def _custo_real(self, x: int, y: int) -> int:
        return int(self.terrain[y, x])

    @staticmethod
    def _manhattan(a: Coord, b: Coord) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _escolher_movimento(self) -> Optional[Coord]:
        x, y = self.pos
        gx, gy = self.end

        candidatos = []
        for nx, ny in self._vizinhos4(x, y):
            if (nx, ny) in self.visited and (nx, ny) != self.end:
                continue
            c = self.known_costs[ny][nx]
            if c is None:
                c = self._custo_real(nx, ny)
            if   (nx, ny) == (x, y+1): direc = 'S'
            elif (nx, ny) == (x+1, y): direc = 'E'
            elif (nx, ny) == (x-1, y): direc = 'O'
            else:                      direc = 'N'
            candidatos.append(((nx, ny), int(c), direc))

        if not candidatos:
            return None

        min_cost = min(c for (_, c, _) in candidatos)
        best = [(p, c, d) for (p, c, d) in candidatos if c == min_cost]

        col_now = abs(x - gx)
        align = [(p, c, d) for (p, c, d) in best if abs(p[0] - gx) < col_now]
        if align:
            target_dir = 'E' if x < gx else 'O'
            for (p, c, d) in align:
                if d == target_dir:
                    return p

            d_after = [(p, self._manhattan(p, self.end), d) for (p, _, d) in align]
            min_d = min(dd for (_, dd, _) in d_after)
            tie = [(p, d) for (p, dd, d) in d_after if dd == min_d]
            for d in (target_dir, 'S', 'E', 'O', 'N'):
                for (p, d2) in tie:
                    if d2 == d:
                        return p
            return tie[0][0]

        for (p, c, d) in best:
            if d == 'S':
                return p

        d_after = [(p, self._manhattan(p, self.end), d) for (p, _, d) in best]
        min_d = min(dd for (_, dd, _) in d_after)
        best2 = [(p, d) for (p, dd, d) in d_after if dd == min_d]

        dx, dy = gx - x, gy - y
        order = []
        if dx > 0: order += ['E']
        if dx < 0: order += ['O']
        if dy > 0: order += ['S']
        if dy < 0: order += ['N']
        for d in ('E','O','S','N'):
            if d not in order:
                order.append(d)

        for d in order:
            for (p, d2) in best2:
                if d2 == d:
                    return p

        return best2[0][0] if best2 else best[0][0]

    def step(self) -> bool:
        if self.pos == self.end:
            return False

        nxt = self._escolher_movimento()
        if nxt is None:
            return False

        x2, y2 = nxt
        self.total_true_cost += float(self._custo_real(x2, y2))
        self.pos = (x2, y2)
        self.trajeto.append(self.pos)
        self.steps += 1

        self.visited.add(self.pos)

        self._revelar_vizinhanca(self.pos, raio=self.raio)
        return True

    def run(self, max_steps=10_000, on_step=None) -> bool:
        while self.steps < max_steps and self.pos != self.end:
            moved = self.step()
            if on_step:
                on_step(self)
            if not moved:
                break
        return self.pos == self.end

    def _dijkstra_full_true_cost(self) -> Tuple[Optional[List[Coord]], float]:
        start, goal = self.start, self.end
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

            for nx, ny in self._vizinhos4(x, y):
                step = float(self._custo_real(nx, ny))
                nd = d + step
                if nd < dist.get((nx, ny), float('inf')):
                    dist[(nx, ny)] = nd
                    prev[(nx, ny)] = (x, y)
                    heapq.heappush(heap, (nd, (nx, ny)))

        return None, float('inf')

    def metricas(self) -> Dict[str, float]:
        return {
            "sucesso": self.pos == self.end,
            "custo_total_real": float(self.total_true_cost) if self.pos == self.end else None,
            "passos": self.steps,
        }

    def metricas_comparadas(self) -> Dict[str, float]:
        _, opt_cost = self._dijkstra_full_true_cost()
        real = float(self.total_true_cost) if self.pos == self.end else float('inf')
        excesso = (real - opt_cost) if np.isfinite(opt_cost) and np.isfinite(real) else None
        return {
            "sucesso": self.pos == self.end,
            "custo_total_real": None if not self.pos == self.end else real,
            "custo_otimo_pleno": None if not np.isfinite(opt_cost) else float(opt_cost),
            "excesso_de_custo": excesso if excesso is None else float(excesso),
            "passos": self.steps,
        }


def visualizar_tempo_real_utilidade_v2(
    agente: AgenteUtilidadeV2, step_delay: float = 0.03, salvar_em: Optional[str] = None
) -> bool:
    n = agente.grid_size

    cmap = ListedColormap(["#2ecc71", "#f1c40f", "#e74c3c"])
    norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5], cmap.N)

    plt.ion()
    fig, ax = plt.subplots(figsize=(7, 7))

    for i in range(n + 1):
        ax.plot([-.5, n-.5], [i-.5, i-.5], color="k", lw=0.5, alpha=0.4)
        ax.plot([i-.5, i-.5], [-.5, n-.5], color="k", lw=0.5, alpha=0.4)

    ax.imshow(
        agente.terrain, cmap=cmap, norm=norm, origin="upper",
        extent=[-0.5, n-0.5, n-0.5, -0.5]
    )

    ax.scatter([agente.start[0]], [agente.start[1]], s=140, marker='s', c="white", edgecolors="k", label='Início')
    ax.scatter([agente.end[0]],   [agente.end[1]],   s=140, marker='^', c="white", edgecolors="k", label='Fim')

    (path_ln,) = ax.plot([], [], marker='o', lw=2, c="white", label='Caminho')
    agent_sc = ax.scatter([agente.pos[0]], [agente.pos[1]], s=150, marker='*', c="white", edgecolors="k", label='Agente')

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_aspect('equal', adjustable='box')
    ax.invert_yaxis()
    ax.legend(loc='upper right')
    ax.set_title(f"V2 (raio={agente.raio}) — iniciando...")

    fig.canvas.draw(); plt.pause(step_delay)

    xs, ys = [agente.pos[0]], [agente.pos[1]]
    while True:
        moved = agente.step()
        xs.append(agente.pos[0]); ys.append(agente.pos[1])

        path_ln.set_data(xs, ys)
        agent_sc.set_offsets([agente.pos])
        ax.set_title(f"V2 (raio={agente.raio}) • passo={agente.steps} • pos={agente.pos} • custo={agente.total_true_cost:.1f}")
        plt.pause(step_delay)

        if not moved:
            break

    ok = (agente.pos == agente.end)

    opt_path, opt_cost = agente._dijkstra_full_true_cost()
    if opt_path:
        opx = [p[0] for p in opt_path]; opy = [p[1] for p in opt_path]
        ax.plot(opx, opy, linestyle=':', linewidth=2.4, color='white', alpha=0.95, label='Ótimo (pleno)')
        ax.legend(loc='upper right')

    ax.set_title(f"{'SUCESSO' if ok else 'FIM'} • passos={agente.steps} • custo_real={agente.total_true_cost:.1f}"
                 + (f"\nÓtimo={opt_cost:.1f} • gap={agente.total_true_cost - opt_cost:.1f}"
                    if ok and opt_path else ""))

    plt.pause(max(0.6, step_delay))
    plt.ioff()
    if salvar_em:
        plt.savefig(salvar_em, bbox_inches='tight')
    plt.show()

    return ok


if __name__ == "__main__":
    n = 11
    terreno = terreno_figura_11x11()
    inicio = (5, 0) 
    fim    = (5, 9)  

    agente = AgenteUtilidadeV2(
        grid_size=n, inicio=inicio, fim=fim, terreno=terreno,
        raio_visao=1
    )

    sucesso = visualizar_tempo_real_utilidade_v2(agente, step_delay=0.03)

    print("Métricas (V2):", agente.metricas())
    print("Comparado ao ótimo:", agente.metricas_comparadas())
