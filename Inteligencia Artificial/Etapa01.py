import random
import matplotlib.pyplot as plt

class AgenteDeterministico:
    """Agente determinístico: sobe, direita, esquerda, desce (nessa ordem)."""
    def __init__(self, grid_size: int):
        self.grid_size = grid_size
        self.position = self.random_initial_position()
        self.perimeter_detected = set()

    def random_initial_position(self):
        x = random.randint(0, self.grid_size - 1)
        y = random.randint(0, self.grid_size - 1)
        print(f"Posição inicial: ({x}, {y})")
        return (x, y)

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

    def _move_one(self, direction: str):
        """Move uma célula na direção, respeitando limites."""
        x, y = self.position
        if direction == 'N' and y < self.grid_size - 1:
            self.position = (x, y + 1)
        elif direction == 'S' and y > 0:
            self.position = (x, y - 1)
        elif direction == 'E' and x < self.grid_size - 1:
            self.position = (x + 1, y)
        elif direction == 'O' and x > 0:
            self.position = (x - 1, y)

    def _move_to_wall(self, direction: str, *, step_idx_ref, on_step=None):
        """
        Anda na 'direction' até colidir com a parede, chamando on_step a cada passo.
        step_idx_ref: dict com {'i': <int>} para manter contador por referência.
        """
        def can_move():
            x, y = self.position
            if direction == 'N': return y < self.grid_size - 1
            if direction == 'S': return y > 0
            if direction == 'E': return x < self.grid_size - 1
            if direction == 'O': return x > 0
            return False

        while can_move():
            self._move_one(direction)
            self.perceive()
            if on_step is not None:
                step_i = step_idx_ref['i']
                try:
                    on_step(step_i, self.position, frozenset(self.perimeter_detected))
                except Exception as e:
                    print(f"on_step error: {e}")
                step_idx_ref['i'] = step_i + 1

    def explore(self, *, seed=None, on_step=None, max_steps=10_000):
        """Executa a sequência: N → E → O → S. Retorna o conjunto de paredes tocadas."""
        if seed is not None:
            random.seed(seed)

        self.perceive()

        step_idx_ref = {'i': 0}
        ordem = ['N', 'E', 'O', 'S']

        for direc in ordem:
            if step_idx_ref['i'] >= max_steps:
                break
            self._move_to_wall(direc, step_idx_ref=step_idx_ref, on_step=on_step)

        return self.perimeter_detected


def visualizar_tempo_real_etapa1_deterministico(agente, *, step_delay=0.03, seed=None, max_steps=10000):
    """Anima o agente determinístico em tempo real (sobe→direita→esquerda→desce)."""
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))
    n = agente.grid_size

    for i in range(n + 1):
        ax.plot([-.5, n-.5], [i-.5, i-.5])
        ax.plot([i-.5, i-.5], [-.5, n-.5])

    px, py = [agente.position[0]], [agente.position[1]]
    (linha_path,) = ax.plot(px, py, marker='o', label='Caminho')
    agente_sc = ax.scatter([agente.position[0]], [agente.position[1]], s=140, marker='*', label='Agente')

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_aspect('equal', adjustable='box')
    ax.invert_yaxis()
    ax.legend(loc='upper right')
    ax.set_title("Exploração determinística — iniciando...")

    fig.canvas.draw()
    plt.pause(step_delay)

    def on_step(step_idx, pos_after, paredes_tocadas):
        px.append(pos_after[0]); py.append(pos_after[1])
        linha_path.set_data(px, py)
        agente_sc.set_offsets([pos_after])

        paredes = ''.join(sorted(paredes_tocadas))
        ax.set_title(f"Etapa 1 (det.) • passo={step_idx} • pos={pos_after} • paredes={paredes or '-'}")
        plt.pause(step_delay)

    agente.explore(seed=seed, on_step=on_step, max_steps=max_steps)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    print("Iniciando exploração determinística...")
    agente = AgenteDeterministico(grid_size=10)
    visualizar_tempo_real_etapa1_deterministico(agente, step_delay=0.03, seed=42, max_steps=10000)
    print("Perímetro detectado:", agente.perimeter_detected)
    print("Detecção completa do perímetro:", len(agente.perimeter_detected) == 4)
