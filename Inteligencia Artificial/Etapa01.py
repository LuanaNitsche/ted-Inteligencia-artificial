"""
Etapa 1: Agente Reativo Simples

Descrição
---------
Nesta fase, o agente não tem memória (estado interno) das posições visitadas.
Sua decisão de movimento é baseada apenas na sua percepção atual (sua posição e
se há uma parede nos limites do grid). O ambiente é um grid vazio, sem obstáculos.

Objetivo do Agente
------------------
Explorar o ambiente até ter colidido com as quatro paredes limites
(norte, sul, leste e oeste).

Métrica de Avaliação
--------------------
Detecção completa do perímetro: o agente conseguiu determinar corretamente
os limites do grid (sim/não).
"""
import random

import matplotlib.pyplot as plt


def visualizar_tempo_real_etapa1(agente, *, step_delay=0.03, max_steps=10000, seed=None):
    """Anima a exploração do agente reativo (Etapa 1) em tempo real.

    Resumo:
        Abre uma janela interativa do Matplotlib e atualiza, a cada passo, a
        posição do agente e a trilha percorrida enquanto ele explora o grid
        de forma puramente reativa até detectar todas as bordas ou atingir
        o limite de passos.

    Args:
        agente (AgenteReativoSimples):
            Instância já criada do agente reativo simples.
        step_delay (float, opcional):
            Pausa (em segundos) entre frames/atualizações da animação.
            Padrão: 0.03.
        max_steps (int, opcional):
            Limite máximo de passos para a exploração (failsafe).
            Padrão: 10000.
        seed (int | None, opcional):
            Semente aleatória para tornar a execução reprodutível. Padrão: None.

    Returns:
        None: A função apenas exibe a animação e não retorna valor.
    """

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
    ax.set_title("Exploração (Etapa 1) — iniciando...")


    fig.canvas.draw()
    plt.pause(step_delay)

    def on_step(step_idx, pos_after, paredes_tocadas):
        """Callback interno para atualizar a cena a cada passo do agente.

        Resumo:
            Atualiza a trilha, a posição do agente e o título com
            informações do passo atual.

        Args:
            step_idx (int): Índice do passo (0, 1, 2, ...).
            pos_before (tuple[int, int]): Posição (x, y) antes do movimento.
            direc (str): Direção escolhida no passo ('N', 'S', 'E', 'O').
            pos_after (tuple[int, int]): Posição (x, y) após o movimento.
            paredes_tocadas (frozenset[str]): Conjunto imutável com bordas já detectadas.

        Returns:
            None
        """
        px.append(pos_after[0]); py.append(pos_after[1])
        linha_path.set_data(px, py)
        agente_sc.set_offsets([pos_after])

        paredes = ''.join(sorted(paredes_tocadas))
        ax.set_title(f"Etapa 1 • passo={step_idx} • pos={pos_after} • paredes={paredes if paredes else '-'}")
        plt.pause(step_delay)

    agente.explore(max_steps=max_steps, verbose=False, seed=seed, on_step=on_step)

    plt.ioff()
    plt.show()


class AgenteReativoSimples:
    """Agente reativo sem memória para explorar os limites de um grid vazio.

    Resumo:
        Mantém apenas a posição atual e o conjunto de bordas (N, S, E, O)
        já detectadas. Em cada passo, escolhe aleatoriamente uma direção
        válida e se move. A exploração termina quando as quatro bordas
        foram tocadas ou quando o limite de passos é atingido.

    Atributos:
        grid_size (int): Tamanho do grid (grid_size x grid_size).
        position (tuple[int, int]): Posição atual do agente (x, y).
        perimeter_detected (set[str]): Conjunto de bordas já detectadas
            {'N', 'S', 'E', 'O'}.

    Args:
        grid_size (int):
            Tamanho do grid a ser explorado.
    """
    def __init__(self, grid_size):
        """Inicializa o agente com grid e posição aleatória.

        Resumo:
            Cria o agente, sorteia uma posição inicial válida dentro do
            grid e zera o conjunto de bordas detectadas.

        Args:
            grid_size (int): Tamanho do grid (lado).

        Returns:
            None
        """
        self.grid_size = grid_size
        self.position = self.random_initial_position()
        self.perimeter_detected = set() 

    def random_initial_position(self):
        """Sorteia e retorna uma posição inicial válida no grid.

        Resumo:
            Escolhe coordenadas x e y uniformemente em [0, grid_size-1].

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
        """Atualiza a posição do agente de acordo com a direção.

        Resumo:
            Move o agente em uma das quatro direções cardeais,
            respeitando os limites do grid (movimentos inválidos são ignorados).

        Args:
            direction (str):
                Direção do movimento: 'N' (norte), 'S' (sul),
                'E' (leste) ou 'O' (oeste).

        Returns:
            None
        """
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
        """Atualiza o conjunto de bordas detectadas pela posição atual.

        Resumo:
            Se a posição atual está em alguma borda do grid, adiciona a
            letra correspondente ao conjunto: 'N', 'S', 'E' ou 'O'.

        Args:
            None

        Returns:
            None
        """
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
        """Informa se as quatro bordas já foram detectadas.

        Resumo:
            Verifica se o conjunto `perimeter_detected` contém
            {'N', 'S', 'E', 'O'}.

        Args:
            None

        Returns:
            bool: True se as quatro bordas foram detectadas; False caso contrário.
        """
        return len(self.perimeter_detected) == 4
    
    def explore(self, max_steps=10000, verbose=False, seed=None, on_step=None):
        """Informa se as quatro bordas já foram detectadas.

        Resumo:
            Verifica se o conjunto `perimeter_detected` contém
            {'N', 'S', 'E', 'O'}.

        Args:
            None

        Returns:
            bool: True se as quatro bordas foram detectadas; False caso contrário.
        """
        if seed is not None:
            random.seed(seed)

        self.perceive()
        steps = 0

        while not self.has_detected_perimeter() and steps < max_steps:
            x, y = self.position
            acoes_validas = []

            if y < self.grid_size - 1:
                acoes_validas.append('N')
            if y > 0:
                acoes_validas.append('S')
            if x < self.grid_size - 1:
                acoes_validas.append('E')
            if x > 0:
                acoes_validas.append('O')

            direcao = random.choice(acoes_validas)

            if verbose:
                print(f"[passo {steps}] pos={self.position} -> ação={direcao}")

            pos_antes = self.position
            self.move(direcao)
            self.perceive()
            pos_depois = self.position

            if on_step is not None:
                try:
                    on_step(steps, pos_depois, frozenset(self.perimeter_detected))
                except Exception as e:
                    print(f"on_step error: {e}")

            steps += 1
            
        if verbose:
            print(f"Fim: steps={steps}, perímetro={self.perimeter_detected}, pos_final={self.position}")
        

        return self.perimeter_detected

def coletar_trilha(agent, seed=42, max_steps=10000):
    """Executa a exploração e devolve trilha, direções e histórico das bordas.

    Resumo:
        Envolve a chamada a `explore` com um callback que coleta:
        - a sequência de posições visitadas (trilha),
        - a direção escolhida em cada passo,
        - o "snapshot" do conjunto de bordas detectadas após cada passo.

    Args:
        agent (AgenteReativoSimples):
            Instância do agente a ser executado.
        seed (int, opcional):
            Semente de aleatoriedade. Padrão: 42.
        max_steps (int, opcional):
            Limite máximo de passos. Padrão: 10000.

    Returns:
        tuple[list[tuple[int,int]], list[str], list[frozenset[str]]]:
            - trilha: lista de posições (x, y) incluindo a inicial e a final,
            - direcoes: lista com as direções tomadas em cada passo,
            - paredes: lista com os conjuntos imutáveis de bordas tocadas após cada passo.
    """
    trilha = [agent.position]                 
    direcoes = []                             
    paredes = [frozenset(agent.perimeter_detected)]  

    def on_step(step, pos_before, direc, pos_after, perimeter_snapshot):
        trilha.append(pos_after)
        direcoes.append(direc)
        paredes.append(perimeter_snapshot)

    agent.explore(max_steps=max_steps, seed=seed, on_step=on_step, verbose=False)
    return trilha, direcoes, paredes



def plot_trilha(trilha, grid_size, paredes, salvar_em=None):
    """Plota a trilha percorrida pelo agente em um grid.

    Resumo:
        Desenha a grade, a linha da trilha, o marcador de início e o de fim.
        O título exibe as bordas detectadas ao final e o número de passos.

    Args:
        trilha (list[tuple[int,int]]):
            Sequência de posições (x, y) visitadas pelo agente.
        grid_size (int):
            Tamanho do grid.
        paredes (list[frozenset[str]]):
            Histórico dos conjuntos de bordas detectadas (mesmo retornado em `coletar_trilha`).
        salvar_em (str | None, opcional):
            Caminho de arquivo para salvar a figura. Se None, somente exibe. Padrão: None.

    Returns:
        None
    """
    xs = [p[0] for p in trilha]
    ys = [p[1] for p in trilha]

    plt.figure(figsize=(6, 6))

    for i in range(grid_size + 1):
        plt.plot([-.5, grid_size-.5], [i-.5, i-.5])  
        plt.plot([i-.5, i-.5], [-.5, grid_size-.5])  

    plt.plot(xs, ys, marker='o')

    if trilha:
        plt.scatter([xs[0]], [ys[0]], s=100, marker='s', label='Início')
        plt.scatter([xs[-1]], [ys[-1]], s=100, marker='*', label='Fim')

    final_perimeter = ''.join(sorted(list(paredes[-1])))
    plt.title(f"Caminho no grid {grid_size}x{grid_size} | Paredes tocadas: {final_perimeter} | Passos: {len(trilha)-1}")
    plt.xlim(-0.5, grid_size - 0.5)
    plt.ylim(-0.5, grid_size - 0.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().invert_yaxis()  
    plt.legend(loc='upper right')

    if salvar_em:
        plt.savefig(salvar_em, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    print("Iniciando exploração do grid...")
    agente = AgenteReativoSimples(grid_size=10)

    visualizar_tempo_real_etapa1(agente, step_delay=0.03, max_steps=10000, seed=42)

    print("Perímetro detectado:", agente.perimeter_detected)
    print("Detecção completa do perímetro:", agente.has_detected_perimeter())
    print("Figura salva em: etapa1_realtime.png")

