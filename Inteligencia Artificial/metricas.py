from collections import deque
from typing import Set, Tuple, Dict

def celulas_acessiveis_a_partir(self, posicao_inicial: Tuple[int, int]) -> Set[Tuple[int, int]]:
    """
    Faz um BFS (flood fill) a partir de `posicao_inicial` para descobrir
    TODAS as células acessíveis (dentro do grid e sem obstáculos).

    Retorna:
        Conjunto de tuplas (x, y) com todas as células acessíveis.
    """
    fila_exploracao = deque([posicao_inicial])
    celulas_alcancaveis = {posicao_inicial}

    while fila_exploracao:
        atual_x, atual_y = fila_exploracao.popleft()

        for _direcao, (prox_x, prox_y) in self.vizinhos_livres(atual_x, atual_y):
            proxima_posicao = (prox_x, prox_y)
            if proxima_posicao not in celulas_alcancaveis:
                celulas_alcancaveis.add(proxima_posicao)
                fila_exploracao.append(proxima_posicao)

    return celulas_alcancaveis

def calcular_metricas(self) -> Dict[str, float]:
    """
    Calcula as métricas pedidas para a Etapa 2.

    Retorna:
        dict com:
          - 'completude': porcentagem (0..1) de células acessíveis que foram visitadas
          - 'passos_totais': total de movimentos realizados
          - 'passos_redundantes': quantos movimentos foram para células já visitadas (backtracking etc.)
          - 'sucesso_desvio': 1.0 se completude == 1.0, senão 0.0 (Sim/Não em formato numérico)
          - 'acessiveis': quantidade de células acessíveis a partir da posição inicial
          - 'visitadas': quantidade de células únicas visitadas
    """
    acessiveis = self.celulas_acessiveis_a_partir(self.posicao_inicial)
    visitadas_unicas = self.celulas_visitadas 

    total_acessiveis = len(acessiveis)
    total_visitadas = len(visitadas_unicas)

    visitadas_e_acessiveis = len(visitadas_unicas & acessiveis)

    completude = visitadas_e_acessiveis / total_acessiveis if total_acessiveis > 0 else 0.0
    sucesso_desvio = 1.0 if completude == 1.0 else 0.0

    return {
        "completude": completude,
        "passos_totais": float(self.passos_totais),
        "passos_redundantes": float(self.passos_redundantes),
        "sucesso_desvio": sucesso_desvio,
        "acessiveis": float(total_acessiveis),
        "visitadas": float(total_visitadas),
    }
