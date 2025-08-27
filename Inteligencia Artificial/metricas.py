from collections import deque
from typing import Dict, Set, Tuple


def celulas_acessiveis_a_partir(agent, posicao_inicial: Tuple[int, int]) -> Set[Tuple[int, int]]:
    """BFS (flood-fill) para descobrir células acessíveis a partir de uma origem.

    Resumo:
        Executa uma busca em largura (BFS) começando em `posicao_inicial`, usando
        a função `agent.vizinhos_livres(x, y)` para respeitar limites do grid e
        obstáculos do próprio agente. Retorna o conjunto de todas as posições
        (x, y) alcançáveis sem atravessar obstáculos.

    Args:
        agent: Objeto do agente que expõe o método
            `vizinhos_livres(x: int, y: int) -> Iterable[tuple[str, tuple[int,int]]]`,
            onde cada item é (direcao, (nx, ny)) apenas para vizinhos válidos/livres.
        posicao_inicial (tuple[int, int]): Posição (x, y) onde a BFS começa.

    Returns:
        set[tuple[int, int]]: Conjunto de coordenadas acessíveis a partir de
        `posicao_inicial`, incluindo a própria `posicao_inicial`.
    """
    fila = deque([posicao_inicial])
    acessiveis = {posicao_inicial}

    while fila:
        x, y = fila.popleft()
        for _dir, (nx, ny) in agent.vizinhos_livres(x, y):
            pos = (nx, ny)
            if pos not in acessiveis:
                acessiveis.add(pos)
                fila.append(pos)

    return acessiveis


def calcular_metricas(agent) -> Dict[str, object]:
    """Calcula as métricas da Etapa 2 com base no estado atual do agente.

    Resumo:
        - Determina todas as células acessíveis via BFS a partir da posição inicial.
        - Cruza com o conjunto de células visitadas pelo agente.
        - Agrega estatísticas de completude e eficiência da exploração, incluindo
          passos redundantes e (se disponíveis) backtracking e redundâncias puras.

    Args:
        agent: Objeto do agente com os seguintes atributos esperados:
            - `posicao_inicial` (tuple[int,int]) ou `position` como fallback:
                origem para o cálculo de acessibilidade.
            - `visited` (set[tuple[int,int]]): células já visitadas (se ausente,
                assume-se `set()`).
            - `passos_totais` (int, opcional): total de passos executados.
            - `passos_redundantes` (int, opcional): total de revisitas (inclui backtracking).
            - `passos_backtracking` (int, opcional): revisitas que retornam ao último nó.
            - `passos_redundantes_puros` (int, opcional): revisitas que não são backtracking.
            - Método `vizinhos_livres(x, y)` conforme usado por `celulas_acessiveis_a_partir`.

    Returns:
        dict: Dicionário com as chaves:
            - "acessiveis" (int): número de células alcançáveis a partir da origem.
            - "visitadas" (int): número de células acessíveis que foram visitadas.
            - "completude" (float): fração visitadas/acessíveis (0..1).
            - "completude_%" (float): completude em percentual (0..100).
            - "passos_totais" (int): total de passos registrados no agente (ou 0 se ausente).
            - "passos_redundantes" (int): total de revisitas (ou 0 se ausente).
            - "sucesso_desvio" (bool): True se `completude == 1.0`.
            - "passos_backtracking" (int, opcional): incluído se existir no agente.
            - "passos_redundantes_puros" (int, opcional): incluído se existir no agente.
    """
    pos_inicial = getattr(agent, "posicao_inicial", getattr(agent, "position"))
    visitadas_unicas = getattr(agent, "visited", set())

    acessiveis = celulas_acessiveis_a_partir(agent, pos_inicial)

    total_acessiveis = len(acessiveis)
    total_visitadas = len(visitadas_unicas & acessiveis)

    completude = (total_visitadas / total_acessiveis) if total_acessiveis > 0 else 0.0
    sucesso_desvio = (completude == 1.0)

    passos_totais = int(getattr(agent, "passos_totais", 0))
    passos_redundantes = int(getattr(agent, "passos_redundantes", 0))

    passos_backtracking = getattr(agent, "passos_backtracking", None)
    passos_redundantes_puros = getattr(agent, "passos_redundantes_puros", None)

    resultado = {
        "acessiveis": total_acessiveis,
        "visitadas": total_visitadas,
        "completude": round(completude, 4),
        "completude_%": round(completude * 100.0, 2),
        "passos_totais": passos_totais,
        "passos_redundantes": passos_redundantes,
        "sucesso_desvio": sucesso_desvio,
    }

    if isinstance(passos_backtracking, int):
        resultado["passos_backtracking"] = passos_backtracking
    if isinstance(passos_redundantes_puros, int):
        resultado["passos_redundantes_puros"] = passos_redundantes_puros

    return resultado
