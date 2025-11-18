# This code should be saved as a Python file (e.g., app.py) and run from the terminal using 'streamlit run app.py'

import streamlit as st
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import random
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd
import io

# --- INÍCIO DA CORREÇÃO DE PERSISTÊNCIA: Inicialização do Session State ---
if 'simulacao_concluida' not in st.session_state:
    st.session_state['simulacao_concluida'] = False
if 'resultados_salvos' not in st.session_state:
    st.session_state['resultados_salvos'] = {}
# --- FIM DA CORREÇÃO DE PERSISTÊNCIA ---

# --- FUNÇÃO PARA CÁLCULO DA DURAÇÃO DA MÁQUINA DE LAVAR (NOVA) ---
def calcular_tempo_enchimento(volume_litros, vazao_L_por_s):
    """Calcula o tempo (em segundos) necessário para encher a máquina."""
    if vazao_L_por_s > 0:
        return volume_litros / vazao_L_por_s
    return 0

# Define the main title of the application
st.title("Simulação de Vazão em Prédio Residencial")

# Add a brief description or introduction
st.write("""
Este aplicativo realiza uma simulação de Monte Carlo para estimar a vazão de água em um prédio residencial,
considerando o comportamento fuzzy dos moradores em relação ao uso do chuveiro e outros aparelhos sanitários,
variando a temperatura ambiente.
""")

# Create a sidebar for user inputs
st.sidebar.title("Configurações da Simulação")
st.sidebar.markdown("---") # Separator

# Definição de como calcular a diferença em segundos dos horários inseridos
def diferenca_tempo_em_segundos(horario_str1, horario_str2):
    """Calcula a diferença em segundos entre dois horários em HH:MM."""
    try:
        objeto_horario1 = datetime.strptime(horario_str1, "%H:%M")
        objeto_horario2 = datetime.strptime(horario_str2, "%H:%M")
        # Garante a ordem correta para diferença positiva
        if objeto_horario1 > objeto_horario2:
            objeto_horario1, objeto_horario2 = objeto_horario2, objeto_horario1
        diferenca_tempo = objeto_horario2 - objeto_horario1
        return diferenca_tempo.total_seconds()
    except ValueError:
        st.sidebar.error("Formato de horário inválido. Use HH:MM.")
        return 0

# Passo 0: Configurando parâmetros do prédio e simulação (inputs do Streamlit)
st.sidebar.subheader("Parâmetros do Prédio")
apartamentos_por_pavimento = st.sidebar.number_input("Apartamentos por pavimento:", min_value=1, value=4, step=1)
quantidade_pavimentos = st.sidebar.number_input("Quantidade de pavimentos:", min_value=1, value=10, step=1)
quantidade_moradores_por_apartamento = st.sidebar.number_input("Quantidade de moradores por apartamento:", min_value=1, value=5, step=1)
quantidade_banheiros_por_apartamento = st.sidebar.number_input("Quantidade de banheiros por apartamento:", min_value=1, value=2, step=1)

# --- INÍCIO DA ALTERAÇÃO 1: Configuração das Regras Fuzzy por Morador ---
st.sidebar.markdown("---")
st.sidebar.subheader("Configuração das Regras Fuzzy")

# Tabela de opções para o usuário
st.sidebar.markdown("""
**Escolha a regra fuzzy para cada morador (1=Pai, 2=Mãe, 3=Filho):**
""")

# Cria uma lista para armazenar as regras escolhidas
regras_por_morador = []
regras_map_nome = {1: "Pai (Morador 1)", 2: "Mãe (Morador 2)", 3: "Filho (Morador 3+)"}
regras_default = {1: 1, 2: 2}

# Loop para criar o seletor para cada morador
for i in range(1, quantidade_moradores_por_apartamento + 1):
    # Regra padrão: Pai (1) para o 1º, Mãe (2) para o 2º, Filho (3) para os demais
    default_value = regras_default.get(i, 3)
    
    regra_escolhida = st.sidebar.selectbox(
        f"Morador {i} (Regra padrão: {regras_map_nome.get(default_value)}):",
        options=[1, 2, 3],
        index=default_value - 1, # Define o valor padrão
        key=f"regra_morador_{i}"
    )
    regras_por_morador.append(regra_escolhida)

# --- FIM DA ALTERAÇÃO 1: Configuração das Regras Fuzzy por Morador ---


st.sidebar.markdown("---") # Separator
st.sidebar.subheader("Parâmetros de Tempo")
# usuário insere os intervalos:
horario_inicio_str = st.sidebar.text_input("Horário de início do intervalo (HH:MM):", value="04:45")
horario_fim_str = st.sidebar.text_input("Horário de término do intervalo (HH:MM):", value="09:00")

# Converte os horários para segundos
intervalo_segundos = diferenca_tempo_em_segundos(horario_inicio_str, horario_fim_str)

# Solicita a duração da simulação com base no intervalo calculado
duracao_simulacao = st.sidebar.number_input(f"Duração da simulação (segundos, mínimo {intervalo_segundos:.0f}):", min_value=int(intervalo_segundos), value=max(int(intervalo_segundos), 15300), step=60)


st.sidebar.markdown("---") # Separator
st.sidebar.subheader("Parâmetros de Temperatura")
temperatura_minima = st.sidebar.number_input("Temperatura mínima (°C):", value=-1.3, step=0.1)
temperatura_maxima = st.sidebar.number_input("Temperatura máxima (°C):", value=39.2, step=0.1)

# Solicita as temperaturas a serem simuladas
temperaturas_str = st.sidebar.text_input("Temperaturas para simulação (separadas por vírgula, ex: 30, 35):", value="39.2, 29.8")

# Converte a string de temperaturas para uma lista de floats
temperaturas = [] # Initialize temperatures here
try:
    temperaturas = [float(temp.strip()) for temp in temperaturas_str.split(',') if temp.strip()]
    if not temperaturas:
        st.sidebar.warning("Nenhuma temperatura válida inserida para simulação. Usando temperaturas padrão [25.0].")
        temperaturas = [25.0] # Default if input is empty or invalid
except ValueError:
    st.sidebar.error("Entrada de temperatura inválida. Por favor, insira números separados por vírgula.")
    temperaturas = [25.0] # Default in case of error

# Exibe as temperaturas a serem simuladas (opcional, para verificação)
st.sidebar.write(f"Temperaturas a serem simuladas: {temperaturas}")

st.sidebar.markdown("---") # Separator
st.sidebar.subheader("Parâmetros da Simulação Monte Carlo")
# --- Configuração do Critério de Parada por Convergência ---
n_simulacoes_minimo = st.sidebar.number_input("Simulações mínimas para convergência:", min_value=1, value=200, step=10)
limiar_convergencia = st.sidebar.number_input("Limiar de convergência (L/s):", min_value=0.0, value=0.005, step=0.001, format="%.4f")
verificar_a_cada_n_simulacoes = st.sidebar.number_input("Verificar convergência a cada N simulações:", min_value=1, value=5, step=1)

# Número máximo de simulações (ainda é bom ter um limite superior)
n_simulacoes_maximo = st.sidebar.number_input("Máximo de simulações:", min_value=1, value=5000, step=100)

st.sidebar.markdown("---") # Separator
# Removed the checkbox for showing membership functions
# show_membership_functions = st.sidebar.checkbox("Mostrar Funções de Pertinência Fuzzy")


# Passo 2: Definindo as variáveis fuzzy
# O universo para 'inicio_do_banho' deve ir de 0 até a duração total da simulação
inicio_do_banho = ctrl.Antecedent(np.arange(0, duracao_simulacao + 1, 1), 'inicio_do_banho')
temperatura_do_ar = ctrl.Antecedent(np.arange(temperatura_minima, temperatura_maxima + 0.1, 0.1), 'temperatura_do_ar')
duracao_do_banho = ctrl.Consequent(np.arange(0, 16, 0.01), 'duracao_do_banho')

# Passo 3: Definindo funções de pertinência para as variáveis de entrada

# Calculando os limites para os conjuntos de início do banho
# O step_tempo deve ser baseado na duração total da simulação, não apenas no intervalo
if duracao_simulacao > 0:
    step_tempo = duracao_simulacao / 4 # 5 conjuntos, 4 intervalos entre eles
else:
    step_tempo = 1 # Prevent division by zero if duracao_simulacao is 0


inicio_do_banho['Very early'] = fuzz.trimf(inicio_do_banho.universe, [0, 0, step_tempo])
inicio_do_banho['Early'] = fuzz.trimf(inicio_do_banho.universe, [0, step_tempo, 2 * step_tempo])
inicio_do_banho['On time'] = fuzz.trimf(inicio_do_banho.universe, [step_tempo, 2 * step_tempo, 3 * step_tempo])
inicio_do_banho['Delayed'] = fuzz.trimf(inicio_do_banho.universe, [2 * step_tempo, 3 * step_tempo, 4 * step_tempo])
inicio_do_banho['Very delayed'] = fuzz.trimf(inicio_do_banho.universe, [3 * step_tempo, 4 * step_tempo, 4 * step_tempo])


# Calculando os limites para os conjuntos de temperatura do ar
if temperatura_maxima > temperatura_minima:
    step_temp = (temperatura_maxima - temperatura_minima) / 4 # 5 conjuntos, 4 intervalos entre eles
else:
    step_temp = 1 # Prevent division by zero

temperatura_do_ar['Very cold'] = fuzz.trimf(temperatura_do_ar.universe, [temperatura_minima, temperatura_minima, temperatura_minima + step_temp])
temperatura_do_ar['Cold'] = fuzz.trimf(temperatura_do_ar.universe, [temperatura_minima, temperatura_minima + step_temp, temperatura_minima + 2 * step_temp])
temperatura_do_ar['Pleasant'] = fuzz.trimf(temperatura_do_ar.universe, [temperatura_minima + step_temp, temperatura_minima + 2 * step_temp, temperatura_minima + 3 * step_temp])
# Corrected Hot and Very Hot ranges to ensure they are within the universe and have correct triangular/trapezoidal shapes
temperatura_do_ar['Hot'] = fuzz.trimf(temperatura_do_ar.universe, [temperatura_minima + 2 * step_temp, temperatura_minima + 3 * step_temp, temperatura_maxima])
temperatura_do_ar['Very hot'] = fuzz.trimf(temperatura_do_ar.universe, [temperatura_minima + 3 * step_temp, temperatura_maxima, temperatura_maxima])


# Passo 4: Definindo funções de pertinência para a variável de saída (Mantido como antes)
duracao_do_banho['No shower'] = fuzz.trapmf(duracao_do_banho.universe, [0, 0, 3, 3.01])
duracao_do_banho['Very fast'] = fuzz.trimf(duracao_do_banho.universe, [3.02, 3.02, 5])
duracao_do_banho['Fast'] = fuzz.trimf(duracao_do_banho.universe, [3, 5, 10])
duracao_do_banho['Normal'] = fuzz.trimf(duracao_do_banho.universe, [5, 10, 15])
duracao_do_banho['Long'] = fuzz.trimf(duracao_do_banho.universe, [10, 15, 15])


# Passo 5: Regras fuzzy (Mantido como antes) - sem alteração nos conjuntos de regras
morador_1_rules = [
    ctrl.Rule(temperatura_do_ar['Very cold'] & inicio_do_banho['Very early'], duracao_do_banho['Very fast']),
    ctrl.Rule(temperatura_do_ar['Cold'] & inicio_do_banho['Very early'], duracao_do_banho['Fast']),
    ctrl.Rule(temperatura_do_ar['Pleasant'] & inicio_do_banho['Very early'], duracao_do_banho['Normal']),
    ctrl.Rule(temperatura_do_ar['Hot'] & inicio_do_banho['Very early'], duracao_do_banho['Normal']),
    ctrl.Rule(temperatura_do_ar['Very hot'] & inicio_do_banho['Very early'], duracao_do_banho['Long']),

    ctrl.Rule(temperatura_do_ar['Very cold'] & inicio_do_banho['Early'], duracao_do_banho['Very fast']),
    ctrl.Rule(temperatura_do_ar['Cold'] & inicio_do_banho['Early'], duracao_do_banho['Fast']),
    ctrl.Rule(temperatura_do_ar['Pleasant'] & inicio_do_banho['Early'], duracao_do_banho['Normal']),
    ctrl.Rule(temperatura_do_ar['Hot'] & inicio_do_banho['Early'], duracao_do_banho['Normal']),
    ctrl.Rule(temperatura_do_ar['Very hot'] & inicio_do_banho['Early'], duracao_do_banho['Long']),

    ctrl.Rule(temperatura_do_ar['Very cold'] & inicio_do_banho['On time'], duracao_do_banho['Very fast']),
    ctrl.Rule(temperatura_do_ar['Cold'] & inicio_do_banho['On time'], duracao_do_banho['Fast']),
    ctrl.Rule(temperatura_do_ar['Pleasant'] & inicio_do_banho['On time'], duracao_do_banho['Fast']),
    ctrl.Rule(temperatura_do_ar['Hot'] & inicio_do_banho['On time'], duracao_do_banho['Normal']),
    ctrl.Rule(temperatura_do_ar['Very hot'] & inicio_do_banho['On time'], duracao_do_banho['Normal']),

    ctrl.Rule(temperatura_do_ar['Very cold'] & inicio_do_banho['Delayed'], duracao_do_banho['No shower']),
    ctrl.Rule(temperatura_do_ar['Cold'] & inicio_do_banho['Delayed'], duracao_do_banho['No shower']),
    ctrl.Rule(temperatura_do_ar['Pleasant'] & inicio_do_banho['Delayed'], duracao_do_banho['No shower']),
    ctrl.Rule(temperatura_do_ar['Hot'] & inicio_do_banho['Delayed'], duracao_do_banho['Very fast']),
    ctrl.Rule(temperatura_do_ar['Very hot'] & inicio_do_banho['Delayed'], duracao_do_banho['Very fast']),

    ctrl.Rule(temperatura_do_ar['Very cold'] & inicio_do_banho['Very delayed'], duracao_do_banho['No shower']),
    ctrl.Rule(temperatura_do_ar['Cold'] & inicio_do_banho['Very delayed'], duracao_do_banho['No shower']),
    ctrl.Rule(temperatura_do_ar['Pleasant'] & inicio_do_banho['Very delayed'], duracao_do_banho['No shower']),
    ctrl.Rule(temperatura_do_ar['Hot'] & inicio_do_banho['Very delayed'], duracao_do_banho['No shower']),
    ctrl.Rule(temperatura_do_ar['Very hot'] & inicio_do_banho['Very delayed'], duracao_do_banho['No shower'])
]


morador_2_rules = [
    ctrl.Rule(temperatura_do_ar['Very cold'] & inicio_do_banho['Very early'], duracao_do_banho['Very fast']),
    ctrl.Rule(temperatura_do_ar['Cold'] & inicio_do_banho['Very early'], duracao_do_banho['Fast']),
    ctrl.Rule(temperatura_do_ar['Pleasant'] & inicio_do_banho['Very early'], duracao_do_banho['Fast']),
    ctrl.Rule(temperatura_do_ar['Hot'] & inicio_do_banho['Very early'], duracao_do_banho['Normal']),
    ctrl.Rule(temperatura_do_ar['Very hot'] & inicio_do_banho['Very early'], duracao_do_banho['Normal']),

    ctrl.Rule(temperatura_do_ar['Very cold'] & inicio_do_banho['Early'], duracao_do_banho['Very fast']),
    ctrl.Rule(temperatura_do_ar['Cold'] & inicio_do_banho['Early'], duracao_do_banho['Fast']),
    ctrl.Rule(temperatura_do_ar['Pleasant'] & inicio_do_banho['Early'], duracao_do_banho['Fast']),
    ctrl.Rule(temperatura_do_ar['Hot'] & inicio_do_banho['Early'], duracao_do_banho['Normal']),
    ctrl.Rule(temperatura_do_ar['Very hot'] & inicio_do_banho['Early'], duracao_do_banho['Long']),

    ctrl.Rule(temperatura_do_ar['Very cold'] & inicio_do_banho['On time'], duracao_do_banho['Very fast']),
    ctrl.Rule(temperatura_do_ar['Cold'] & inicio_do_banho['On time'], duracao_do_banho['Fast']),
    ctrl.Rule(temperatura_do_ar['Pleasant'] & inicio_do_banho['On time'], duracao_do_banho['Fast']),
    ctrl.Rule(temperatura_do_ar['Hot'] & inicio_do_banho['On time'], duracao_do_banho['Normal']),
    ctrl.Rule(temperatura_do_ar['Very hot'] & inicio_do_banho['On time'], duracao_do_banho['Normal']),

    ctrl.Rule(temperatura_do_ar['Very cold'] & inicio_do_banho['Delayed'], duracao_do_banho['No shower']),
    ctrl.Rule(temperatura_do_ar['Cold'] & inicio_do_banho['Delayed'], duracao_do_banho['No shower']),
    ctrl.Rule(temperatura_do_ar['Pleasant'] & inicio_do_banho['Delayed'], duracao_do_banho['No shower']),
    ctrl.Rule(temperatura_do_ar['Hot'] & inicio_do_banho['Delayed'], duracao_do_banho['Very fast']),
    ctrl.Rule(temperatura_do_ar['Very hot'] & inicio_do_banho['Delayed'], duracao_do_banho['Very fast']),

    ctrl.Rule(temperatura_do_ar['Very cold'] & inicio_do_banho['Very delayed'], duracao_do_banho['No shower']),
    ctrl.Rule(temperatura_do_ar['Cold'] & inicio_do_banho['Very delayed'], duracao_do_banho['No shower']),
    ctrl.Rule(temperatura_do_ar['Pleasant'] & inicio_do_banho['Very delayed'], duracao_do_banho['No shower']),
    ctrl.Rule(temperatura_do_ar['Hot'] & inicio_do_banho['Very delayed'], duracao_do_banho['No shower']),
    ctrl.Rule(temperatura_do_ar['Very hot'] & inicio_do_banho['Very delayed'], duracao_do_banho['No shower'])
]

morador_3_rules = [
    ctrl.Rule(temperatura_do_ar['Very cold'] & inicio_do_banho['Very early'], duracao_do_banho['No shower']),
    ctrl.Rule(temperatura_do_ar['Cold'] & inicio_do_banho['Very early'], duracao_do_banho['No shower']),
    ctrl.Rule(temperatura_do_ar['Pleasant'] & inicio_do_banho['Very early'], duracao_do_banho['Fast']),
    ctrl.Rule(temperatura_do_ar['Hot'] & inicio_do_banho['Very early'], duracao_do_banho['Long']),
    ctrl.Rule(temperatura_do_ar['Very hot'] & inicio_do_banho['Very early'], duracao_do_banho['Long']),

    ctrl.Rule(temperatura_do_ar['Very cold'] & inicio_do_banho['Early'], duracao_do_banho['No shower']),
    ctrl.Rule(temperatura_do_ar['Cold'] & inicio_do_banho['Early'], duracao_do_banho['No shower']),
    ctrl.Rule(temperatura_do_ar['Pleasant'] & inicio_do_banho['Early'], duracao_do_banho['Normal']),
    ctrl.Rule(temperatura_do_ar['Hot'] & inicio_do_banho['Early'], duracao_do_banho['Normal']),
    ctrl.Rule(temperatura_do_ar['Very hot'] & inicio_do_banho['Early'], duracao_do_banho['Normal']),

    ctrl.Rule(temperatura_do_ar['Very cold'] & inicio_do_banho['On time'], duracao_do_banho['No shower']),
    ctrl.Rule(temperatura_do_ar['Cold'] & inicio_do_banho['On time'], duracao_do_banho['No shower']),
    ctrl.Rule(temperatura_do_ar['Pleasant'] & inicio_do_banho['On time'], duracao_do_banho['Normal']),
    ctrl.Rule(temperatura_do_ar['Hot'] & inicio_do_banho['On time'], duracao_do_banho['Long']),
    ctrl.Rule(temperatura_do_ar['Very hot'] & inicio_do_banho['On time'], duracao_do_banho['Long']),

    # --- LINHA 268 ORIGINALMENTE COM ERRO: CORRIGIDA DE 'temperatura_ar' PARA 'temperatura_do_ar' ---
    ctrl.Rule(temperatura_do_ar['Very cold'] & inicio_do_banho['Delayed'], duracao_do_banho['No shower']),
    ctrl.Rule(temperatura_do_ar['Cold'] & inicio_do_banho['Delayed'], duracao_do_banho['No shower']),
    ctrl.Rule(temperatura_do_ar['Pleasant'] & inicio_do_banho['Delayed'], duracao_do_banho['No shower']),
    ctrl.Rule(temperatura_do_ar['Hot'] & inicio_do_banho['Delayed'], duracao_do_banho['Very fast']),
    ctrl.Rule(temperatura_do_ar['Very hot'] & inicio_do_banho['Delayed'], duracao_do_banho['Very fast']),

    ctrl.Rule(temperatura_do_ar['Very cold'] & inicio_do_banho['Very delayed'], duracao_do_banho['No shower']),
    ctrl.Rule(temperatura_do_ar['Cold'] & inicio_do_banho['Very delayed'], duracao_do_banho['No shower']),
    ctrl.Rule(temperatura_do_ar['Pleasant'] & inicio_do_banho['Very delayed'], duracao_do_banho['No shower']),
    ctrl.Rule(temperatura_do_ar['Hot'] & inicio_do_banho['Very delayed'], duracao_do_banho['No shower']),
    ctrl.Rule(temperatura_do_ar['Very hot'] & inicio_do_banho['Very delayed'], duracao_do_banho['No shower'])
]

# Mapeia o TIPO de morador para o conjunto de regras
rules_map = {
    1: morador_1_rules, # Pai
    2: morador_2_rules, # Mãe
    3: morador_3_rules  # Filho
}

# Cria UMA lista de simuladores (três, um para cada conjunto de regras)
simuladores = []
for tipo_regra in range(1, 4):
    regras_morador_atual = rules_map[tipo_regra]
    control_system_atual = ctrl.ControlSystem(regras_morador_atual)
    simulador_morador_atual = ctrl.ControlSystemSimulation(control_system_atual)
    simuladores.append(simulador_morador_atual)

# Vazões dos aparelhos (L/s) e durações fixas (em segundos)
chuveiro = 0.12
vaso = 0.15
lavatorio = 0.07
pia = 0.10
duracao_vaso = 60
duracao_lavatorio = 30
duracao_pia = 40

# --- LÓGICA DO VASO: INÍCIO 90s ANTES DO BANHO (CORREÇÃO DE LÓGICA) ---
# Se o vaso dura 60s e deve começar 90s antes do banho, o fim do vaso é 30s antes do banho.
# O início do vaso é: inicio_banho - 90
TEMPO_ANTES_DO_BANHO_PARA_INICIO_VASO = 90
# --- FIM DA CORREÇÃO DE LÓGICA ---

# --- INÍCIO DA ALTERAÇÃO MÁQUINA DE LAVAR (V2) ---
# Vazão para enchimento da máquina de lavar (L/s) - Constante
vazao_enchimento_mlr = 0.135 

# Volumes dos modelos de máquina de lavar (L) - Três modelos
volumes_maquina_lavar = {
    'pequena': 174,
    'media': 202,
    'grande': 260
}
# --- FIM DA ALTERAÇÃO MÁQUINA DE LAVAR (V2) ---


# Cria a lista de todos os moradores do prédio com suas características e apartamento
total_apartamentos = apartamentos_por_pavimento * quantidade_pavimentos
total_moradores_predio = total_apartamentos * quantidade_moradores_por_apartamento

st.write(f"Calculando moradores para {total_apartamentos} apartamentos com {quantidade_moradores_por_apartamento} moradores por apartamento...")

moradores_predio = []

# Creating the list of all residents, identified by apartment, type, and rule chosen by the user
for apt_num in range(1, total_apartamentos + 1):
    moradores_no_apartamento = []
    for morador_num_no_apt in range(1, quantidade_moradores_por_apartamento + 1):
        # Determine the resident type based on the user's choice
        tipo_regra_escolhida = regras_por_morador[morador_num_no_apt - 1]
        
        moradores_no_apartamento.append({
            'nome': f'Morador {morador_num_no_apt}', # Identificador no apartamento
            'tipo_regra': tipo_regra_escolhida, # 1, 2 ou 3
            'apartamento': apt_num,
            'usa_pia': False, # Initialize all as False for kitchen sink
            'usa_mlr': False, # Máquina de Lavar
            'fim_pia_simulacao': 0, # Variável para armazenar o fim da pia na simulação (necessário para a lógica da MLR)
            'inicio_banho_sorteado': 0 # NOVO: Armazena o tempo sorteado para ordenação
        })
    moradores_predio.extend(moradores_no_apartamento)

# Randomly select one resident per apartment to use the sink and one for the washing machine
# Group residents by apartment for easier selection
moradores_por_apartamento_dict = {}
for morador in moradores_predio:
    if morador['apartamento'] not in moradores_por_apartamento_dict:
        moradores_por_apartamento_dict[morador['apartamento']] = []
    moradores_por_apartamento_dict[morador['apartamento']].append(morador)

# Randomly select one resident to use the sink and one for the washing machine in each apartment
for apt_num, lista_moradores_apt in moradores_por_apartamento_dict.items():
    if lista_moradores_apt: # Ensures there are residents in the apartment
        # Kitchen Sink Selection (existing logic)
        morador_usa_pia = random.choice(lista_moradores_apt)
        morador_usa_pia['usa_pia'] = True
        
        # Seleção da Máquina de Lavar
        # Select one resident for the washing machine (it can be the same as the one who uses the sink)
        morador_usa_mlr = random.choice(lista_moradores_apt)
        morador_usa_mlr['usa_mlr'] = True

# The final list of residents for the simulation is 'moradores_predio'
st.write(f"Lista criada com {len(moradores_predio)} moradores para todo o prédio.")

# Calculate the total number of bathrooms in the building (assuming each apartment has the same quantity)
total_banheiros_predio = total_apartamentos * quantidade_banheiros_por_apartamento

# Variável para armazenar o relatório da última simulação da primeira temperatura
relatorio_simulacao = []


# Main loop over each temperature to be simulated - Executes only if there are valid temperatures
if temperaturas and duracao_simulacao > 0 and total_moradores_predio > 0:
    if st.sidebar.button("Executar Simulação"):
        st.info("Iniciando simulação de Monte Carlo...")
        # Use st.progress to show the overall simulation progress
        progress_bar = st.progress(0)
        total_temperaturas_simular = len(temperaturas)
        temp_counter = 0
        
        # Limpa os estados salvos ao iniciar uma nova simulação
        st.session_state['resultados_salvos'] = {}
        st.session_state['simulacao_concluida'] = False

        # Dictionary to store results (flow rate, statistics, etc.) for each temperature
        resultados_por_temperatura = {}

        for temperatura_atual in temperaturas:
            st.subheader(f"Simulação para Temperatura: {temperatura_atual}°C")
            st.info(f"Executando simulações para Temperatura: {temperatura_atual}°C")

            # List to store the flow rate time series of each Monte Carlo simulation for this temperature
            resultados_vazao_temperatura = []
            max_p95_anterior = float('inf') # Initialize with infinity to ensure the first difference is large
            convergencia_atingida = False
            
            # Limpa o relatório a cada nova temperatura
            relatorio_simulacao_atual = []


            # Monte Carlo simulation loop
            # We use n_simulacoes_maximo as an upper limit, but the loop can stop earlier due to convergence
            for i in range(n_simulacoes_maximo):
                # Inicializa o relatório para esta iteração. Só será salvo se for a última.
                relatorio_simulacao_temp = []
                
                # Update the progress bar (approximate)
                progress = (temp_counter / total_temperaturas_simular) + (i / n_simulacoes_maximo / total_temperaturas_simular)
                progress_bar.progress(min(progress, 1.0)) # Ensures it doesn't exceed 100%

                # Initialize the flow rate time series for this simulation (total building flow rate per second)
                vazao_simulacao = np.zeros(duracao_simulacao)

                # Initialize the occupation state of ALL BATHROOMS IN THE BUILDING.
                banheiros_livres_em = np.zeros(total_banheiros_predio) # Stores the second when each bathroom will be free

                
                # --- Sorteia o horário e preenche o campo para ordenação ---
                for m in moradores_predio:
                    m['inicio_banho_sorteado'] = random.randint(0, duracao_simulacao - 1)
                
                # --- Ordenar os moradores pelo horário sorteado (SOLUÇÃO 1) ---
                moradores_predio_ordenado = sorted(moradores_predio, key=lambda m: m['inicio_banho_sorteado'])


                # --- Simulation logic for each resident and bathroom usage (coleta o relatório) ---
                # Itera sobre a lista ORDENADA
                for m in moradores_predio_ordenado:
                    # O horário de início do banho é o sorteado, agora usado em ordem
                    inicio_banho = m['inicio_banho_sorteado']

                    # Determine which fuzzy simulator to use based on the 'tipo_regra' attribute set by the user's choice
                    tipo_regra_num = m['tipo_regra']
                    simulador_morador_atual = simuladores[tipo_regra_num - 1]
                    
                    # Identificação para o relatório
                    id_morador = f"{m['nome']} (Apto {m['apartamento']})"

                    # Use the fuzzy simulator with the current temperature and shower start time
                    try:
                        # Ensure inputs are within the defined universe
                        clipped_inicio_banho = np.clip(inicio_banho, inicio_do_banho.universe.min(), inicio_do_banho.universe.max())
                        clipped_temperatura_atual = np.clip(temperatura_atual, temperatura_do_ar.universe.min(), temperatura_do_ar.universe.max())

                        simulador_morador_atual.input['inicio_do_banho'] = clipped_inicio_banho
                        simulador_morador_atual.input['temperatura_do_ar'] = clipped_temperatura_atual
                        simulador_morador_atual.compute()
                        dur_banho_minutos = simulador_morador_atual.output['duracao_do_banho']
                        dur_banho_segundos = int(dur_banho_minutos * 60) # Shower duration in seconds
                        fim_banho = inicio_banho + dur_banho_segundos # Fim do banho é usado para o cálculo do início da MLR

                        # --- LOG: Duração do Banho e Horário Inicial ---
                        regra_nome = regras_map_nome.get(m['tipo_regra'])
                        relatorio_simulacao_temp.append(f"[{id_morador}] (Regra: {regra_nome}, Temp: {temperatura_atual}°C) - Horário inicial sorteado: {inicio_banho}s. Duração fuzzy: {dur_banho_minutos:.2f} min ({dur_banho_segundos}s).")
                        
                        
                        # --- INÍCIO DA LÓGICA MÁQUINA DE LAVAR (V2: Escolha e Início Condicional) ---
                        usa_mlr_na_simulacao = False
                        inicio_mlr_clamped = 0
                        fim_mlr_clamped = 0
                        nome_volume_escolhido = ""
                        duracao_enchimento_mlr = 0

                        # Check if the resident is selected to use the machine in this apartment
                        if m['usa_mlr']:
                            # Check the fuzzy membership for 'On time' or anterior (Early, Very early)
                            pertinencia_delayed = fuzz.interp_membership(inicio_do_banho.universe, inicio_do_banho['Delayed'].mf, clipped_inicio_banho)
                            pertinencia_very_delayed = fuzz.interp_membership(inicio_do_banho.universe, inicio_do_banho['Very delayed'].mf, clipped_inicio_banho)
                            
                            # Rule: Use the machine if the shower start is NOT primarily Delayed or Very Delayed
                            if pertinencia_delayed + pertinencia_very_delayed < 0.5:
                                usa_mlr_na_simulacao = True
                                
                                # 1. Escolher o modelo da máquina aleatoriamente
                                nome_volume_escolhido, volume_escolhido = random.choice(list(volumes_maquina_lavar.items()))
                                
                                # 2. Calcular o tempo de enchimento (duração da vazão)
                                duracao_enchimento_mlr = calcular_tempo_enchimento(volume_escolhido, vazao_enchimento_mlr)
                                
                                # --- LOG: Máquina de Lavar Escolhida ---
                                relatorio_simulacao_temp.append(f"[{id_morador}] **SORTEADO P/ MLR.** Volume: {nome_volume_escolhido} ({volume_escolhido}L). Duração enchimento: {duracao_enchimento_mlr:.0f}s.")
                                
                                # 3. Determinar o início condicional (depende do uso da pia)
                                inicio_mlr = fim_banho + 120 # 120s após o banho (Default - corrigido depois)


                            else:
                                relatorio_simulacao_temp.append(f"[{id_morador}] **SORTEADO P/ MLR, mas desiste.** (Horário de banho muito atrasado).")
                        # --- FIM DA LÓGICA MÁQUINA DE LAVAR (V2) ---


                        # Calculate bathroom occupation intervals for toilet, shower, and sink
                        # --- CORREÇÃO DO VASO (SOLUÇÃO 2) ---
                        # Vaso começa 90s antes do banho.
                        inicio_vaso = max(0, inicio_banho - TEMPO_ANTES_DO_BANHO_PARA_INICIO_VASO) 
                        fim_vaso = inicio_vaso + duracao_vaso
                        # --- FIM DA CORREÇÃO ---
                        
                        inicio_lavatorio = fim_banho + 30 # 30 segundos após o banho
                        fim_lavatorio = inicio_lavatorio + duracao_lavatorio

                        # Ocupação começa na hora que o vaso inicia e termina quando o lavatório termina
                        intervalo_ocupacao_inicio = max(0, inicio_vaso)
                        intervalo_ocupacao_fim = min(duracao_simulacao, fim_lavatorio)


                        # --- INÍCIO DA LÓGICA DE FILA DE ESPERA (ALTERAÇÃO PRINCIPAL) ---
                        apt_num = m['apartamento']
                        primeiro_indice_banheiro_apt = (apt_num - 1) * quantidade_banheiros_por_apartamento
                        ultimo_indice_banheiro_apt = primeiro_indice_banheiro_apt + quantidade_banheiros_por_apartamento - 1

                        # 1. Encontrar o banheiro que ficará livre mais cedo dentro do apartamento
                        # Isso garante que SEMPRE haverá um banheiro a ser usado (mesmo que com espera)
                        banheiros_do_apt = banheiros_livres_em[primeiro_indice_banheiro_apt : ultimo_indice_banheiro_apt + 1]

                        # Encontra o índice LOCAL do banheiro que estará livre mais cedo (ou agora)
                        banheiro_usado_idx_local = np.argmin(banheiros_do_apt)
                        # Converte para o índice GLOBAL
                        banheiro_disponivel_indice_global = primeiro_indice_banheiro_apt + banheiro_usado_idx_local
                        tempo_liberacao_banheiro = banheiros_livres_em[banheiro_disponivel_indice_global]


                        # 2. Determinar o Início Real da Rotina (Espera ou Imediato)
                        # O uso real só pode começar após o tempo de liberação do banheiro E não antes do tempo sorteado
                        tempo_inicio_rotina_real = max(intervalo_ocupacao_inicio, tempo_liberacao_banheiro)
                        
                        # 3. Se houve espera, recalcular todos os tempos
                        if tempo_inicio_rotina_real > intervalo_ocupacao_inicio:
                            tempo_espera = tempo_inicio_rotina_real - intervalo_ocupacao_inicio
                            
                            # Atualiza a variável base do banho para o novo início
                            inicio_banho = tempo_inicio_rotina_real
                            
                            # Recalcula as dependências com o novo início do banho
                            inicio_vaso = max(0, inicio_banho - TEMPO_ANTES_DO_BANHO_PARA_INICIO_VASO) 
                            fim_vaso = inicio_vaso + duracao_vaso
                            fim_banho = inicio_banho + dur_banho_segundos
                            inicio_lavatorio = fim_banho + 30
                            fim_lavatorio = inicio_lavatorio + duracao_lavatorio
                            
                            # Atualiza os intervalos de ocupação
                            intervalo_ocupacao_inicio = inicio_vaso
                            intervalo_ocupacao_fim = min(duracao_simulacao, fim_lavatorio)
                            
                            relatorio_simulacao_temp.append(f"[{id_morador}] **AGUARDA {tempo_espera:.0f}s** (Banheiro {banheiro_usado_idx_local + 1} livre em {tempo_liberacao_banheiro:.0f}s). Novo Início: {tempo_inicio_rotina_real:.0f}s.")
                        else:
                            # Não houve espera, usa no tempo sorteado
                            relatorio_simulacao_temp.append(f"[{id_morador}] **USA BANHEIRO {banheiro_usado_idx_local + 1}** (Livre em: {intervalo_ocupacao_fim:.0f}s).")

                        # 4. Ocupa o banheiro com o novo tempo de liberação
                        # (O tempo final de ocupação é o mesmo, mas o início pode ter sido atrasado)
                        banheiros_livres_em[banheiro_disponivel_indice_global] = intervalo_ocupacao_fim
                        # --- FIM DA LÓGICA DE FILA DE ESPERA ---


                        # --- LÓGICA DE VAZÃO (usa as variáveis que podem ter sido ajustadas) ---

                        # Vaso
                        inicio_vaso_clamped = int(max(0, inicio_vaso))
                        fim_vaso_clamped = int(min(duracao_simulacao, fim_vaso))
                        if fim_vaso_clamped > inicio_vaso_clamped:
                            vazao_simulacao[inicio_vaso_clamped:fim_vaso_clamped] += vaso
                            relatorio_simulacao_temp.append(f"  - Vaso ({vaso}L/s): {inicio_vaso_clamped}s a {fim_vaso_clamped}s. Fim Vaso: {fim_vaso_clamped}s.")

                        # Chuveiro
                        inicio_banho_clamped = int(max(0, inicio_banho))
                        fim_banho_clamped = int(min(duracao_simulacao, inicio_banho + dur_banho_segundos))
                        if fim_banho_clamped > inicio_banho_clamped:
                            vazao_simulacao[inicio_banho_clamped : fim_banho_clamped] += chuveiro
                            relatorio_simulacao_temp.append(f"  - Chuveiro ({chuveiro}L/s): {inicio_banho_clamped}s a {fim_banho_clamped}s.")

                        # Lavatório
                        inicio_lavatorio_clamped = int(max(0, inicio_lavatorio))
                        fim_lavatorio_clamped = int(min(duracao_simulacao, fim_lavatorio))
                        if fim_lavatorio_clamped > inicio_lavatorio_clamped:
                            vazao_simulacao[inicio_lavatorio_clamped:fim_lavatorio_clamped] += lavatorio
                            relatorio_simulacao_temp.append(f"  - Lavatório ({lavatorio}L/s): {inicio_lavatorio_clamped}s a {fim_lavatorio_clamped}s.")


                        # Pia de Cozinha (O início da Pia também deve ser recalculado se o banho atrasou)
                        if m['usa_pia']:
                            inicio_pia = fim_banho + 120 # 120s após o NOVO fim do banho
                            fim_pia = inicio_pia + duracao_pia
                            inicio_pia_clamped = int(max(0, inicio_pia))
                            fim_pia_clamped = int(min(duracao_simulacao, fim_pia))
                            if fim_pia_clamped > inicio_pia_clamped:
                                vazao_simulacao[inicio_pia_clamped:fim_pia_clamped] += pia
                                m['fim_pia_simulacao'] = fim_pia_clamped
                                relatorio_simulacao_temp.append(f"  - Pia Cozinha ({pia}L/s): {inicio_pia_clamped}s a {fim_pia_clamped}s.")
                            else:
                                m['fim_pia_simulacao'] = 0 
                                relatorio_simulacao_temp.append(f"  - Pia Cozinha: Não usada (tempo fora do intervalo).")
                        else:
                            m['fim_pia_simulacao'] = 0 
                                
                                
                        # --- MLR (Ajuste Final e Vazão) ---
                        if usa_mlr_na_simulacao:
                            # Recalcula o início da MLR com base no NOVO fim_banho/fim_pia
                            if m['usa_pia'] and m['fim_pia_simulacao'] > 0:
                                inicio_mlr = m['fim_pia_simulacao'] + 30 
                                motivo_inicio = "30s após Pia"
                            else:
                                inicio_mlr = fim_banho + 120 
                                motivo_inicio = "120s após Banho"
                                
                            # Re-Clampa os limites
                            inicio_mlr_clamped = int(max(0, inicio_mlr))
                            fim_mlr_clamped = int(min(duracao_simulacao, inicio_mlr + int(duracao_enchimento_mlr)))

                            # Adiciona a vazão
                            if fim_mlr_clamped > inicio_mlr_clamped:
                                vazao_simulacao[inicio_mlr_clamped:fim_mlr_clamped] += vazao_enchimento_mlr
                                relatorio_simulacao_temp.append(f"[{id_morador}] **USA MLR ({nome_volume_escolhido}).** Início: {motivo_inicio}. Vazão ({vazao_enchimento_mlr}L/s): {inicio_mlr_clamped}s a {fim_mlr_clamped}s.")
                            else:
                                relatorio_simulacao_temp.append(f"[{id_morador}] MLR Cancelada (tempo fora do intervalo).")
                            
                        

                    except ValueError as e:
                            st.warning(f"Erro na computação fuzzy para morador {m['nome']} do apto {m['apartamento']} na temperatura {temperatura_atual}°C: {e}")


                # Add the flow rate time series of this simulation to the results list for this temperature
                resultados_vazao_temperatura.append(vazao_simulacao)

                # --- Check for Convergence Criterion ---
                if (i + 1) % verificar_a_cada_n_simulacoes == 0 and (i + 1) >= n_simulacoes_minimo:
                    resultados_cumulativos = np.array(resultados_vazao_temperatura)
                    p95_cumulativo_ts = np.percentile(resultados_cumulativos, 95, axis=0)
                    max_p95_cumulativo = np.max(p95_cumulativo_ts)

                    if abs(max_p95_cumulativo - max_p95_anterior) < limiar_convergencia:
                        st.success(f"Convergência atingida após {i + 1} simulações para {temperatura_atual}°C.")
                        convergencia_atingida = True
                    else:
                        max_p95_anterior = max_p95_cumulativo
                
                # CORREÇÃO DE PERFORMANCE: Salva o relatório e dá o BREAK
                if (convergencia_atingida or (i + 1) == n_simulacoes_maximo):
                    # Só salva se for a primeira temperatura e 1 apartamento (para evitar logs enormes)
                    if temperatura_atual == temperaturas[0] and total_apartamentos == 1:
                        relatorio_simulacao_atual = relatorio_simulacao_temp.copy()
                
                    if convergencia_atingida:
                        break
                
                    if i + 1 == n_simulacoes_maximo and not convergencia_atingida:
                        st.warning(f"Número máximo de simulações ({n_simulacoes_maximo}) atingido sem convergência para {temperatura_atual}°C.")
                        break


            # Após todas as simulações, se for a primeira temperatura e 1 apartamento, salva o relatório final
            if temperatura_atual == temperaturas[0] and total_apartamentos == 1:
                relatorio_simulacao = relatorio_simulacao_atual.copy()


            # After all Monte Carlo simulations for this temperature (until convergence or maximum), calculate statistics
            resultados_finais_ts = np.array(resultados_vazao_temperatura)

            # Calculate the mean, P5, and P95 over time (for each second) using the final results
            media_vazao_ts = np.mean(resultados_finais_ts, axis=0)
            p95_vazao_ts = np.percentile(resultados_finais_ts, 95, axis=0)
            p5_vazao_ts = np.percentile(resultados_finais_ts, 5, axis=0)

            # Calculate general statistics (maximum mean, maximum P95) over the entire simulation
            max_media_vazao = np.max(media_vazao_ts)
            max_p95_vazao = np.max(p95_vazao_ts)


            # Store the statistical results and time series for this temperature
            resultados_por_temperatura[temperatura_atual] = {
                'media_ts': media_vazao_ts, # Mean time series
                'p5_ts': p5_vazao_ts,       # P5 time series
                'p95_ts': p95_vazao_ts,     # P95 time series
                'max_media': max_media_vazao, # Maximum mean over time
                'max_p95': max_p95_vazao,    # Maximum P95 over time
                'tempo': np.arange(duracao_simulacao) # The time x-axis
            }
            temp_counter += 1 # Increment the counter for simulated temperatures

        # Finalize the progress bar
        progress_bar.progress(1.0)
        st.success("Simulação concluída.")
        
        # --- INÍCIO DA ALTERAÇÃO 2: Salva os resultados no Session State ---
        st.session_state['resultados_salvos'] = resultados_por_temperatura
        st.session_state['simulacao_concluida'] = True
        # --- FIM DA ALTERAÇÃO 2 ---


# --- INÍCIO DA ALTERAÇÃO 3: Bloco de Visualização Executa após Simulação ---
# Este bloco agora é executado se a flag 'simulacao_concluida' for True,
# independente se o botão 'Executar Simulação' foi clicado nesta rodada.
if st.session_state.get('simulacao_concluida', False):
    resultados_por_temperatura = st.session_state['resultados_salvos']

    # --- NOVA SEÇÃO: RELATÓRIO TEXTUAL DA SIMULAÇÃO (apenas 1 apto) ---
    # O relatório textual só é exibido se for 1 apartamento, mas o dado 'relatorio_simulacao' não persiste
    # diretamente via session_state sem mais modificações na lógica. 
    # Mantenho o bloco condicional para o caso de 1 apartamento, assumindo que a variável existe
    # na memória da última execução (embora tecnicamente menos robusto que session state).
    if total_apartamentos == 1 and 'relatorio_simulacao' in locals():
        st.markdown("---")
        st.header("Relatório Textual Detalhado da 1ª Simulação (1 Apto) 📄")
        st.info(f"Relatório detalhado para a primeira temperatura ({list(resultados_por_temperatura.keys())[0]}°C) e uma única simulação Monte Carlo.")
        
        # O código de formatação anterior para o relatório textual...
        formatted_report = []
        
        for line in relatorio_simulacao:
            if line.startswith('['):
                if len(formatted_report) > 0:
                    formatted_report.append("\n---\n") 
                parts = line.split(' - ')
                title_part = parts[0].replace('[', '### ').replace(']', '')
                detail_part = parts[1] if len(parts) > 1 else ""
                formatted_report.append(f"{title_part} 🚿🛀") 
                formatted_report.append(f"**Detalhes da Rotina:** {detail_part}\n") 
            elif line.startswith('  - '):
                formatted_report.append(f"- **Vazão Ativa:** {line.strip()[4:]}")
            else:
                clean_line = line.replace('[', '**').replace(']', '**: ')
                formatted_report.append(f"* {clean_line.strip()}")

        st.markdown("\n".join(formatted_report))

    # --- FIM DA NOVA SEÇÃO ---


    # --- Visualize results for each temperature ---
    st.markdown("---") # Separator
    st.header("Resultados da Simulação")

    for temperatura_atual, resultados in resultados_por_temperatura.items():
        st.subheader(f"Temperatura: {temperatura_atual}°C")
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(resultados['tempo'], resultados['media_ts'], label='Média Vazão')
        ax.plot(resultados['tempo'], resultados['p95_ts'], label='P95 Vazão', linestyle='--')
        # ax.plot(resultados['tempo'], resultados['p5_ts'], label='P5 Vazão', linestyle='--') # P5 usually not plotted for maximum flow rate
        ax.fill_between(resultados['tempo'], resultados['p5_ts'], resultados['p95_ts'], color='gray', alpha=0.2, label='Faixa P5–P95')

        ax.set_xlabel('Tempo (s)')
        ax.set_ylabel('Vazão (L/s)')
        ax.legend()
        ax.set_title(f'Série Temporal de Vazão - Temperatura: {temperatura_atual}°C')
        ax.grid(True)

        # Add max mean and max P95 as text on the plot
        max_media_text = f"Máx Média: {resultados['max_media']:.2f} L/s"
        max_p95_text = f"Máx P95: {resultados['max_p95']:.2f} L/s"
        ax.text(0.01, 0.99, max_media_text, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
        ax.text(0.01, 0.92, max_p95_text, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

        plt.tight_layout()

        # Save the figure to a BytesIO object
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)

        st.image(buf, caption=f"Série Temporal de Vazão - Temperatura: {temperatura_atual}°C")
        plt.close(fig) # Close the figure to free up memory

        # Display general statistics for this temperature using st.metric or a table
        st.write("Estatísticas Gerais:")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Máximo da Vazão Média", value=f"{resultados['max_media']:.2f} L/s")
        with col2:
            st.metric(label="Máximo da Vazão P95", value=f"{resultados['max_p95']:.2f} L/s")

        # Add download button for the image
        st.download_button(
            label=f"Download Gráfico ({temperatura_atual}°C)",
            data=buf,
            file_name=f"grafico_vazao_temp_{temperatura_atual}C.png",
            mime="image/png"
        )


        st.markdown("---") # Separator between temperatures
# --- FIM DA ALTERAÇÃO 3 ---

else:
    # --- INÍCIO DA CORREÇÃO DE DUPLICIDADE: REMOÇÃO DO BOTÃO REPETIDO ---
    # Este bloco agora apenas exibe avisos se as condições mínimas de entrada não forem atendidas.
    if not temperaturas:
        st.warning("Por favor, insira temperaturas válidas para simular.")
    if duracao_simulacao <= 0:
          st.warning("A duração da simulação deve ser maior que zero.")
    if total_moradores_predio <= 0:
          st.warning("O número total de moradores no prédio deve ser maior que zero.")
    if not (not temperaturas or duracao_simulacao <= 0 or total_moradores_predio <= 0):
          # This case should not be reached if the outer if condition is correct, but as a fallback:
          st.error("Ocorreu um erro inesperado. Verifique os parâmetros de entrada.")
    # --- FIM DA CORREÇÃO DE DUPLICIDADE ---
