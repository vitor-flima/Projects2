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
st.sidebar.subheader("Opções de Visualização")
show_membership_functions = st.sidebar.checkbox("Mostrar Funções de Pertinência Fuzzy")


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


# Passo 5: Regras fuzzy (Mantido como antes)
morador_1_rules = [
    ctrl.Rule(temperatura_do_ar['Very cold'] & inicio_do_banho['Very early'], duracao_do_banho['Very fast']),
    ctrl.Rule(temperatura_do_ar['Cold'] & inicio_do_banho['Very early'], duracao_do_banho['Fast']),
    ctrl.Rule(temperatura_do_ar['Pleasant'] & inicio_do_banho['Very early'], duracao_do_banho['Normal']),
    ctrl.Rule(temperatura_do_ar['Hot'] & inicio_do_banho['Very early'], duracao_do_banho['Normal']),
    ctrl.Rule(temperatura_do_ar['Very hot'] & inicio_do_banho['Very early'], duracao_do_banho['Long']), # Corrected typo here

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
    # Corrected typo here
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

    ctrl.Rule(temperatura_do_ar['Very cold'] & inicio_do_banho['Delayed'], duracao_do_banho['No shower']),
    ctrl.Rule(temperatura_do_ar['Cold'] & inicio_do_banho['Delayed'], duracao_do_banho['No shower']),
    ctrl.Rule(temperatura_do_ar['Pleasant'] & inicio_do_banho['Delayed'], duracao_do_banho['No shower']),
    ctrl.Rule(temperatura_do_ar['Hot'] & inicio_do_banho['Delayed'], duracao_do_banho['Very fast']),
    ctrl.Rule(temperatura_do_ar['Very hot'] & inicio_do_banho['Delayed'], duracao_do_banho['Fast']),

    ctrl.Rule(temperatura_do_ar['Very cold'] & inicio_do_banho['Very delayed'], duracao_do_banho['No shower']),
    ctrl.Rule(temperatura_do_ar['Cold'] & inicio_do_banho['Very delayed'], duracao_do_banho['No shower']),
    ctrl.Rule(temperatura_do_ar['Pleasant'] & inicio_do_banho['Very delayed'], duracao_do_banho['No shower']),
    ctrl.Rule(temperatura_do_ar['Hot'] & inicio_do_banho['Very delayed'], duracao_do_banho['No shower']),
    ctrl.Rule(temperatura_do_ar['Very hot'] & inicio_do_banho['Very delayed'], duracao_do_banho['No shower'])
]

# Mapeia a quantidade de morador para o conjunto de regras
rules_map = {
    1: morador_1_rules,
    2: morador_2_rules,
    3: morador_3_rules
}

# Cria uma lista de simuladores com base na quantidade de moradores por apartamento
simuladores = []
for i in range(quantidade_moradores_por_apartamento):
    # Select rules: morador 1, 2, 3, and morador 3 for the others
    if i + 1 in rules_map:
        regras_morador_atual = rules_map[i + 1]
    else:
        regras_morador_atual = morador_3_rules # Uses morador 3 rules for morador 4 onwards

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

# Cria a lista de todos os moradores do prédio com suas características e apartamento
total_apartamentos = apartamentos_por_pavimento * quantidade_pavimentos
total_moradores_predio = total_apartamentos * quantidade_moradores_por_apartamento

st.write(f"Calculando moradores para {total_apartamentos} apartamentos com {quantidade_moradores_por_apartamento} moradores por apartamento...")

moradores_predio = []

# Creating the list of all residents, identified by apartment and type
for apt_num in range(1, total_apartamentos + 1):
    moradores_no_apartamento = []
    for morador_num_no_apt in range(1, quantidade_moradores_por_apartamento + 1):
        # Determine the resident type (1, 2, or 3+ using morador 3 rules)
        tipo_morador = min(morador_num_no_apt, 3)
        nome_morador = f'morador {tipo_morador}'

        moradores_no_apartamento.append({
            'nome': nome_morador,
            'apartamento': apt_num,
            'usa_pia': False # Initialize all as False
        })
    moradores_predio.extend(moradores_no_apartamento)

# Randomly select one resident per apartment to use the sink
# Group residents by apartment for easier selection
moradores_por_apartamento_dict = {}
for morador in moradores_predio:
    if morador['apartamento'] not in moradores_por_apartamento_dict:
        moradores_por_apartamento_dict[morador['apartamento']] = []
    moradores_por_apartamento_dict[morador['apartamento']].append(morador)

# Randomly select one resident to use the sink in each apartment
for apt_num, lista_moradores_apt in moradores_por_apartamento_dict.items():
    if lista_moradores_apt: # Ensures there are residents in the apartment
        morador_usa_pia = random.choice(lista_moradores_apt)
        morador_usa_pia['usa_pia'] = True

# The final list of residents for the simulation is 'moradores_predio'
st.write(f"Lista criada com {len(moradores_predio)} moradores para todo o prédio.")

# Calculate the total number of bathrooms in the building (assuming each apartment has the same quantity)
total_banheiros_predio = total_apartamentos * quantidade_banheiros_por_apartamento


# --- Visualize Fuzzy Membership Functions (if option is selected) ---
if show_membership_functions:
    st.subheader("Funções de Pertinência Fuzzy")

    # Input variable: inicio_do_banho
    fig_inicio_do_banho, ax_inicio = plt.subplots()
    inicio_do_banho.view(ax=ax_inicio)
    st.pyplot(fig_inicio_do_banho)
    plt.close(fig_inicio_do_banho)

    # Input variable: temperatura_do_ar
    fig_temperatura_do_ar, ax_temp = plt.subplots()
    temperatura_do_ar.view(ax=ax_temp)
    st.pyplot(fig_temperatura_do_ar)
    plt.close(fig_temperatura_do_ar)

    # Output variable: duracao_do_banho
    fig_duracao_do_banho, ax_duracao = plt.subplots()
    duracao_do_banho.view(ax=ax_duracao)
    st.pyplot(fig_duracao_do_banho)
    plt.close(fig_duracao_do_banho)

    st.markdown("---") # Separator


# Main loop over each temperature to be simulated - Executes only if there are valid temperatures
if temperaturas and duracao_simulacao > 0 and total_moradores_predio > 0:
    if st.sidebar.button("Executar Simulação"):
        st.info("Iniciando simulação de Monte Carlo...")
        # Use st.progress to show the overall simulation progress
        progress_bar = st.progress(0)
        total_temperaturas_simular = len(temperaturas)
        temp_counter = 0

        # Dictionary to store results (flow rate, statistics, etc.) for each temperature
        resultados_por_temperatura = {}

        for temperatura_atual in temperaturas:
            st.subheader(f"Simulação para Temperatura: {temperatura_atual}°C")
            st.info(f"Executando simulações para Temperatura: {temperatura_atual}°C")

            # List to store the flow rate time series of each Monte Carlo simulation for this temperature
            resultados_vazao_temperatura = []
            max_p95_anterior = float('inf') # Initialize with infinity to ensure the first difference is large
            convergencia_atingida = False

            # Monte Carlo simulation loop
            # We use n_simulacoes_maximo as an upper limit, but the loop can stop earlier due to convergence
            for i in range(n_simulacoes_maximo):
                # Update the progress bar (approximate)
                progress = (temp_counter / total_temperaturas_simular) + (i / n_simulacoes_maximo / total_temperaturas_simular)
                progress_bar.progress(min(progress, 1.0)) # Ensures it doesn't exceed 100%

                # Initialize the flow rate time series for this simulation (total building flow rate per second)
                vazao_simulacao = np.zeros(duracao_simulacao)

                # Initialize the occupation state of ALL BATHROOMS IN THE BUILDING.
                banheiros_livres_em = np.zeros(total_banheiros_predio) # Stores the second when each bathroom will be free


                # --- Simulation logic for each resident and bathroom usage ---
                # Iterate over each resident in 'moradores_predio'
                for m in moradores_predio:
                    # Generate random shower start time within the total simulation interval
                    inicio_banho = random.randint(0, duracao_simulacao - 1)

                    # Determine which fuzzy simulator to use based on the resident type ('morador X' name)
                    # The name is 'morador Y', so the simulator index is Y-1. We use min(..., 2) to get simulator 1, 2, or 3.
                    try:
                        tipo_morador_num = int(m['nome'].split(' ')[1])
                        simulador_morador_atual = simuladores[min(tipo_morador_num - 1, len(simuladores)-1)] # Ensure index is within bounds
                    except (ValueError, IndexError):
                        # Fallback in case of unexpected morador name format
                        st.warning(f"Formato de nome de morador inesperado: {m['nome']}. Usando regras do último tipo de morador.")
                        simulador_morador_atual = simuladores[-1]


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

                        # Calculate bathroom occupation intervals for toilet, shower, and sink
                        # Considering fixed times and intervals provided by the user
                        inicio_vaso = max(0, inicio_banho - duracao_vaso) # Adjusted to start 60s before the end of toilet use
                        fim_vaso = inicio_vaso + duracao_vaso

                        inicio_lavatorio = inicio_banho + dur_banho_segundos + 30
                        fim_lavatorio = inicio_lavatorio + duracao_lavatorio

                        # Calculate the total bathroom occupation interval for this resident
                        # From the start of toilet use to the end of sink use
                        intervalo_ocupacao_inicio = inicio_vaso
                        intervalo_ocupacao_fim = fim_lavatorio

                        # Ensure intervals do not exceed the total simulation duration
                        intervalo_ocupacao_inicio = max(0, intervalo_ocupacao_inicio)
                        intervalo_ocupacao_fim = min(duracao_simulacao, intervalo_ocupacao_fim)


                        # --- Check bathroom availability WITHIN THE RESIDENT'S APARTMENT ---
                        # Calculate the indices of the bathrooms for this resident's apartment
                        apt_num = m['apartamento']
                        primeiro_indice_banheiro_apt = (apt_num - 1) * quantidade_banheiros_por_apartamento
                        ultimo_indice_banheiro_apt = primeiro_indice_banheiro_apt + quantidade_banheiros_por_apartamento - 1

                        banheiro_disponivel_indice_global = -1 # Global index of the available bathroom in the building
                        # Look for an available bathroom ONLY WITHIN THIS APARTMENT
                        for idx_banheiro_global in range(primeiro_indice_banheiro_apt, ultimo_indice_banheiro_apt + 1):
                            if idx_banheiro_global < len(banheiros_livres_em) and banheiros_livres_em[idx_banheiro_global] <= intervalo_ocupacao_inicio:
                                banheiro_disponivel_indice_global = idx_banheiro_global
                                break # Found an available bathroom in this apartment

                        # If a bathroom is available in this apartment, "occupy" it and add flow rate
                        if banheiro_disponivel_indice_global != -1:
                            # Record that this bathroom will be occupied until the end of the occupation interval
                            banheiros_livres_em[banheiro_disponivel_indice_global] = intervalo_ocupacao_fim

                            # Add flow rate to the time series for the corresponding intervals
                            # Toilet: from inicio_vaso to fim_vaso
                            # Ensure indices are within the limits of the flow rate time series
                            inicio_vaso_clamped = max(0, inicio_vaso)
                            fim_vaso_clamped = min(duracao_simulacao, fim_vaso)
                            if fim_vaso_clamped > inicio_vaso_clamped:
                                vazao_simulacao[inicio_vaso_clamped:fim_vaso_clamped] += vaso

                            # Shower: from inicio_banho to inicio_banho + dur_banho_segundos
                            inicio_banho_clamped = max(0, inicio_banho)
                            fim_banho_clamped = min(duracao_simulacao, inicio_banho + dur_banho_segundos)
                            if fim_banho_clamped > inicio_banho_clamped:
                                vazao_simulacao[inicio_banho_clamped : fim_banho_clamped] += chuveiro

                            # Sink: from inicio_lavatorio to fim_lavatorio
                            inicio_lavatorio_clamped = max(0, inicio_lavatorio)
                            fim_lavatorio_clamped = min(duracao_simulacao, fim_lavatorio)
                            if fim_lavatorio_clamped > inicio_lavatorio_clamped:
                                vazao_simulacao[inicio_lavatorio_clamped:fim_lavatorio_clamped] += lavatorio


                            # Kitchen sink (if the resident uses it) - Kitchen sink usage does not occupy the physical bathroom
                            # but contributes to the total flow rate. It is assumed the kitchen sink is separate from the bathroom.
                            if m['usa_pia']:
                                 # Kitchen sink usage occurs independently of physical bathroom availability
                                 # Assuming kitchen sink usage occurs after the bathroom sink,
                                 # or at a fixed time after the shower as in previous logic (120s after shower end)
                                 inicio_pia = inicio_banho + dur_banho_segundos + 120
                                 fim_pia = inicio_pia + duracao_pia
                                 # Ensure kitchen sink intervals do not exceed the total simulation duration
                                 inicio_pia_clamped = max(0, inicio_pia)
                                 fim_pia_clamped = min(duracao_simulacao, fim_pia)
                                 if fim_pia_clamped > inicio_pia_clamped:
                                      vazao_simulacao[inicio_pia_clamped:fim_pia_clamped] += pia


                    except ValueError as e:
                         # This can happen if the inputs to the fuzzy simulator are outside the universe of the fuzzy variables
                         # For example, if inicio_banho or temperatura_atual are outside the defined ranges
                         st.warning(f"Erro na computação fuzzy para morador {m['nome']} do apto {m['apartamento']} na temperatura {temperatura_atual}°C: {e}")


                # Add the flow rate time series of this simulation to the results list for this temperature
                resultados_vazao_temperatura.append(vazao_simulacao)

                # --- Check for Convergence Criterion ---
                if (i + 1) % verificar_a_cada_n_simulacoes == 0 and (i + 1) >= n_simulacoes_minimo:
                     # Convert the cumulative list to a numpy array for percentile calculation
                     resultados_cumulativos = np.array(resultados_vazao_temperatura)
                     # Calculate the cumulative P95 over time
                     p95_cumulativo_ts = np.percentile(resultados_cumulativos, 95, axis=0)
                     # Calculate the maximum of the cumulative P95
                     max_p95_cumulativo = np.max(p95_cumulativo_ts)

                     # Check the stopping criterion
                     if abs(max_p95_cumulativo - max_p95_anterior) < limiar_convergencia:
                         st.success(f"Convergência atingida após {i + 1} simulações para {temperatura_atual}°C.")
                         convergencia_atingida = True
                         break # Exit the Monte Carlo simulation loop
                     else:
                         max_p95_anterior = max_p95_cumulativo # Update the previous value

                # If the loop reached the maximum number of simulations without convergence, report it
                if i + 1 == n_simulacoes_maximo and not convergencia_atingida:
                     st.warning(f"Número máximo de simulações ({n_simulacoes_maximo}) atingido sem convergência para {temperatura_atual}°C.")


            # After all Monte Carlo simulations for this temperature (until convergence or maximum), calculate statistics
            # Convert the list of flow rate arrays to a numpy array for easier statistical calculations
            # If the loop stopped due to convergence, resultados_vazao_temperatura already contains the simulations up to the stopping point.
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
                'p5_ts': p5_vazao_ts,     # P5 time series
                'p95_ts': p95_vazao_ts,    # P95 time series
                'max_media': max_media_vazao, # Maximum mean over time
                'max_p95': max_p95_vazao,   # Maximum P95 over time
                'tempo': np.arange(duracao_simulacao) # The time x-axis
            }
            temp_counter += 1 # Increment the counter for simulated temperatures

        # Finalize the progress bar
        progress_bar.progress(1.0)
        st.success("Simulação concluída.")

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

else:
    if st.sidebar.button("Executar Simulação"): # Only show the button if conditions are met
        if not temperaturas:
            st.warning("Por favor, insira temperaturas válidas para simular.")
        if duracao_simulacao <= 0:
             st.warning("A duração da simulação deve ser maior que zero.")
        if total_moradores_predio <= 0:
             st.warning("O número total de moradores no prédio deve ser maior que zero.")
        if not (not temperaturas or duracao_simulacao <= 0 or total_moradores_predio <= 0):
             # This case should not be reached if the outer if condition is correct, but as a fallback:
             st.error("Ocorreu um erro inesperado. Verifique os parâmetros de entrada.")
