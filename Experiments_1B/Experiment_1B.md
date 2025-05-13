# Experimentos - Estratégias Evolutivas

## Estrutura do Repositório

O repositório está organizado da seguinte forma:

- **`MainOptimizationScript.py`**: Contém a classe da implementação principal do algoritmo genético, incluindo as funções de otimização, avaliação de fitness, manutenção de diversidade, e geração de gráficos.
- **`Library`**: Diretório que contém módulos auxiliares, como métodos de seleção, cruzamento e mutação.
- **`Experiments_1B`**: Diretório onde os resultados dos experimentos são armazenados, incluindo gráficos e tabelas gerados.

## Experimentos 
### Experimento 1 - Avaliação do desempenho das estratégias evolutivas (μ, λ) e (μ+λ) para ambos problemas de otimização considerando um número k de inicializações do algoritmo.

O script que executa esse experimento pode ser encontrado em: [`ExecVarExperiment.py`](./01_ExecutionVariation/ExecVarExperiment.py)

Nesse experimento realiza-se ambas as estratégias evolutivas para as duas funções custo conhecidas como "Levi" e "Drop-wave" avaliando a influência do número de execuções nas suas variáveis. O número de execuções é avaliado para: [10, 50, 100, 150, 200] execuções.

#### Experimento 1 - Drop-wave - (μ+λ)

A figura a seguir apresenta a média da melhor solução encontrada em conjunto com o seu desvio padrão. Percebe-se que a influência do número de execuções é praticamente nula uma vez que há pouca diferença no valor da média e desvio padrão conforme variação de número de execuções. 

![AvgBestFitnessDropWaveMiPlusLambda](./01_ExecutionVariation/Drop-Wave/Execs%20-%20Average%20Best%20Fitness%20-%20(mi%20+%20lambda).png)

A figura a seguir apresenta a taxa de sucesso para cada número de execuções e é possível verificar que houve uma taxa de sucesso muito próxima de 100% nesse experimento.

![SucessRateDropWaveMiPlusLambda](./01_ExecutionVariation/Drop-Wave/Execs%20-%20Success%20Rate%20-%20(mi%20+%20lambda).png)

Nas figuras a seguir é possível visualizar as médias de pontos ótimos encontrados e seus respectivos desvios padrão para cada gene. 

![Gene0DropWaveMiPlusLambda](./01_ExecutionVariation/Drop-Wave/Mean%20of%20Optimal%20Points%20-%20Gene%200%20-%20(mi%20+%20lambda).png)

![Gene1DropWaveMiPlusLambda](./01_ExecutionVariation/Drop-Wave/Mean%20of%20Optimal%20Points%20-%20Gene%201%20-%20(mi%20+%20lambda).png)

As figuras a seguir apresentam alguns resultados para o experimento com 200 execuções. 

![BestFitnessperGenDropWaveMiPlusLambda](./01_ExecutionVariation/Drop-Wave/mi_plus_lambda_NEXC200_/Aggregated%20Best%20Fitness%20Per%20Generation.png)

![StepSizeperGenDropWaveMiPlusLambda](./01_ExecutionVariation/Drop-Wave/mi_plus_lambda_NEXC200_/Aggregated%20Step%20Size%20Per%20Generation.png)

![StepSizeperGenDropWaveMiPlusLambda](./01_ExecutionVariation/Drop-Wave/mi_plus_lambda_NEXC200_/Optimal%20Points%20Distribution.png)

#### Experimento 1 - Drop-wave - (μ,λ)

A figura a seguir apresenta a média da melhor solução encontrada em conjunto com o seu desvio padrão. Percebe-se que a influência do número de execuções é praticamente nula uma vez que há pouca diferença no valor da média e desvio padrão conforme variação de número de execuções. 

![AvgBestFitnessDropWaveMiCommaLambda](./01_ExecutionVariation/Drop-Wave/Execs%20-%20Average%20Best%20Fitness%20-%20(mi,lambda).png)

A figura a seguir apresenta a taxa de sucesso para cada número de execuções e é possível verificar, que para o caso de (μ,λ), uma quantidade de execuções maior do que 150 vezes indica uma certa estabilidade para lidar com a característica estocástica do modelo uma vez que a taxa de sucesso parece estabilizar para valores de execuções acima de 150.

![SucessRateDropWaveMiPlusLambda](./01_ExecutionVariation/Drop-Wave/Execs%20-%20Success%20Rate%20-%20(mi,lambda).png)

Nas figuras a seguir é possível visualizar as médias de pontos ótimos encontrados e seus respectivos desvios padrão para cada gene. 

![Gene0DropWaveMiPlusLambda](./01_ExecutionVariation/Drop-Wave/Mean%20of%20Optimal%20Points%20-%20Gene%200%20-%20(mi,lambda).png)

![Gene1DropWaveMiPlusLambda](./01_ExecutionVariation/Drop-Wave/Mean%20of%20Optimal%20Points%20-%20Gene%201%20-%20(mi,lambda).png)

As figuras a seguir apresentam alguns resultados para o experimento com 200 execuções. 

![BestFitnessperGenDropWaveMiPlusLambda](./01_ExecutionVariation/Drop-Wave/mi_comma_lambda_NEXC200_/Aggregated%20Best%20Fitness%20Per%20Generation.png)

![StepSizeperGenDropWaveMiPlusLambda](./01_ExecutionVariation/Drop-Wave/mi_comma_lambda_NEXC200_/Aggregated%20Step%20Size%20Per%20Generation.png)

![StepSizeperGenDropWaveMiPlusLambda](./01_ExecutionVariation/Drop-Wave/mi_comma_lambda_NEXC200_/Optimal%20Points%20Distribution.png)


#### Experimento 1 - Levi - (μ+λ)

A figura a seguir apresenta a média da melhor solução encontrada em conjunto com o seu desvio padrão. Percebe-se que a influência do número de execuções é praticamente nula uma vez que há pouca diferença no valor da média e desvio padrão conforme variação de número de execuções. 

![AvgBestFitnessLeviMiPlusLambda](./01_ExecutionVariation/Levi/Execs%20-%20Average%20Best%20Fitness%20-%20(mi%20+%20lambda).png)

A figura a seguir apresenta a taxa de sucesso para cada número de execuções e é possível verificar que houve uma taxa de sucesso muito próxima de 100% nesse experimento.

![SucessRateLeviMiPlusLambda](./01_ExecutionVariation/Levi/Execs%20-%20Success%20Rate%20-%20(mi%20+%20lambda).png)

Nas figuras a seguir é possível visualizar as médias de pontos ótimos encontrados e seus respectivos desvios padrão para cada gene. 

![Gene0LeviMiPlusLambda](./01_ExecutionVariation/Levi/Mean%20of%20Optimal%20Points%20-%20Gene%200%20-%20(mi%20+%20lambda).png)

![Gene1LeviMiPlusLambda](./01_ExecutionVariation/Levi/Mean%20of%20Optimal%20Points%20-%20Gene%201%20-%20(mi%20+%20lambda).png)

As figuras a seguir apresentam alguns resultados para o experimento com 200 execuções. 

![BestFitnessperGenLeviMiPlusLambda](./01_ExecutionVariation/Levi/mi_plus_lambda_NEXC200_/Aggregated%20Best%20Fitness%20Per%20Generation.png)

![StepSizeperGenLeviMiPlusLambda](./01_ExecutionVariation/Levi/mi_plus_lambda_NEXC200_/Aggregated%20Step%20Size%20Per%20Generation.png)

![StepSizeperGenLeviMiPlusLambda](./01_ExecutionVariation/Levi/mi_plus_lambda_NEXC200_/Optimal%20Points%20Distribution.png)

#### Experimento 1 - Levi - (μ,λ)

A figura a seguir apresenta a média da melhor solução encontrada em conjunto com o seu desvio padrão. Percebe-se que a influência do número de execuções é praticamente nula uma vez que há pouca diferença no valor da média e desvio padrão conforme variação de número de execuções. 

![AvgBestFitnessLeviMiCommaLambda](./01_ExecutionVariation/Levi/Execs%20-%20Average%20Best%20Fitness%20-%20(mi,lambda).png)

A figura a seguir apresenta a taxa de sucesso para cada número de execuções.

![SucessRateLeviMiPlusLambda](./01_ExecutionVariation/Levi/Execs%20-%20Success%20Rate%20-%20(mi,lambda).png)

Nas figuras a seguir é possível visualizar as médias de pontos ótimos encontrados e seus respectivos desvios padrão para cada gene. 

![Gene0LeviMiPlusLambda](./01_ExecutionVariation/Levi/Mean%20of%20Optimal%20Points%20-%20Gene%200%20-%20(mi,lambda).png)

![Gene1LeviMiPlusLambda](./01_ExecutionVariation/Levi/Mean%20of%20Optimal%20Points%20-%20Gene%201%20-%20(mi,lambda).png)

As figuras a seguir apresentam alguns resultados para o experimento com 200 execuções. 

![BestFitnessperGenLeviMiPlusLambda](./01_ExecutionVariation/Levi/mi_comma_lambda_NEXC200_/Aggregated%20Best%20Fitness%20Per%20Generation.png)

![StepSizeperGenLeviMiPlusLambda](./01_ExecutionVariation/Levi/mi_comma_lambda_NEXC200_/Aggregated%20Step%20Size%20Per%20Generation.png)

![StepSizeperGenLeviMiPlusLambda](./01_ExecutionVariation/Levi/mi_comma_lambda_NEXC200_/Optimal%20Points%20Distribution.png)


### Experimento 2 - Avaliação do desempenho das estratégias evolutivas (μ, λ) e (μ+λ) para ambos problemas de otimização em função do número máximo de iterações (gerações).

O script que executa esse experimento pode ser encontrado em: [`GenVarExperiment.py`](./02_GenerationVariation/GenVarExperiment.py)

Nesse experimento realiza-se ambas as estratégias evolutivas para as duas funções custo conhecidas como "Levi" e "Drop-wave" avaliando a influência do número máximo de iterações. O número de iterações é avaliado para: [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000] gerações.

#### Experimento 2 - Drop-wave - (μ+λ)

A figura a seguir apresenta a média da melhor solução encontrada em conjunto com o seu desvio padrão. Percebe-se que a influência do número de iterações é significativo onde não só a média se apróxima do valor teórico mas também o desvio padrão se apróxima de 0 conforme aumentamos a quantidade de iterações. Porém, é claro que há um custo computacional que também se eleva.

 

![AvgBestFitnessDropWaveMiPlusLambda](./02_GenerationVariation/Drop-Wave/Gens%20-%20Average%20Best%20Fitness%20-%20(mi%20+%20lambda).png)

A figura a seguir apresenta a taxa de sucesso para cada número de iterações.

![SucessRateDropWaveMiPlusLambda](./02_GenerationVariation/Drop-Wave/Gens%20-%20Success%20Rate%20-%20(mi%20+%20lambda).png)

Nas figuras a seguir é possível visualizar as médias de pontos ótimos encontrados e seus respectivos desvios padrão para cada gene. 

![Gene0DropWaveMiPlusLambda](./02_GenerationVariation/Drop-Wave/Mean%20of%20Optimal%20Points%20-%20Gene%200%20-%20(mi%20+%20lambda).png)

![Gene1DropWaveMiPlusLambda](./02_GenerationVariation/Drop-Wave/Mean%20of%20Optimal%20Points%20-%20Gene%201%20-%20(mi%20+%20lambda).png)

As figuras a seguir apresentam alguns resultados para o experimento com 400 iterações. 

![BestFitnessperGenDropWaveMiPlusLambda](./02_GenerationVariation/Drop-Wave/mi_plus_lambda_NGEN400_/Aggregated%20Best%20Fitness%20Per%20Generation.png)

![StepSizeperGenDropWaveMiPlusLambda](./02_GenerationVariation/Drop-Wave/mi_plus_lambda_NGEN400_/Aggregated%20Step%20Size%20Per%20Generation.png)

![StepSizeperGenDropWaveMiPlusLambda](./02_GenerationVariation/Drop-Wave/mi_plus_lambda_NGEN400_/Optimal%20Points%20Distribution.png)

#### Experimento 2 - Drop-wave - (μ,λ)

A figura a seguir apresenta a média da melhor solução encontrada em conjunto com o seu desvio padrão. Percebe-se que a assim como para o caso (μ+λ) a quantidade de iterações influência significativamente a performance do algorítimo de estratégia evolutiva.

![AvgBestFitnessDropWaveMiCommaLambda](./02_GenerationVariation/Drop-Wave/Gens%20-%20Average%20Best%20Fitness%20-%20(mi,lambda).png)

A figura a seguir apresenta a taxa de sucesso para cada número de iteração.

![SucessRateDropWaveMiPlusLambda](./02_GenerationVariation/Drop-Wave/Gens%20-%20Success%20Rate%20-%20(mi,lambda).png)

Nas figuras a seguir é possível visualizar as médias de pontos ótimos encontrados e seus respectivos desvios padrão para cada gene. 

![Gene0DropWaveMiPlusLambda](./02_GenerationVariation/Drop-Wave/Mean%20of%20Optimal%20Points%20-%20Gene%200%20-%20(mi,lambda).png)

![Gene1DropWaveMiPlusLambda](./02_GenerationVariation/Drop-Wave/Mean%20of%20Optimal%20Points%20-%20Gene%201%20-%20(mi,lambda).png)

As figuras a seguir apresentam alguns resultados para o experimento com 400 iterações. 

![BestFitnessperGenDropWaveMiPlusLambda](./02_GenerationVariation/Drop-Wave/mi_comma_lambda_NGEN400_/Aggregated%20Best%20Fitness%20Per%20Generation.png)

![StepSizeperGenDropWaveMiPlusLambda](./02_GenerationVariation/Drop-Wave/mi_comma_lambda_NGEN400_/Aggregated%20Step%20Size%20Per%20Generation.png)

![StepSizeperGenDropWaveMiPlusLambda](./02_GenerationVariation/Drop-Wave/mi_comma_lambda_NGEN400_/Optimal%20Points%20Distribution.png)


#### Experimento 2 - Levi - (μ+λ)

A figura a seguir apresenta a média da melhor solução encontrada em conjunto com o seu desvio padrão. 

![AvgBestFitnessLeviMiPlusLambda](./02_GenerationVariation/Levi/Gens%20-%20Average%20Best%20Fitness%20-%20(mi%20+%20lambda).png)

A figura a seguir apresenta a taxa de sucesso para cada número de iterações e é possível verificar que houve uma taxa de sucesso muito próxima de 100% nesse experimento quando se utiliza mais 500 iterações.

![SucessRateLeviMiPlusLambda](./02_GenerationVariation/Levi/Gens%20-%20Success%20Rate%20-%20(mi%20+%20lambda).png)

Nas figuras a seguir é possível visualizar as médias de pontos ótimos encontrados e seus respectivos desvios padrão para cada gene. 

![Gene0LeviMiPlusLambda](./02_GenerationVariation/Levi/Mean%20of%20Optimal%20Points%20-%20Gene%200%20-%20(mi%20+%20lambda).png)

![Gene1LeviMiPlusLambda](./02_GenerationVariation/Levi/Mean%20of%20Optimal%20Points%20-%20Gene%201%20-%20(mi%20+%20lambda).png)

As figuras a seguir apresentam alguns resultados para o experimento com 200 execuções. 

![BestFitnessperGenLeviMiPlusLambda](./02_GenerationVariation/Levi/mi_plus_lambda_NGEN400_/Aggregated%20Best%20Fitness%20Per%20Generation.png)

![StepSizeperGenLeviMiPlusLambda](./02_GenerationVariation/Levi/mi_plus_lambda_NGEN400_/Aggregated%20Step%20Size%20Per%20Generation.png)

![StepSizeperGenLeviMiPlusLambda](./02_GenerationVariation/Levi/mi_plus_lambda_NGEN400_/Optimal%20Points%20Distribution.png)

#### Experimento 2 - Levi - (μ,λ)

A figura a seguir apresenta a média da melhor solução encontrada em conjunto com o seu desvio padrão. 

![AvgBestFitnessLeviMiCommaLambda](./02_GenerationVariation/Levi/Gens%20-%20Average%20Best%20Fitness%20-%20(mi,lambda).png)

A figura a seguir apresenta a taxa de sucesso para cada número de iterações.

![SucessRateLeviMiPlusLambda](./02_GenerationVariation/Levi/Gens%20-%20Success%20Rate%20-%20(mi,lambda).png)

Nas figuras a seguir é possível visualizar as médias de pontos ótimos encontrados e seus respectivos desvios padrão para cada gene. 

![Gene0LeviMiPlusLambda](./02_GenerationVariation/Levi/Mean%20of%20Optimal%20Points%20-%20Gene%200%20-%20(mi,lambda).png)

![Gene1LeviMiPlusLambda](./02_GenerationVariation/Levi/Mean%20of%20Optimal%20Points%20-%20Gene%201%20-%20(mi,lambda).png)

As figuras a seguir apresentam alguns resultados para o experimento com 200 execuções. 

![BestFitnessperGenLeviMiPlusLambda](./02_GenerationVariation/Levi/mi_comma_lambda_NGEN400_/Aggregated%20Best%20Fitness%20Per%20Generation.png)

![StepSizeperGenLeviMiPlusLambda](./02_GenerationVariation/Levi/mi_comma_lambda_NGEN400_/Aggregated%20Step%20Size%20Per%20Generation.png)

![StepSizeperGenLeviMiPlusLambda](./02_GenerationVariation/Levi/mi_comma_lambda_NGEN400_/Optimal%20Points%20Distribution.png)

### Experimento 3 - Implementar operadores de recombinação e avaliar o desempenho para a estratégia evolutiva (μ + λ).

Foi implementada a estratégia de recommbinação intermédia, portanto compara-se nas imagens a seguir a implementação (μ + λ) e (μ / ρ + λ) com ρ = 10.

O script que realiza essa implementação pode ser encontrado em: [RecombinationExp.py](./03_RecombinationOperators/RecombinationExp.py)

#### Experimento 3 - Drop-wave 

A seguir apresenta-se a curva de convergência para ambos os casos. 

![ConvergenceCurve](./03_RecombinationOperators/Drop-Wave/Best%20Fitness%20-%20(mi%20+%20lambda).png)

#### Experimento 3 - Levi 

A seguir apresenta-se a curva de convergência para ambos os casos. 

![ConvergenceCurve](./03_RecombinationOperators/Levi/Best%20Fitness%20-%20(mi%20+%20lambda).png)


Parece não existir influência significativa da operação de recombinação, talvez por conta da quantidade de variáveis pois ao tirar a média de apenas duas variáveis não é exercida uma diferença significativa na recombinação.

### Experimento 4 - Aplicação do framework CMA-ES

O script que realiza a utilização do framework CMA-ES pode ser encontrado em: [CMAFrameworkExp.py](./04_CMAFramework/CMAFrameworkExp.py). 

Os experimentos utilizando o CMA-ES tiveram uma certa dificuldade em conseguir convergir para resultados aceitáveis de pontos ótimos quando comparado às estratégias apresentadas anteriormente.


#### Experimento 4 - Drop-wave

Dados de performance

| Métrica                                | Valor                     |
|---------------------------------------|---------------------------|
| Total Execution Time (s)              | 632.10                   |
| Success Rate (%)                      | 30.00                    |
| Average Best Fitness                  | -0.9553                  |
| Standard Deviation of Best Fitness    | 0.0293                   |
| Best Solution Found                   | -1.0000                  |
| Chromosome for Best Solution          | [-1.69896e-05, 1.94847e-05] |
| Mean of Optimal Points                | [0.0187, -0.0969]         |
| Standard Deviation of Optimal Points  | [0.1803, 0.3825]          |

Curvas de resultados

![Gene1](./04_CMAFramework/Drop-Wave/CMAEStrategy_/CMA-ES%20Gene%201%20Per%20Generation.png)

![Gene2](./04_CMAFramework/Drop-Wave/CMAEStrategy_/CMA-ES%20Gene%202%20Per%20Generation.png)

![StepCurve](./04_CMAFramework/Drop-Wave/CMAEStrategy_/CMA-ES%20Step%20Size%20Per%20Generation.png)

![OptimalPoints](./04_CMAFramework/Drop-Wave/CMAEStrategy_/Optimal%20Points%20Distribution.png)


#### Experimento 4 - Levi

Dados de performance

| Métrica                                | Valor                     |
|---------------------------------------|---------------------------|
| Total Execution Time (s)              | 502.49                   |
| Success Rate (%)                      | 86.00                    |
| Average Best Fitness                  | 0.000044                 |
| Standard Deviation of Best Fitness    | 0.000087                 |
| Best Solution Found                   | 0.000000                 |
| Chromosome for Best Solution          | [1.000002, 1.000097]     |
| Mean of Optimal Points                | [1.000025, 1.000078]     |
| Standard Deviation of Optimal Points  | [0.000099, 0.006548]     |

Curvas de resultados

Curvas de convergência de cada gene apresentadas para a execução com melhor fitness.

![Gene1](./04_CMAFramework/Levi/CMAEStrategy_/CMA-ES%20Gene%201%20Per%20Generation.png)

![Gene2](./04_CMAFramework/Levi/CMAEStrategy_/CMA-ES%20Gene%202%20Per%20Generation.png)

![StepCurve](./04_CMAFramework/Levi/CMAEStrategy_/CMA-ES%20Step%20Size%20Per%20Generation.png)

![OptimalPoints](./04_CMAFramework/Levi/CMAEStrategy_/Optimal%20Points%20Distribution.png)
