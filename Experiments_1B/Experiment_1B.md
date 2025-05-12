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


#### Experimento 1 - Levi - (μ+λ)

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

#### Experimento 1 - Levi - (μ,λ)

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