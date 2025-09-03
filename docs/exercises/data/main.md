# Exploração de dados

Nessa seção, eu explorei a natureza dos dados, como eles podem se comportar com classes separadas de forma linear, não linear, entendendo como elas funcionam e visualizando isso em gráficos. Ao final, foi realizada de forma breve uma manipulação com base no dataset Spaceship Titanic, explorando suas variáveis, a distribuição dos dados e como lidar com algumas observações com dados faltantes. Além disso, explorei também como faria para pré processar os dados visando utilizá-los como entrada em uma rede neural.

## Separação de dados em 2D

Buscando explorar dados primeiramente em apenas duas dimensões, gerei amostras de 100 observações em cada, cujas distribuições eram ditadas pelas seguintes regras:
    * **Class 0:** Mean = $[2, 3]$, Standard Deviation = $[0.8, 2.5]$
    * **Class 1:** Mean = $[5, 6]$, Standard Deviation = $[1.2, 1.9]$
    * **Class 2:** Mean = $[8, 1]$, Standard Deviation = $[0.9, 0.9]$
    * **Class 3:** Mean = $[15, 4]$, Standard Deviation = $[0.5, 2.0]$

Para isso, utilizei uma das bibliotecas mais clássicas do Python, NumPy, para gerar cada uma das classes, com a sua respectiva distribuição. Tendo em vista a replicabilidade desse estudo, eu utilizei uma __RANDOM SEED__ (42).

