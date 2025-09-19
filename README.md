# Sistema de Predição de Evasão de Alunos - Pacote Completo

Este pacote contém tudo o que você precisa para treinar e executar o sistema de predição de evasão de alunos, incluindo scripts, modelo treinado e documentação.

## 🚀 Visão Geral

O sistema utiliza um modelo de Machine Learning (XGBoost) para prever a probabilidade de evasão de alunos, com foco em três fatores principais: Pendência Financeira, Faltas Consecutivas e Pendência Acadêmica.

### Performance do Modelo

- **Acurácia Multiclasse:** 71.78%
- **Acurácia Binária (Evasão vs. Matriculado):** 97.53%

## 📂 Arquivos Inclusos

- **`predicao_matricula_corrigida.py`**: Script principal para fazer predições em novos dados de alunos.
- **`treinar_modelo_xgboost.py`**: Script completo para treinar o modelo XGBoost do zero.
- **`preprocessamento_dados.py`**: Script para pré-processar os dados de treinamento.
- **`requirements.txt`**: Lista de bibliotecas Python necessárias.
- **`output/`**: Pasta contendo o modelo treinado, mapeamento de classes e outros arquivos gerados.
  - `modelo_xgboost_sem_classes_criticas.pkl`: O modelo de machine learning treinado.
  - `class_mapping_otimizado.pkl`: Mapeamento das classes de situação dos alunos.
- **`upload/`**: Pasta com exemplos de planilhas de dados.
  - `alunos_ativos_atual.xlsx`: Exemplo de planilha de alunos ativos para predição.
  - `Planilhabasedados.xlsx`: Exemplo de planilha de base de dados para treinamento.
- **`README_COMPLETO.md`**: Este arquivo de instruções detalhadas.

## 🛠️ Como Usar

### 1. Instalação

Primeiro, instale todas as dependências necessárias:

```bash
pip install -r requirements.txt
```

### 2. Fazendo Predições (Uso Principal)

Para fazer predições em uma nova planilha de alunos, use o script `predicao_matricula_corrigida.py`.

```bash
python predicao_matricula_corrigida.py caminho/para/sua/planilha.xlsx
```

**Exemplo:**

```bash
python predicao_matricula_corrigida.py upload/alunos_ativos_atual.xlsx
```

O script irá gerar um arquivo CSV na pasta `output/` com as predições detalhadas para cada aluno, incluindo a **Matrícula correta**.

### 3. Treinando o Modelo (Avançado)

Se você quiser treinar o modelo com seus próprios dados, siga estes passos:

**a. Pré-processe seus dados:**

Use o script `preprocessamento_dados.py` para limpar e preparar sua base de dados.

```bash
python preprocessamento_dados.py caminho/para/sua/base_de_dados.xlsx
```

**Exemplo:**

```bash
python preprocessamento_dados.py upload/Planilhabasedados.xlsx
```

Isso irá gerar um arquivo `dados_preprocessados.csv` na pasta `output/`.

**b. Treine o modelo XGBoost:**

Use o script `treinar_modelo_xgboost.py` para treinar o modelo com os dados pré-processados.

```bash
python treinar_modelo_xgboost.py caminho/para/sua/base_de_dados.xlsx
```

**Exemplo:**

```bash
python treinar_modelo_xgboost.py upload/Planilhabasedados.xlsx
```

Este script irá:
- Treinar um novo modelo XGBoost.
- Salvar o modelo treinado em `output/modelo_xgboost_sem_classes_criticas.pkl`.
- Gerar relatórios de performance, matriz de confusão e importância das features.

## ✅ Correções e Melhorias

- **Coluna Matrícula Corrigida:** O sistema agora utiliza a coluna `Matrícula` como identificador principal, garantindo que os resultados sejam corretamente associados a cada aluno.
- **Pacote Completo:** Todos os scripts e arquivos necessários para treinamento e predição estão inclusos.
- **Documentação Detalhada:** Instruções claras para uso básico e avançado.

---

