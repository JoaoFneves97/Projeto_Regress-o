# Projeto de Machine Learning - Regressão Logistica

Este projeto prevê a probabilidade de aprovação de emprestimos com base nas caracteristica dos clientes.

*Link kaggle:* https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset
<br>

- Colunas:

 **loan_id** - Identificacao do cliente<br>
 **no_of_dependents** - Quantidade de dependents<br>
 **education** - Escolaridade<br>
 **self_employed** - Trabalhador automono<br>
 **income_anuum** - Rendimento anual<br>
 **loan_amount** - Valor do emprestimo<br>
 **loan_term** - Tempo de emprestimo<br>
 **cibil_score** - Score do cliente<br>
 **residential_asses_value** - Ativos residenciais do cliente<br>
 **commercial_assets_value** - Ativos comerciais do cliente<br>
 **luxury_assets_value** - Ativos de luxo do cliente<br>
 **bank_asset_value** - Ativos bancarios do cliente
 **loan_status** - Status do emprestismo

**Metodo**<br>
Nesse projeto iremos utilizar a metodologia do *Crisp-DM*, como base para o desenvolvimento, que consiste em dividir o projeto em seis etapas e como objetivo  desenvolver modelos a partir da análise de informações e dados de um negócio para prever futuras falhas e soluções.

<br>

## Estrutura do Projeto

- `data/`: Diretório para arquivos de dados.
- `notebooks/`: Diretório para notebooks Jupyter (se necessário).
- `src/`: Diretório para os módulos de código fonte.
    - `data_collection.py`: Módulo para coleta dos dados.
    - `data_preparation.py`: Módulo para preparação dos dados.
    - `model_modelling.py`: Módulo para modelagem dos dados.
    - `model_preprocessing.py`: Módulo para pre processamento dos dados.
    - `model_evaluation.py`: Módulo para avaliação do modelo.
    
- `main.py`: Script principal para executar o pipeline.
- `.gitignore`: Arquivo para excluir arquivos desnecessários do Git.
- `requirements.txt`: Arquivo para listar dependências do projeto.
- `README.md`: Documentação do projeto.

## Configuração do Ambiente

1. Crie um ambiente virtual e ative-o:
    ```sh
    python -m venv venv
    source venv/bin/activate  # No Windows: venv\Scripts\activate
    ```

2. Instale as dependências:
    ```sh
    pip install -r requirements.txt
    ```

## Executando o Projeto

1. Coloque o arquivo de dados no diretório `data/`.
2. Execute o script principal:
    ```sh
    python main.py
    ```

## Personalização

Ajuste o caminho do arquivo de dados conforme necessário no arquivo `main.py`.
