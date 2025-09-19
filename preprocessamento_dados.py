#!/usr/bin/env python3
"""
Script de pré-processamento de dados para o sistema de predição de evasão
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import joblib
import os

class PreprocessadorDados:
    """
    Classe para pré-processamento de dados
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.imputers = {}
        self.features_numericas = ['Faltas Consecutivas', 'Módulo atual', 'Cód.Curso', 'Pend. Financ.', 'Identidade']
        self.features_categoricas = ['Curso', 'Currículo', 'Sexo', 'Disciplina atual', 'Pend. Acad.', 'Turma Atual', 'Cód.Disc. atual']
        
    def detectar_header(self, arquivo):
        """Detectar linha do header automaticamente"""
        df_raw = pd.read_excel(arquivo, header=None)
        
        for i in range(min(5, len(df_raw))):
            row = df_raw.iloc[i]
            row_str = ' '.join([str(val) for val in row.values if pd.notna(val)]).upper()
            if any(col in row_str for col in ['MATRÍCULA', 'MATRICULA', 'NOME', 'CURSO', 'SITUAÇÃO']):
                return i
        return 0
    
    def carregar_dados(self, arquivo):
        """Carregar dados do arquivo Excel"""
        print(f"Carregando dados de: {arquivo}")
        
        # Detectar header
        header_row = self.detectar_header(arquivo)
        print(f"Header detectado na linha: {header_row}")
        
        # Carregar dados
        df = pd.read_excel(arquivo, header=header_row)
        print(f"Dados carregados: {df.shape[0]} registros, {df.shape[1]} colunas")
        
        return df
    
    def analisar_dados(self, df):
        """Analisar qualidade dos dados"""
        print("\n" + "="*50)
        print("ANÁLISE DA QUALIDADE DOS DADOS")
        print("="*50)
        
        print(f"Dimensões: {df.shape[0]} linhas x {df.shape[1]} colunas")
        
        # Valores ausentes
        print("\nValores ausentes por coluna:")
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        
        for col in df.columns:
            if missing[col] > 0:
                print(f"  {col:<25}: {missing[col]:4d} ({missing_pct[col]:5.1f}%)")
        
        # Distribuição da variável target
        if 'Situação' in df.columns:
            print("\nDistribuição da variável target (Situação):")
            distribuicao = df['Situação'].value_counts()
            for situacao, count in distribuicao.items():
                pct = (count / len(df)) * 100
                print(f"  {situacao:<25}: {count:4d} ({pct:5.1f}%)")
        
        # Estatísticas das features numéricas
        features_num_disponiveis = [col for col in self.features_numericas if col in df.columns]
        if features_num_disponiveis:
            print(f"\nEstatísticas das features numéricas:")
            print(df[features_num_disponiveis].describe())
        
        return missing, distribuicao if 'Situação' in df.columns else None
    
    def limpar_dados(self, df):
        """Limpar e preparar dados"""
        print("\n" + "="*50)
        print("LIMPEZA DOS DADOS")
        print("="*50)
        
        df_limpo = df.copy()
        
        # Remover linhas completamente vazias
        linhas_antes = len(df_limpo)
        df_limpo = df_limpo.dropna(how='all')
        linhas_removidas = linhas_antes - len(df_limpo)
        if linhas_removidas > 0:
            print(f"Removidas {linhas_removidas} linhas completamente vazias")
        
        # Remover duplicatas baseadas em Matrícula (se existir)
        if 'Matrícula' in df_limpo.columns:
            duplicatas_antes = len(df_limpo)
            df_limpo = df_limpo.drop_duplicates(subset=['Matrícula'], keep='first')
            duplicatas_removidas = duplicatas_antes - len(df_limpo)
            if duplicatas_removidas > 0:
                print(f"Removidas {duplicatas_removidas} duplicatas baseadas em Matrícula")
        
        # Padronizar valores categóricos
        for col in df_limpo.columns:
            if df_limpo[col].dtype == 'object':
                # Remover espaços extras
                df_limpo[col] = df_limpo[col].astype(str).str.strip()
                # Substituir valores vazios por NaN
                df_limpo[col] = df_limpo[col].replace(['', 'nan', 'NaN', 'None'], np.nan)
        
        print(f"Dados após limpeza: {df_limpo.shape[0]} registros")
        
        return df_limpo
    
    def preprocessar_features(self, df, salvar_preprocessadores=True):
        """Preprocessar features para treinamento"""
        print("\n" + "="*50)
        print("PRÉ-PROCESSAMENTO DAS FEATURES")
        print("="*50)
        
        # Selecionar features disponíveis
        features_numericas_disp = [col for col in self.features_numericas if col in df.columns]
        features_categoricas_disp = [col for col in self.features_categoricas if col in df.columns]
        
        print(f"Features numéricas disponíveis: {len(features_numericas_disp)}")
        print(f"Features categóricas disponíveis: {len(features_categoricas_disp)}")
        
        df_processado = df.copy()
        
        # Processar features numéricas
        if features_numericas_disp:
            print("\nProcessando features numéricas...")
            
            # Converter para numérico
            for col in features_numericas_disp:
                df_processado[col] = pd.to_numeric(df_processado[col], errors='coerce')
            
            # Imputar valores ausentes
            imputer_num = SimpleImputer(strategy='median')
            df_processado[features_numericas_disp] = imputer_num.fit_transform(df_processado[features_numericas_disp])
            
            if salvar_preprocessadores:
                self.imputers['numericas'] = imputer_num
        
        # Processar features categóricas
        if features_categoricas_disp:
            print("Processando features categóricas...")
            
            # Imputar valores ausentes
            imputer_cat = SimpleImputer(strategy='most_frequent')
            df_processado[features_categoricas_disp] = imputer_cat.fit_transform(df_processado[features_categoricas_disp])
            
            if salvar_preprocessadores:
                self.imputers['categoricas'] = imputer_cat
            
            # Codificar variáveis categóricas
            for col in features_categoricas_disp:
                le = LabelEncoder()
                df_processado[col] = le.fit_transform(df_processado[col].astype(str))
                if salvar_preprocessadores:
                    self.label_encoders[col] = le
                
                print(f"  {col}: {len(le.classes_)} categorias únicas")
        
        # Processar target (se existir)
        if 'Situação' in df_processado.columns and salvar_preprocessadores:
            le_target = LabelEncoder()
            df_processado['Situação_encoded'] = le_target.fit_transform(df_processado['Situação'])
            self.label_encoders['target'] = le_target
            print(f"Target (Situação): {len(le_target.classes_)} classes")
        
        print(f"Dados pré-processados: {df_processado.shape}")
        
        return df_processado
    
    def salvar_preprocessadores(self, caminho='output/preprocessadores.pkl'):
        """Salvar preprocessadores para uso futuro"""
        os.makedirs(os.path.dirname(caminho), exist_ok=True)
        
        preprocessadores = {
            'label_encoders': self.label_encoders,
            'imputers': self.imputers,
            'features_numericas': self.features_numericas,
            'features_categoricas': self.features_categoricas
        }
        
        joblib.dump(preprocessadores, caminho)
        print(f"Preprocessadores salvos em: {caminho}")
    
    def carregar_preprocessadores(self, caminho='output/preprocessadores.pkl'):
        """Carregar preprocessadores salvos"""
        if os.path.exists(caminho):
            preprocessadores = joblib.load(caminho)
            self.label_encoders = preprocessadores['label_encoders']
            self.imputers = preprocessadores['imputers']
            self.features_numericas = preprocessadores['features_numericas']
            self.features_categoricas = preprocessadores['features_categoricas']
            print(f"Preprocessadores carregados de: {caminho}")
            return True
        return False
    
    def gerar_relatorio_preprocessamento(self, df_original, df_processado, missing_original, distribuicao_original):
        """Gerar relatório de pré-processamento"""
        relatorio = f"""
RELATÓRIO DE PRÉ-PROCESSAMENTO DE DADOS
=======================================

DADOS ORIGINAIS:
- Registros: {df_original.shape[0]}
- Colunas: {df_original.shape[1]}
- Valores ausentes: {missing_original.sum()} ({(missing_original.sum()/(df_original.shape[0]*df_original.shape[1]))*100:.1f}%)

DADOS PROCESSADOS:
- Registros: {df_processado.shape[0]}
- Colunas: {df_processado.shape[1]}
- Registros removidos: {df_original.shape[0] - df_processado.shape[0]}

FEATURES PROCESSADAS:
- Numéricas: {len([col for col in self.features_numericas if col in df_processado.columns])}
- Categóricas: {len([col for col in self.features_categoricas if col in df_processado.columns])}

LABEL ENCODERS CRIADOS:
{chr(10).join([f"  • {col}: {len(encoder.classes_)} classes" for col, encoder in self.label_encoders.items()])}

DISTRIBUIÇÃO DO TARGET (ORIGINAL):
{chr(10).join([f"  • {situacao}: {count} ({(count/df_original.shape[0])*100:.1f}%)" for situacao, count in distribuicao_original.items()]) if distribuicao_original is not None else "  Target não encontrado"}

ARQUIVOS GERADOS:
- output/preprocessadores.pkl
- output/dados_preprocessados.csv
- output/relatorio_preprocessamento.txt
"""
        
        # Salvar relatório
        os.makedirs('output', exist_ok=True)
        with open('output/relatorio_preprocessamento.txt', 'w', encoding='utf-8') as f:
            f.write(relatorio)
        
        print("Relatório salvo em: output/relatorio_preprocessamento.txt")
        print(relatorio)

def main():
    """Função principal"""
    import sys
    
    if len(sys.argv) != 2:
        print("Uso: python preprocessamento_dados.py <arquivo_dados.xlsx>")
        print("Exemplo: python preprocessamento_dados.py upload/Planilhabasedados.xlsx")
        sys.exit(1)
    
    arquivo = sys.argv[1]
    
    if not os.path.exists(arquivo):
        print(f"Erro: Arquivo não encontrado: {arquivo}")
        sys.exit(1)
    
    print("="*60)
    print("PRÉ-PROCESSAMENTO DE DADOS PARA PREDIÇÃO DE EVASÃO")
    print("="*60)
    
    # Inicializar preprocessador
    preprocessador = PreprocessadorDados()
    
    try:
        # Carregar dados
        df_original = preprocessador.carregar_dados(arquivo)
        
        # Analisar dados
        missing_original, distribuicao_original = preprocessador.analisar_dados(df_original)
        
        # Limpar dados
        df_limpo = preprocessador.limpar_dados(df_original)
        
        # Preprocessar features
        df_processado = preprocessador.preprocessar_features(df_limpo)
        
        # Salvar preprocessadores
        preprocessador.salvar_preprocessadores()
        
        # Salvar dados processados
        os.makedirs('output', exist_ok=True)
        df_processado.to_csv('output/dados_preprocessados.csv', index=False, encoding='utf-8-sig')
        print("Dados processados salvos em: output/dados_preprocessados.csv")
        
        # Gerar relatório
        preprocessador.gerar_relatorio_preprocessamento(df_original, df_processado, missing_original, distribuicao_original)
        
        print("="*60)
        print("✅ PRÉ-PROCESSAMENTO CONCLUÍDO COM SUCESSO!")
        print("="*60)
        
    except Exception as e:
        print(f"Erro durante o pré-processamento: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

