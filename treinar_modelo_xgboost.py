#!/usr/bin/env python3
"""
Script completo para treinamento do modelo XGBoost de predição de evasão
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
import shap
import os

class TreinadorXGBoost:
    """
    Classe para treinar o modelo XGBoost de predição de evasão
    """
    
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.features_selecionadas = [
            'Pend. Financ.', 'Faltas Consecutivas', 'Pend. Acad.', 'Módulo atual',
            'Cód.Curso', 'Curso', 'Currículo', 'Sexo', 'Identidade', 
            'Turma Atual', 'Cód.Disc. atual', 'Disciplina atual'
        ]
        
        # Classes críticas a serem removidas (baixa performance)
        self.classes_criticas = ['Cancelamento Interno', 'Transferência Interna']
        
        # Classes mantidas após otimização
        self.classes_mantidas = [
            'Cancelamento Comercial', 'Cancelamento Unidade', 'Não Formados',
            'Limpeza Academica', 'Limpeza Financeira', 'Limpeza de Frequencia',
            'Matriculado', 'Nunca Compareceu'
        ]
    
    def detectar_header(self, arquivo):
        """Detectar linha do header automaticamente"""
        df_raw = pd.read_excel(arquivo, header=None)
        
        for i in range(min(5, len(df_raw))):
            row = df_raw.iloc[i]
            row_str = ' '.join([str(val) for val in row.values if pd.notna(val)]).upper()
            if any(col in row_str for col in ['MATRÍCULA', 'MATRICULA', 'NOME', 'CURSO', 'SITUAÇÃO']):
                return i
        return 0
    
    def carregar_dados(self, arquivo_base_dados):
        """Carregar e preprocessar dados de treinamento"""
        print(f"Carregando dados de: {arquivo_base_dados}")
        
        # Detectar header
        header_row = self.detectar_header(arquivo_base_dados)
        print(f"Header detectado na linha: {header_row}")
        
        # Carregar dados
        df = pd.read_excel(arquivo_base_dados, header=header_row)
        print(f"Dados carregados: {df.shape[0]} registros, {df.shape[1]} colunas")
        
        # Remover classes críticas
        print(f"Removendo classes críticas: {self.classes_criticas}")
        df_filtrado = df[~df['Situação'].isin(self.classes_criticas)].copy()
        print(f"Dados após remoção: {df_filtrado.shape[0]} registros")
        
        # Verificar distribuição das classes
        print("Distribuição das classes:")
        distribuicao = df_filtrado['Situação'].value_counts()
        for classe, count in distribuicao.items():
            pct = (count / len(df_filtrado)) * 100
            print(f"  {classe:<25}: {count:4d} ({pct:5.1f}%)")
        
        return df_filtrado
    
    def preprocessar_dados(self, df):
        """Preprocessar dados para treinamento"""
        print("Preprocessando dados...")
        
        # Verificar features disponíveis
        features_disponiveis = [col for col in self.features_selecionadas if col in df.columns]
        features_faltando = [col for col in self.features_selecionadas if col not in df.columns]
        
        if features_faltando:
            print(f"AVISO: Features não encontradas: {features_faltando}")
        
        print(f"Features disponíveis: {len(features_disponiveis)}/{len(self.features_selecionadas)}")
        
        # Selecionar features e target
        X = df[features_disponiveis].copy()
        y = df['Situação'].copy()
        
        # Converter colunas numéricas
        numeric_columns = ['Faltas Consecutivas', 'Módulo atual', 'Cód.Curso', 'Pend. Financ.', 'Identidade']
        numeric_columns = [col for col in numeric_columns if col in X.columns]
        
        for col in numeric_columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        
        # Tratar valores ausentes
        if numeric_columns:
            imputer_num = SimpleImputer(strategy='median')
            X[numeric_columns] = imputer_num.fit_transform(X[numeric_columns])
        
        categorical_columns = ['Curso', 'Currículo', 'Sexo', 'Disciplina atual', 'Pend. Acad.', 'Turma Atual', 'Cód.Disc. atual']
        categorical_columns = [col for col in categorical_columns if col in X.columns]
        
        if categorical_columns:
            imputer_cat = SimpleImputer(strategy='most_frequent')
            X[categorical_columns] = imputer_cat.fit_transform(X[categorical_columns])
        
        # Codificar variáveis categóricas
        for col in categorical_columns:
            if col in X.columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
        
        # Codificar target
        le_target = LabelEncoder()
        y_encoded = le_target.fit_transform(y)
        self.label_encoders['target'] = le_target
        
        # Garantir que não há NaNs
        X = X.fillna(0)
        
        print(f"Dados preprocessados: {X.shape}")
        print(f"Classes no target: {len(le_target.classes_)}")
        
        return X, y_encoded, le_target.classes_
    
    def treinar_modelo(self, X, y):
        """Treinar modelo XGBoost"""
        print("Treinando modelo XGBoost...")
        
        # Dividir dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Dados de treino: {X_train.shape}")
        print(f"Dados de teste: {X_test.shape}")
        
        # Configurar modelo
        self.model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='mlogloss'
        )
        
        # Treinar modelo
        self.model.fit(X_train, y_train)
        
        # Avaliar modelo
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Acurácia no conjunto de teste: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='accuracy')
        print(f"Acurácia média (CV): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Relatório de classificação
        target_names = self.label_encoders['target'].classes_
        print("\nRelatório de Classificação:")
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        return X_test, y_test, y_pred, accuracy
    
    def gerar_matriz_confusao(self, y_test, y_pred, classes):
        """Gerar e salvar matriz de confusão"""
        print("Gerando matriz de confusão...")
        
        # Criar matriz de confusão
        cm = confusion_matrix(y_test, y_pred)
        
        # Plotar matriz
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=classes, yticklabels=classes)
        plt.title('Matriz de Confusão - Modelo XGBoost')
        plt.xlabel('Predição')
        plt.ylabel('Real')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Salvar
        os.makedirs('output', exist_ok=True)
        plt.savefig('output/matriz_confusao_xgboost_treinamento.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Matriz de confusão salva em: output/matriz_confusao_xgboost_treinamento.png")
    
    def gerar_importancia_features(self, X):
        """Gerar gráfico de importância das features"""
        print("Gerando importância das features...")
        
        # Obter importâncias
        importances = self.model.feature_importances_
        feature_names = X.columns
        
        # Criar DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Plotar
        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df.head(10), x='importance', y='feature')
        plt.title('Top 10 Features Mais Importantes - XGBoost')
        plt.xlabel('Importância')
        plt.tight_layout()
        
        # Salvar
        plt.savefig('output/importancia_features_xgboost_treinamento.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Importância das features salva em: output/importancia_features_xgboost_treinamento.png")
        
        # Mostrar top features
        print("\nTop 10 Features Mais Importantes:")
        for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
            print(f"  {i+1:2d}. {row['feature']:<25}: {row['importance']:.4f}")
        
        return importance_df
    
    def gerar_analise_shap(self, X_sample):
        """Gerar análise SHAP"""
        print("Gerando análise SHAP...")
        
        # Criar explainer
        explainer = shap.TreeExplainer(self.model)
        
        # Calcular SHAP values (amostra pequena para performance)
        sample_size = min(100, len(X_sample))
        X_shap = X_sample.sample(n=sample_size, random_state=42)
        shap_values = explainer(X_shap)
        
        # Summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_shap, show=False)
        plt.tight_layout()
        plt.savefig('output/shap_summary_treinamento.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Bar plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig('output/shap_bar_treinamento.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Análise SHAP salva em: output/shap_summary_treinamento.png e output/shap_bar_treinamento.png")
    
    def salvar_modelo(self):
        """Salvar modelo e informações"""
        print("Salvando modelo...")
        
        os.makedirs('output', exist_ok=True)
        
        # Salvar modelo
        joblib.dump(self.model, 'output/modelo_xgboost_sem_classes_criticas.pkl')
        
        # Salvar informações das classes
        class_info = {
            'classes_mantidas': self.classes_mantidas,
            'classes_criticas_removidas': self.classes_criticas,
            'label_encoders': self.label_encoders,
            'features_selecionadas': self.features_selecionadas
        }
        joblib.dump(class_info, 'output/class_mapping_otimizado.pkl')
        
        print("Modelo salvo em: output/modelo_xgboost_sem_classes_criticas.pkl")
        print("Informações salvas em: output/class_mapping_otimizado.pkl")
    
    def gerar_relatorio_treinamento(self, accuracy, importance_df):
        """Gerar relatório de treinamento"""
        print("Gerando relatório de treinamento...")
        
        relatorio = f"""
RELATÓRIO DE TREINAMENTO - MODELO XGBOOST
=========================================

CONFIGURAÇÃO DO MODELO:
- Algoritmo: XGBoost Classifier
- Features selecionadas: {len(self.features_selecionadas)}
- Classes mantidas: {len(self.classes_mantidas)}
- Classes removidas: {len(self.classes_criticas)}

PERFORMANCE:
- Acurácia no teste: {accuracy:.4f} ({accuracy*100:.2f}%)

CLASSES MANTIDAS:
{chr(10).join([f"  • {classe}" for classe in self.classes_mantidas])}

CLASSES REMOVIDAS (CRÍTICAS):
{chr(10).join([f"  • {classe}" for classe in self.classes_criticas])}

TOP 5 FEATURES MAIS IMPORTANTES:
{chr(10).join([f"  {i+1}. {row['feature']}: {row['importance']:.4f}" for i, (_, row) in enumerate(importance_df.head(5).iterrows())])}

ARQUIVOS GERADOS:
- output/modelo_xgboost_sem_classes_criticas.pkl
- output/class_mapping_otimizado.pkl
- output/matriz_confusao_xgboost_treinamento.png
- output/importancia_features_xgboost_treinamento.png
- output/shap_summary_treinamento.png
- output/shap_bar_treinamento.png

COMO USAR:
1. Use o script predicao_matricula_corrigida.py para fazer predições
2. O modelo carrega automaticamente os arquivos .pkl gerados
3. Forneça um arquivo Excel com os dados dos alunos
"""
        
        # Salvar relatório
        with open('output/relatorio_treinamento_xgboost.txt', 'w', encoding='utf-8') as f:
            f.write(relatorio)
        
        print("Relatório salvo em: output/relatorio_treinamento_xgboost.txt")
        print(relatorio)

def main():
    """Função principal"""
    import sys
    
    if len(sys.argv) != 2:
        print("Uso: python treinar_modelo_xgboost.py <arquivo_base_dados.xlsx>")
        print("Exemplo: python treinar_modelo_xgboost.py upload/Planilhabasedados.xlsx")
        sys.exit(1)
    
    arquivo_base = sys.argv[1]
    
    if not os.path.exists(arquivo_base):
        print(f"Erro: Arquivo não encontrado: {arquivo_base}")
        sys.exit(1)
    
    print("="*60)
    print("TREINAMENTO DO MODELO XGBOOST DE PREDIÇÃO DE EVASÃO")
    print("="*60)
    
    # Inicializar treinador
    treinador = TreinadorXGBoost()
    
    try:
        # Carregar dados
        df = treinador.carregar_dados(arquivo_base)
        
        # Preprocessar dados
        X, y, classes = treinador.preprocessar_dados(df)
        
        # Treinar modelo
        X_test, y_test, y_pred, accuracy = treinador.treinar_modelo(X, y)
        
        # Gerar visualizações
        treinador.gerar_matriz_confusao(y_test, y_pred, classes)
        importance_df = treinador.gerar_importancia_features(X)
        treinador.gerar_analise_shap(X)
        
        # Salvar modelo
        treinador.salvar_modelo()
        
        # Gerar relatório
        treinador.gerar_relatorio_treinamento(accuracy, importance_df)
        
        print("="*60)
        print("✅ TREINAMENTO CONCLUÍDO COM SUCESSO!")
        print("="*60)
        print(f"Acurácia final: {accuracy*100:.2f}%")
        print("Todos os arquivos foram salvos na pasta 'output/'")
        
    except Exception as e:
        print(f"Erro durante o treinamento: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

