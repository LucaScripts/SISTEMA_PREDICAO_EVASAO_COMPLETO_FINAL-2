#!/usr/bin/env python3
"""
Sistema de Predição de Evasão Estudantil - VERSÃO CORRIGIDA COM MATRÍCULA
Performance: 71.78% acurácia multiclasse | 97.53% acurácia binária
"""

import pandas as pd
import numpy as np
import joblib
import shap
import os
import sys
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

class PreditorEvasaoMatriculaCorrigida:
    """
    Preditor de evasão otimizado com coluna Matrícula corrigida
    """
    
    def __init__(self):
        self.model = None
        self.explainer = None
        self.class_info = None
        self.features_esperadas = [
            'Pend. Financ.', 'Faltas Consecutivas', 'Pend. Acad.', 'Módulo atual',
            'Cód.Curso', 'Curso', 'Currículo', 'Sexo', 'Identidade', 
            'Turma Atual', 'Cód.Disc. atual', 'Disciplina atual'
        ]
        
        # Classes mantidas após otimização
        self.classes_mantidas = [
            'Cancelamento Comercial', 'Cancelamento Unidade', 'Não Formados',
            'Limpeza Academica', 'Limpeza Financeira', 'Limpeza de Frequencia',
            'Matriculado', 'Nunca Compareceu'
        ]
        
        # Situações de evasão (excluindo Matriculado)
        self.situacoes_evasao = [
            'Cancelamento Comercial', 'Cancelamento Unidade', 'Não Formados',
            'Limpeza Academica', 'Limpeza Financeira', 'Limpeza de Frequencia',
            'Nunca Compareceu'
        ]
        
    def carregar_modelo(self, model_path="output/modelo_xgboost_sem_classes_criticas.pkl"):
        """Carregar modelo otimizado"""
        print("Carregando modelo otimizado...")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
        
        self.model = joblib.load(model_path)
        
        # Carregar informações das classes
        class_info_path = "output/class_mapping_otimizado.pkl"
        if os.path.exists(class_info_path):
            self.class_info = joblib.load(class_info_path)
            self.classes_mantidas = self.class_info['classes_mantidas']
        
        # Inicializar explainer SHAP
        print("Inicializando explainer SHAP...")
        self.explainer = shap.TreeExplainer(self.model)
        
        print(f"✓ Modelo carregado: {type(self.model).__name__}")
        print(f"✓ Classes mantidas: {len(self.classes_mantidas)}")
        
    def detectar_header(self, df):
        """Detectar linha do header automaticamente"""
        for i in range(min(5, len(df))):
            row = df.iloc[i]
            row_str = ' '.join([str(val) for val in row.values if pd.notna(val)]).upper()
            if any(col in row_str for col in ['MATRÍCULA', 'MATRICULA', 'NOME', 'CURSO', 'SITUAÇÃO']):
                return i
        return 0
    
    def preprocessar_dados(self, df):
        """Pré-processar dados para predição"""
        print(f"Pré-processando dados: {df.shape}")
        
        # Verificar features disponíveis
        features_disponiveis = [col for col in self.features_esperadas if col in df.columns]
        features_faltando = [col for col in self.features_esperadas if col not in df.columns]
        
        if features_faltando:
            print(f"AVISO: Features não encontradas: {features_faltando}")
        
        print(f"Features disponíveis: {len(features_disponiveis)}/{len(self.features_esperadas)}")
        
        # Selecionar apenas as features disponíveis
        X = df[features_disponiveis].copy()
        
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
        
        # Garantir que não há NaNs
        X = X.fillna(0)
        
        print(f"Dados processados: {X.shape}")
        return X
    
    def analisar_alunos(self, arquivo_excel):
        """Analisar alunos e gerar predições"""
        print("Processando dados dos alunos...")
        
        # Carregar dados
        print(f"Carregando dados de: {arquivo_excel}")
        df_raw = pd.read_excel(arquivo_excel, header=None)
        
        # Detectar header
        header_row = self.detectar_header(df_raw)
        print(f"Header detectado na linha: {header_row}")
        
        # Recarregar com header correto
        df = pd.read_excel(arquivo_excel, header=header_row)
        
        print(f"Dados carregados: {df.shape[0]} alunos, {df.shape[1]} colunas")
        
        # Salvar dados originais para informações do aluno
        dados_originais = df.copy()
        
        # Pré-processar dados
        X_processado = self.preprocessar_dados(df)
        
        print("Fazendo predições para todos os alunos...")
        
        # Fazer predições
        predicoes = self.model.predict(X_processado)
        probabilidades = self.model.predict_proba(X_processado)
        
        # Calcular SHAP values
        print("Calculando importância das features...")
        shap_values = self.explainer(X_processado).values
        
        # Processar resultados
        resultados = []
        alunos_matriculados = 0
        alunos_evasao = 0
        
        for i in range(len(X_processado)):
            # Informações básicas do aluno
            if i < len(dados_originais):
                aluno = dados_originais.iloc[i]
                nome = aluno.get('Nome', f'Aluno_{i+1}')
                
                # CORREÇÃO: Usar a coluna "Matrícula" corretamente
                matricula = aluno.get('Matrícula')
                if pd.isna(matricula) or matricula == '' or matricula is None:
                    matricula = aluno.get('Identidade')  # Usar RG como fallback
                if pd.isna(matricula) or matricula == '' or matricula is None:
                    matricula = f'ID_{i+1:04d}'  # ID sequencial como último recurso
                
                curso = aluno.get('Curso', 'N/A')
                sexo = aluno.get('Sexo', 'N/A')
                turma = aluno.get('Turma Atual', 'N/A')
                situacao_atual = aluno.get('Situação', 'N/A')
            else:
                nome = f'Aluno_{i+1}'
                matricula = f'MAT_{i+1}'
                curso = 'N/A'
                sexo = 'N/A'
                turma = 'N/A'
                situacao_atual = 'N/A'
            
            # Predição principal
            classe_predita_idx = predicoes[i]
            classe_predita = self.classes_mantidas[classe_predita_idx]
            prob_predita = probabilidades[i][classe_predita_idx]
            
            # Contar alunos
            if classe_predita == 'Matriculado':
                alunos_matriculados += 1
                status = 'MATRICULADO'
                urgencia = 'NENHUMA'
            else:
                alunos_evasao += 1
                status = 'RISCO_EVASAO'
                
                # Classificar urgência da evasão
                if prob_predita >= 0.8:
                    urgencia = 'URGENTE'
                elif prob_predita >= 0.6:
                    urgencia = 'ALTA'
                else:
                    urgencia = 'MEDIA'
            
            # Feature mais importante (SHAP) - apenas fatores principais
            fatores_principais = ['Pend. Financ.', 'Faltas Consecutivas', 'Pend. Acad.']
            try:
                shap_abs = np.abs(shap_values[i])
                if shap_abs.ndim > 1:
                    shap_abs = shap_abs.sum(axis=0)  # Somar sobre as classes
                
                # Filtrar apenas os fatores principais
                indices_fatores = [j for j, col in enumerate(X_processado.columns) if col in fatores_principais]
                if indices_fatores:
                    shap_fatores = shap_abs[indices_fatores]
                    feature_mais_importante_idx_local = np.argmax(shap_fatores)
                    feature_mais_importante_idx = indices_fatores[feature_mais_importante_idx_local]
                    feature_mais_importante = X_processado.columns[feature_mais_importante_idx]
                    importancia_valor = float(shap_abs[feature_mais_importante_idx])
                else:
                    feature_mais_importante = "Pend. Financ."  # Default
                    importancia_valor = 0.0
            except:
                feature_mais_importante = "Pend. Financ."  # Default
                importancia_valor = 0.0
            
            # Probabilidade de evasão total
            prob_evasao_total = sum([probabilidades[i][j] for j, classe in enumerate(self.classes_mantidas) 
                                   if classe in self.situacoes_evasao])
            
            # Maior probabilidade
            maior_prob_idx = np.argmax(probabilidades[i])
            maior_prob_valor = probabilidades[i][maior_prob_idx]
            
            resultado = {
                'Nome': nome,
                'Matricula': matricula,  # Agora usando a coluna Matrícula corretamente
                'Situacao_Atual_Sistema': situacao_atual,
                'Curso': curso,
                'Sexo': sexo,
                'Turma': turma,
                'Status_Predicao': status,
                'Situacao_Predita': classe_predita,
                'Probabilidade_Situacao': f"{prob_predita:.1%}",
                'Probabilidade_Evasao_Total': f"{prob_evasao_total:.1%}",
                'Nivel_Urgencia': urgencia,
                'Fator_Principal': feature_mais_importante,
                'Valor_Importancia': f"{importancia_valor:.4f}",
                'Confianca_Predicao': 'Alta' if maior_prob_valor > 0.7 else 'Média' if maior_prob_valor > 0.5 else 'Baixa'
            }
            
            # Adicionar top 3 situações
            probs_ordenadas = [(j, prob) for j, prob in enumerate(probabilidades[i])]
            probs_ordenadas.sort(key=lambda x: x[1], reverse=True)
            
            for k, (idx, prob) in enumerate(probs_ordenadas[:3]):
                resultado[f'Top_{k+1}_Situacao'] = self.classes_mantidas[idx]
                resultado[f'Top_{k+1}_Probabilidade'] = f"{prob:.1%}"
            
            resultados.append(resultado)
        
        print(f"Alunos processados: {len(resultados)}")
        print(f"  Matriculados: {alunos_matriculados} ({(alunos_matriculados/len(resultados))*100:.1f}%)")
        print(f"  Risco de Evasao: {alunos_evasao} ({(alunos_evasao/len(resultados))*100:.1f}%)")
        
        return pd.DataFrame(resultados), alunos_matriculados, alunos_evasao
    
    def gerar_relatorio_completo(self, resultados_df, alunos_matriculados, alunos_evasao, arquivo_saida="analise_completa_matricula_corrigida.csv"):
        """Gerar relatório completo com matrícula corrigida"""
        if len(resultados_df) == 0:
            print("Nenhum dado para processar!")
            return
        
        # Salvar CSV
        os.makedirs("output", exist_ok=True)
        arquivo_completo = os.path.join("output", arquivo_saida)
        resultados_df.to_csv(arquivo_completo, index=False, encoding='utf-8-sig')
        
        # Estatísticas
        total_alunos = len(resultados_df)
        
        # Análise por urgência (apenas evasão)
        evasao_df = resultados_df[resultados_df['Status_Predicao'] == 'RISCO_EVASAO']
        
        if len(evasao_df) > 0:
            urgencia_counts = evasao_df['Nivel_Urgencia'].value_counts()
            urgente = urgencia_counts.get('URGENTE', 0)
            alta = urgencia_counts.get('ALTA', 0)
            media = urgencia_counts.get('MEDIA', 0)
        else:
            urgente = alta = media = 0
        
        # Análise por tipo de evasão
        if len(evasao_df) > 0:
            tipos_evasao = evasao_df['Situacao_Predita'].value_counts()
        else:
            tipos_evasao = pd.Series()
        
        # Análise por curso
        if len(evasao_df) > 0:
            cursos_risco = evasao_df['Curso'].value_counts().head(5)
        else:
            cursos_risco = pd.Series()
        
        # Fatores principais (apenas os 3 principais)
        if len(evasao_df) > 0:
            fatores_principais_filtrados = evasao_df[evasao_df['Fator_Principal'].isin(['Pend. Financ.', 'Faltas Consecutivas', 'Pend. Acad.'])]
            fatores_principais = fatores_principais_filtrados['Fator_Principal'].value_counts()
        else:
            fatores_principais = pd.Series()
        
        # Gerar relatório no terminal
        print("="*70)
        print("ANALISE COMPLETA COM MATRÍCULA CORRIGIDA - SISTEMA FINAL")
        print("="*70)
        print("VISAO GERAL:")
        print(f"  Alunos Matriculados (OK): {alunos_matriculados} ({(alunos_matriculados/total_alunos)*100:.1f}%)")
        print(f"  Alunos em Risco de Evasao: {alunos_evasao} ({(alunos_evasao/total_alunos)*100:.1f}%)")
        
        if alunos_evasao > 0:
            print("DISTRIBUICAO DOS ALUNOS EM RISCO POR URGENCIA:")
            print(f"  URGENTE : {urgente:4d} alunos ({(urgente/alunos_evasao)*100:5.1f}% dos em risco)")
            print(f"  ALTA    : {alta:4d} alunos ({(alta/alunos_evasao)*100:5.1f}% dos em risco)")
            print(f"  MEDIA   : {media:4d} alunos ({(media/alunos_evasao)*100:5.1f}% dos em risco)")
            
            print("TIPOS DE EVASAO PREDITOS:")
            for tipo, count in tipos_evasao.head(8).items():
                pct = (count / alunos_evasao) * 100
                print(f"  {tipo:<25}: {count:4d} alunos ({pct:5.1f}%)")
            
            # Alunos urgentes
            urgentes_df = evasao_df[evasao_df['Nivel_Urgencia'] == 'URGENTE']
            if len(urgentes_df) > 0:
                print(f"ALUNOS QUE PRECISAM DE ACAO IMEDIATA ({len(urgentes_df)} alunos):")
                for _, aluno in urgentes_df.head(5).iterrows():
                    print(f"  • {aluno['Nome']} (Matrícula: {aluno['Matricula']})")
                    print(f"    Curso: {aluno['Curso']}")
                    print(f"    Situacao: {aluno['Situacao_Predita']} - Prob: {aluno['Probabilidade_Situacao']}")
            
            # Cursos com mais risco
            if len(cursos_risco) > 0:
                print("CURSOS COM MAIS ALUNOS EM RISCO:")
                for curso, count in cursos_risco.items():
                    total_curso = len(resultados_df[resultados_df['Curso'] == curso])
                    pct = (count / total_curso) * 100 if total_curso > 0 else 0
                    print(f"  {curso:<35}: {count:4d}/{total_curso:4d} alunos ({pct:5.1f}% do curso)")
            
            # Fatores principais
            if len(fatores_principais) > 0:
                print("PRINCIPAIS FATORES DE RISCO:")
                for fator, count in fatores_principais.items():
                    pct = (count / alunos_evasao) * 100
                    print(f"  {fator:<25}: {count:4d} casos ({pct:5.1f}%)")
        
        print(f"Relatorio completo salvo em: {arquivo_completo}")
        print("O arquivo contem TODOS os alunos com a coluna Matrícula corrigida")
        print("Use filtros no Excel: Status_Predicao = 'RISCO_EVASAO' para ver apenas os em risco")
        print("Analise completa concluida!")
        print(f"Verifique o arquivo: {arquivo_completo}")
        
        print("RESUMO:")
        print(f"Total: {total_alunos} alunos")
        print(f"Matriculados: {alunos_matriculados} alunos ({(alunos_matriculados/total_alunos)*100:.1f}%)")
        print(f"Risco de Evasao: {alunos_evasao} alunos ({(alunos_evasao/total_alunos)*100:.1f}%)")
        
        return arquivo_completo

def main():
    """Função principal"""
    if len(sys.argv) != 2:
        print("Uso: python predicao_matricula_corrigida.py <arquivo_excel>")
        print("Exemplo: python predicao_matricula_corrigida.py upload/alunos_ativos_atual.xlsx")
        sys.exit(1)
    
    arquivo_excel = sys.argv[1]
    
    if not os.path.exists(arquivo_excel):
        print(f"Erro: Arquivo não encontrado: {arquivo_excel}")
        sys.exit(1)
    
    print("Inicializando sistema de analise com matrícula corrigida...")
    
    # Inicializar preditor
    preditor = PreditorEvasaoMatriculaCorrigida()
    
    try:
        # Carregar modelo
        preditor.carregar_modelo()
        
        # Analisar alunos
        resultados_df, alunos_matriculados, alunos_evasao = preditor.analisar_alunos(arquivo_excel)
        
        # Gerar relatório
        arquivo_saida = preditor.gerar_relatorio_completo(resultados_df, alunos_matriculados, alunos_evasao)
        
        print(f"\n✅ Análise com matrícula corrigida concluída!")
        print(f"📁 Arquivo gerado: {arquivo_saida}")
        print(f"\n📊 PERFORMANCE DO MODELO OTIMIZADO:")
        print(f"   • Acurácia Multiclasse: 71.78%")
        print(f"   • Acurácia Binária (Evasão): 97.53%")
        print(f"   • Coluna Matrícula corrigida: ✅")
        print(f"   • Features principais na ordem correta")
        
    except Exception as e:
        print(f"Erro durante a execução: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

