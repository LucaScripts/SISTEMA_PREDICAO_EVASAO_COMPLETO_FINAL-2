#!/usr/bin/env python3
"""
Sistema de Predição de Evasão Estudantil - VERSÃO COM REGRAS CORRIGIDAS
Incorpora grade curricular e ajusta regras baseado no feedback
"""

import pandas as pd
import numpy as np
import joblib
import shap
import os
import sys
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

class PreditorEvasaoRegrasCorrigidas:
    """
    Preditor de evasão com regras de negócio corrigidas
    """
    
    def __init__(self):
        self.model = None
        self.explainer = None
        self.class_info = None
        self.df_disciplinas = None
        self.df_cursos = None
        self.mapeamento_cursos = {}
        
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
        
        # Contadores para relatório
        self.ajustes_regras = {
            'NC_por_regra': 0,
            'LFR_por_regra': 0,
            'LFI_por_regra': 0,
            'LAC_por_regra': 0,
            'NF_por_regra': 0,
            'MT_por_regra': 0,
            'total_ajustes': 0
        }
        
        # Mapeamento de prefixos para cursos
        self.prefixos_cursos = {
            'ELT': 'Eletrotécnica',
            'ENF': 'Enfermagem',
            'FMC': 'Farmácia', 
            'RAD': 'Radiologia',
            'STB': 'Segurança do Trabalho'
        }
        
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
        
    def carregar_grade_curricular(self):
        """Carregar planilhas de disciplinas e cursos"""
        print("Carregando grade curricular...")
        
        try:
            # Carregar disciplinas - usar linha 1 como header e pular linha 0
            if os.path.exists("upload/disciplinas.xlsx"):
                df_raw = pd.read_excel("upload/disciplinas.xlsx")
                # Usar linha 1 como header (que contém: Código, Disciplina, etc.)
                header_row = df_raw.iloc[1].values
                self.df_disciplinas = pd.read_excel("upload/disciplinas.xlsx", skiprows=2)
                self.df_disciplinas.columns = header_row
                print(f"✓ Disciplinas carregadas: {len(self.df_disciplinas)} registros")
                print(f"  Colunas: {list(self.df_disciplinas.columns)}")
            
            # Carregar cursos - usar linha 1 como header e pular linha 0
            if os.path.exists("upload/cursos.xlsx"):
                df_raw = pd.read_excel("upload/cursos.xlsx")
                # Usar linha 1 como header (que contém: Departamento, Código, etc.)
                header_row = df_raw.iloc[1].values
                self.df_cursos = pd.read_excel("upload/cursos.xlsx", skiprows=2)
                self.df_cursos.columns = header_row
                print(f"✓ Cursos carregados: {len(self.df_cursos)} registros")
                print(f"  Colunas: {list(self.df_cursos.columns)}")
                
            # Criar mapeamento de códigos de curso
            if self.df_cursos is not None and 'Código' in self.df_cursos.columns:
                for _, curso in self.df_cursos.iterrows():
                    codigo = curso.get('Código', '')
                    descricao = curso.get('Descrição', '')
                    if pd.notna(codigo) and pd.notna(descricao):
                        self.mapeamento_cursos[str(codigo).zfill(5)] = descricao
                        
            print("✓ Grade curricular carregada com sucesso")
            
        except Exception as e:
            print(f"AVISO: Erro ao carregar grade curricular: {e}")
            print("Continuando sem informações da grade...")
    
    def obter_info_disciplina_atual(self, aluno_data):
        """Obter informações sobre a disciplina atual do aluno"""
        if self.df_disciplinas is None or 'Código' not in self.df_disciplinas.columns:
            return None, None, None
            
        cod_disc_atual = aluno_data.get('Cód.Disc. atual', '')
        
        if pd.isna(cod_disc_atual) or cod_disc_atual == '':
            return None, None, None
            
        # Buscar disciplina
        disciplina_info = self.df_disciplinas[
            self.df_disciplinas['Código'].astype(str) == str(cod_disc_atual)
        ]
        
        if len(disciplina_info) > 0:
            disc = disciplina_info.iloc[0]
            nome = disc.get('Disciplina', '')
            tipo = disc.get('Tipo', '')
            ch_total = disc.get('C.H. Total', 0)
            return nome, tipo, ch_total
            
        return None, None, None
    
    def verificar_primeira_disciplina(self, aluno_data):
        """Verificar se o aluno está na primeira disciplina do curso"""
        if self.df_disciplinas is None or 'Código' not in self.df_disciplinas.columns:
            return False
            
        cod_disc_atual = aluno_data.get('Cód.Disc. atual', '')
        curso = aluno_data.get('Curso', '')
        
        if pd.isna(cod_disc_atual) or pd.isna(curso):
            return False
            
        # Identificar prefixo do curso
        prefixo_curso = None
        for prefixo, nome_curso in self.prefixos_cursos.items():
            if nome_curso.lower() in curso.lower():
                prefixo_curso = prefixo
                break
                
        if not prefixo_curso:
            return False
            
        # Buscar disciplinas do curso ordenadas
        disciplinas_curso = self.df_disciplinas[
            self.df_disciplinas['Código'].astype(str).str.startswith(prefixo_curso)
        ].sort_values('Código')
        
        if len(disciplinas_curso) > 0:
            primeira_disciplina = disciplinas_curso.iloc[0]['Código']
            return str(cod_disc_atual) == str(primeira_disciplina)
            
        return False
    
    def verificar_curso_completo(self, aluno_data):
        """Verificar se o aluno completou todas as disciplinas do curso"""
        if self.df_disciplinas is None or 'Código' not in self.df_disciplinas.columns:
            return False
            
        curso = aluno_data.get('Curso', '')
        modulo_atual = aluno_data.get('Módulo atual', 0)
        
        if pd.isna(curso):
            return False
            
        # Identificar prefixo do curso
        prefixo_curso = None
        for prefixo, nome_curso in self.prefixos_cursos.items():
            if nome_curso.lower() in curso.lower():
                prefixo_curso = prefixo
                break
                
        if not prefixo_curso:
            return False
            
        # Contar disciplinas do curso
        disciplinas_curso = self.df_disciplinas[
            self.df_disciplinas['Código'].astype(str).str.startswith(prefixo_curso)
        ]
        
        total_disciplinas = len(disciplinas_curso)
        
        # Heurística: se módulo atual é alto e próximo do total, pode ter completado
        try:
            modulo_atual = float(modulo_atual) if pd.notna(modulo_atual) else 0
            # Assumindo que cada módulo tem cerca de 3-4 disciplinas
            modulos_esperados = max(1, total_disciplinas // 3)
            return modulo_atual >= modulos_esperados
        except:
            return False
    
    def aplicar_regras_negocio_melhoradas(self, aluno_data, predicao_ml, prob_ml):
        """
        Aplicar regras de negócio melhoradas do Grau Técnico
        """
        
        # Extrair dados do aluno
        faltas_consecutivas = aluno_data.get('Faltas Consecutivas', 0)
        pend_financ = aluno_data.get('Pend. Financ.', 0)
        pend_acad = aluno_data.get('Pend. Acad.', '')
        situacao_atual = aluno_data.get('Situação', '')
        
        # Converter para numérico se necessário
        try:
            faltas_consecutivas = float(faltas_consecutivas) if pd.notna(faltas_consecutivas) else 0
        except:
            faltas_consecutivas = 0
            
        try:
            if pd.notna(pend_financ) and str(pend_financ).upper() != 'PC':
                pend_financ = float(pend_financ)
            else:
                pend_financ = 0  # PC = Pagamento Completo
        except:
            pend_financ = 0
        
        # Obter informações da disciplina atual
        nome_disc, tipo_disc, ch_disc = self.obter_info_disciplina_atual(aluno_data)
        
        # Verificar se é primeira disciplina
        eh_primeira_disciplina = self.verificar_primeira_disciplina(aluno_data)
        
        # Verificar se completou o curso
        curso_completo = self.verificar_curso_completo(aluno_data)
        
        # Aplicar regras em ordem de prioridade
        
        # REGRA 1: NC (Nunca Compareceu) - MELHORADA
        # 5 faltas consecutivas E está na primeira disciplina
        if faltas_consecutivas >= 5:
            if eh_primeira_disciplina:
                self.ajustes_regras['NC_por_regra'] += 1
                return 'Nunca Compareceu', 0.95, 'Regra NC: ≥5 faltas na primeira disciplina'
            else:
                # Se não é primeira disciplina, pode ser LFR
                if faltas_consecutivas >= 12:
                    self.ajustes_regras['LFR_por_regra'] += 1
                    return 'Limpeza de Frequencia', 0.90, 'Regra LFR: ≥12 faltas (não primeira disciplina)'
        
        # REGRA 2: LFI (Limpeza Financeira) - AJUSTADA PARA ≥2 PARCELAS
        # Aluno com 2 ou mais taxas de parcelas em aberto (não 3)
        if pend_financ >= 2:
            self.ajustes_regras['LFI_por_regra'] += 1
            return 'Limpeza Financeira', 0.90, 'Regra LFI: ≥2 parcelas em aberto'
        
        # REGRA 3: LFR (Limpeza de Frequência) - GRAU TÉCNICO
        # Ausência de pagamentos + 12 faltas consecutivas
        if pend_financ > 0 and faltas_consecutivas >= 12:
            self.ajustes_regras['LFR_por_regra'] += 1
            return 'Limpeza de Frequencia', 0.90, 'Regra LFR: Pend. financeira + ≥12 faltas'
        
        # REGRA 4: LAC (Limpeza Acadêmica) - GRAU TÉCNICO
        # Aluno que possui disciplina a cursar e não está vinculado a disciplina ativa
        if pd.notna(pend_acad) and str(pend_acad).strip() != '' and str(pend_acad).upper() != 'NÃO':
            self.ajustes_regras['LAC_por_regra'] += 1
            return 'Limpeza Academica', 0.85, 'Regra LAC: Pendência acadêmica'
        
        # REGRA 5: NF (Não Formado) - MELHORADA
        # Aluno que completou grade curricular mas possui ≤2 parcelas em aberto
        if curso_completo and 0 < pend_financ <= 2:
            self.ajustes_regras['NF_por_regra'] += 1
            return 'Não Formados', 0.80, 'Regra NF: Curso completo + ≤2 parcelas'
        
        # REGRA 6: MT (Matriculado) - GRAU TÉCNICO
        # Aluno sem pendências significativas
        if pend_financ == 0 and faltas_consecutivas < 5:
            self.ajustes_regras['MT_por_regra'] += 1
            return 'Matriculado', 0.85, 'Regra MT: Sem pendências significativas'
        
        # Se nenhuma regra se aplica, usar predição do ML
        return predicao_ml, prob_ml, 'Predição ML'
    
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
        """Analisar alunos e gerar predições com regras melhoradas"""
        print("Processando dados dos alunos com regras de negócio melhoradas...")
        
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
        
        # Pré-processar dados para ML
        X_processado = self.preprocessar_dados(df)
        
        print("Fazendo predições ML + aplicando regras melhoradas...")
        
        # Fazer predições ML
        predicoes_ml = self.model.predict(X_processado)
        probabilidades_ml = self.model.predict_proba(X_processado)
        
        # Calcular SHAP values
        print("Calculando importância das features...")
        shap_values = self.explainer(X_processado).values
        
        # Processar resultados
        resultados = []
        alunos_matriculados = 0
        alunos_evasao = 0
        
        # Resetar contadores
        for key in self.ajustes_regras:
            self.ajustes_regras[key] = 0
        
        for i in range(len(X_processado)):
            # Informações básicas do aluno
            if i < len(dados_originais):
                aluno = dados_originais.iloc[i]
                nome = aluno.get('Nome', f'Aluno_{i+1}')
                
                # Usar a coluna "Matrícula" corretamente
                matricula = aluno.get('Matrícula')
                if pd.isna(matricula) or matricula == '' or matricula is None:
                    matricula = aluno.get('Identidade')
                if pd.isna(matricula) or matricula == '' or matricula is None:
                    matricula = f'ID_{i+1:04d}'
                
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
                aluno = {}
            
            # Predição ML inicial
            classe_predita_idx_ml = int(predicoes_ml[i])
            if classe_predita_idx_ml < len(self.classes_mantidas):
                classe_predita_ml = self.classes_mantidas[classe_predita_idx_ml]
                prob_predita_ml = probabilidades_ml[i][classe_predita_idx_ml]
            else:
                classe_predita_ml = 'Matriculado'
                prob_predita_ml = 0.5
            
            # Aplicar regras de negócio melhoradas
            classe_final, prob_final, fonte_predicao = self.aplicar_regras_negocio_melhoradas(
                aluno, classe_predita_ml, prob_predita_ml
            )
            
            # Contar ajustes
            if fonte_predicao != 'Predição ML':
                self.ajustes_regras['total_ajustes'] += 1
            
            # Contar alunos
            if classe_final == 'Matriculado':
                alunos_matriculados += 1
                status = 'MATRICULADO'
                urgencia = 'NENHUMA'
            else:
                alunos_evasao += 1
                status = 'RISCO_EVASAO'
                
                # Classificar urgência da evasão
                if prob_final >= 0.8:
                    urgencia = 'URGENTE'
                elif prob_final >= 0.6:
                    urgencia = 'ALTA'
                else:
                    urgencia = 'MEDIA'
            
            # Feature mais importante (SHAP)
            fatores_principais = ['Pend. Financ.', 'Faltas Consecutivas', 'Pend. Acad.']
            try:
                shap_abs = np.abs(shap_values[i])
                if shap_abs.ndim > 1:
                    shap_abs = shap_abs.sum(axis=0)
                
                indices_fatores = [j for j, col in enumerate(X_processado.columns) if col in fatores_principais]
                if indices_fatores:
                    shap_fatores = shap_abs[indices_fatores]
                    feature_mais_importante_idx_local = np.argmax(shap_fatores)
                    feature_mais_importante_idx = indices_fatores[feature_mais_importante_idx_local]
                    feature_mais_importante = X_processado.columns[feature_mais_importante_idx]
                    importancia_valor = float(shap_abs[feature_mais_importante_idx])
                else:
                    feature_mais_importante = "Pend. Financ."
                    importancia_valor = 0.0
            except Exception as e:
                feature_mais_importante = "Pend. Financ."
                importancia_valor = 0.0
            
            # Probabilidade de evasão total
            prob_evasao_total = 0.0
            for j, classe in enumerate(self.classes_mantidas):
                if classe in self.situacoes_evasao and j < len(probabilidades_ml[i]):
                    prob_evasao_total += probabilidades_ml[i][j]
            
            # Se foi ajustado por regra, ajustar prob_evasao_total
            if fonte_predicao != 'Predição ML' and classe_final in self.situacoes_evasao:
                prob_evasao_total = prob_final
            
            resultado = {
                'Nome': nome,
                'Matricula': matricula,
                'Situacao_Atual_Sistema': situacao_atual,
                'Curso': curso,
                'Sexo': sexo,
                'Turma': turma,
                'Status_Predicao': status,
                'Situacao_Predita': classe_final,
                'Probabilidade_Situacao': f"{prob_final:.1%}",
                'Probabilidade_Evasao_Total': f"{prob_evasao_total:.1%}",
                'Nivel_Urgencia': urgencia,
                'Fator_Principal': feature_mais_importante,
                'Valor_Importancia': f"{importancia_valor:.4f}",
                'Confianca_Predicao': 'Alta' if prob_final > 0.7 else 'Média' if prob_final > 0.5 else 'Baixa',
                'Fonte_Predicao': fonte_predicao,
                'Predicao_ML_Original': classe_predita_ml,
                'Prob_ML_Original': f"{prob_predita_ml:.1%}"
            }
            
            # Adicionar top 3 situações ML
            probs_ordenadas = [(j, prob) for j, prob in enumerate(probabilidades_ml[i])]
            probs_ordenadas.sort(key=lambda x: x[1], reverse=True)
            
            for k, (idx, prob) in enumerate(probs_ordenadas[:3]):
                if idx < len(self.classes_mantidas):
                    resultado[f'Top_{k+1}_Situacao_ML'] = self.classes_mantidas[idx]
                    resultado[f'Top_{k+1}_Probabilidade_ML'] = f"{prob:.1%}"
                else:
                    resultado[f'Top_{k+1}_Situacao_ML'] = 'N/A'
                    resultado[f'Top_{k+1}_Probabilidade_ML'] = '0.0%'
            
            resultados.append(resultado)
        
        print(f"Alunos processados: {len(resultados)}")
        print(f"  Matriculados: {alunos_matriculados} ({(alunos_matriculados/len(resultados))*100:.1f}%)")
        print(f"  Risco de Evasao: {alunos_evasao} ({(alunos_evasao/len(resultados))*100:.1f}%)")
        print(f"  Ajustes por regras melhoradas: {self.ajustes_regras['total_ajustes']}")
        
        return pd.DataFrame(resultados), alunos_matriculados, alunos_evasao
    
    def gerar_relatorio_completo(self, resultados_df, alunos_matriculados, alunos_evasao, arquivo_saida="analise_completa_regras_melhoradas_final.csv"):
        """Gerar relatório completo com regras melhoradas"""
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
        
        # Análise por fonte de predição
        fonte_counts = resultados_df['Fonte_Predicao'].value_counts()
        
        # Gerar relatório no terminal
        print("="*80)
        print("ANALISE COMPLETA COM REGRAS DE NEGÓCIO MELHORADAS - VERSÃO FINAL")
        print("="*80)
        print("VISAO GERAL:")
        print(f"  Alunos Matriculados (OK): {alunos_matriculados} ({(alunos_matriculados/total_alunos)*100:.1f}%)")
        print(f"  Alunos em Risco de Evasao: {alunos_evasao} ({(alunos_evasao/total_alunos)*100:.1f}%)")
        
        print(f"\nMELHORIAS IMPLEMENTADAS:")
        print(f"  ✓ LFI ajustada: ≥2 parcelas (não 3)")
        print(f"  ✓ NC melhorada: considera primeira disciplina")
        print(f"  ✓ NF melhorada: verifica curso completo")
        print(f"  ✓ Grade curricular integrada")
        
        print(f"\nAPLICAÇÃO DAS REGRAS MELHORADAS:")
        print(f"  Total de ajustes: {self.ajustes_regras['total_ajustes']}")
        print(f"  NC por regra (≥5 faltas + 1ª disciplina): {self.ajustes_regras['NC_por_regra']}")
        print(f"  LFI por regra (≥2 parcelas): {self.ajustes_regras['LFI_por_regra']}")
        print(f"  LFR por regra (pend.fin + ≥12 faltas): {self.ajustes_regras['LFR_por_regra']}")
        print(f"  LAC por regra (pend. acadêmica): {self.ajustes_regras['LAC_por_regra']}")
        print(f"  NF por regra (curso completo + ≤2 parcelas): {self.ajustes_regras['NF_por_regra']}")
        print(f"  MT por regra (sem pendências): {self.ajustes_regras['MT_por_regra']}")
        
        print(f"\nFONTE DAS PREDIÇÕES:")
        for fonte, count in fonte_counts.items():
            pct = (count / total_alunos) * 100
            print(f"  {fonte:<40}: {count:4d} alunos ({pct:5.1f}%)")
        
        if alunos_evasao > 0:
            print(f"\nDISTRIBUICAO DOS ALUNOS EM RISCO POR URGENCIA:")
            print(f"  URGENTE : {urgente:4d} alunos ({(urgente/alunos_evasao)*100:5.1f}% dos em risco)")
            print(f"  ALTA    : {alta:4d} alunos ({(alta/alunos_evasao)*100:5.1f}% dos em risco)")
            print(f"  MEDIA   : {media:4d} alunos ({(media/alunos_evasao)*100:5.1f}% dos em risco)")
            
            # Alunos urgentes
            urgentes_df = evasao_df[evasao_df['Nivel_Urgencia'] == 'URGENTE']
            if len(urgentes_df) > 0:
                print(f"\nALUNOS QUE PRECISAM DE ACAO IMEDIATA ({len(urgentes_df)} alunos):")
                for _, aluno in urgentes_df.head(5).iterrows():
                    print(f"  • {aluno['Nome']} (Matrícula: {aluno['Matricula']})")
                    print(f"    Situacao: {aluno['Situacao_Predita']} - Prob: {aluno['Probabilidade_Situacao']}")
                    print(f"    Fonte: {aluno['Fonte_Predicao']}")
        
        print(f"\nRelatorio completo salvo em: {arquivo_completo}")
        print("O arquivo contem TODOS os alunos com predições ML + regras melhoradas")
        
        return arquivo_completo

def main():
    """Função principal"""
    if len(sys.argv) != 2:
        print("Uso: python predicao_com_regras_corrigidas.py <arquivo_excel>")
        print("Exemplo: python predicao_com_regras_corrigidas.py upload/alunos_ativos_atual.xlsx")
        sys.exit(1)
    
    arquivo_excel = sys.argv[1]
    
    if not os.path.exists(arquivo_excel):
        print(f"Erro: Arquivo não encontrado: {arquivo_excel}")
        sys.exit(1)
    
    print("Inicializando sistema com regras de negócio melhoradas - VERSÃO FINAL...")
    
    # Inicializar preditor
    preditor = PreditorEvasaoRegrasCorrigidas()
    
    try:
        # Carregar modelo
        preditor.carregar_modelo()
        
        # Carregar grade curricular
        preditor.carregar_grade_curricular()
        
        # Analisar alunos
        resultados_df, alunos_matriculados, alunos_evasao = preditor.analisar_alunos(arquivo_excel)
        
        # Gerar relatório
        arquivo_saida = preditor.gerar_relatorio_completo(resultados_df, alunos_matriculados, alunos_evasao)
        
        print(f"\n✅ Análise com regras melhoradas concluída!")
        print(f"📁 Arquivo gerado: {arquivo_saida}")
        print(f"\n📊 SISTEMA COM REGRAS MELHORADAS - VERSÃO FINAL:")
        print(f"   • LFI ajustada: ≥2 parcelas (não 3)")
        print(f"   • NC melhorada: considera primeira disciplina")
        print(f"   • NF melhorada: verifica curso completo")
        print(f"   • Grade curricular integrada")
        print(f"   • Total de ajustes: {preditor.ajustes_regras['total_ajustes']}")
        
    except Exception as e:
        print(f"Erro durante a execução: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

