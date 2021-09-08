from typing import List, Set,Mapping
from nltk.tokenize import word_tokenize
from util.time import CheckTime
from query.ranking_models import RankingModel,VectorRankingModel, IndexPreComputedVals
from index.structure import Index, TermOccurrence
from index.indexer import Cleaner

class QueryRunner:
	def __init__(self,ranking_model:RankingModel,index:Index, cleaner:Cleaner):
		self.ranking_model = ranking_model
		self.index = index
		self.cleaner = cleaner


	def get_relevance_per_query(self) -> Mapping[str,Set[int]]:
		"""
		Adiciona a lista de documentos relevantes para um determinada query (os documentos relevantes foram
		fornecidos no ".dat" correspondente. Por ex, belo_horizonte.dat possui os documentos relevantes da consulta "Belo Horizonte"

		"""
		dic_relevance_docs = {}
		for arquiv in ["belo_horizonte","irlanda","sao_paulo"]:
			with open(f"relevant_docs/{arquiv}.dat") as arq:
				dic_relevance_docs[arquiv] = set(arq.readline().split(","))
		return dic_relevance_docs

	def count_topn_relevant(self,n,respostas:List[int],doc_relevantes:Set[int]) -> int:
		"""
		Calcula a quantidade de documentos relevantes na top n posições da lista lstResposta que é a resposta a uma consulta
		Considere que respostas já é a lista de respostas ordenadas por um método de processamento de consulta (BM25, Modelo vetorial).
		Os documentos relevantes estão no parametro docRelevantes
		"""
		#print(f"Respostas: {respostas} doc_relevantes: {doc_relevantes}")
		#print(n)
		relevance_count = 0
		if respostas != []:
			if n > len(respostas):
				n = len(respostas)
			for top in range(n):
				if respostas[top] in doc_relevantes:
					relevance_count += 1
		return relevance_count

	def compute_precision_recall(self, n:int, lst_docs:List[int],relevant_docs:Set[int]) -> (float,float):
		precision = 0
		recall = 0

		recall = self.count_topn_relevant(n, lst_docs, relevant_docs) / len(relevant_docs)
		if len(lst_docs) != 0:
			precision = self.count_topn_relevant(n, lst_docs, relevant_docs) / len(lst_docs)

		return precision, recall

	def get_query_term_occurence(self, query:str) -> Mapping[str,TermOccurrence]:
		"""
			Preprocesse a consulta da mesma forma que foi preprocessado o texto do documento (use a classe Cleaner para isso).
			E transforme a consulta em um dicionario em que a chave é o termo que ocorreu
			e o valor é uma instancia da classe TermOccurrence (feita no trabalho prático passado).
			Coloque o docId como None.
			Caso o termo nao exista no indic, ele será desconsiderado.
		"""
		#print(self.index)
		map_term_occur = {}
		count_query = dict()
		for term in query.split(' '):
			preprocessed_term = self.cleaner.preprocess_word(term)
			occurrence_list = self.index.get_occurrence_list(preprocessed_term)
			if preprocessed_term in count_query:
				count_query[preprocessed_term] += 1
			else:
				count_query[preprocessed_term] = 1
			if occurrence_list != []:
				term_id = self.index.get_term_id(preprocessed_term)
				for next_occur in occurrence_list:
					if term_id == next_occur.term_id:
						map_term_occur[preprocessed_term] = TermOccurrence(None, term_id, count_query[preprocessed_term])
		return map_term_occur

	def get_occurrence_list_per_term(self, terms:List) -> Mapping[str, List[TermOccurrence]]:
		"""
			Retorna dicionario a lista de ocorrencia no indice de cada termo passado como parametro.
			Caso o termo nao exista, este termo possuirá uma lista vazia
		"""
		dic_terms = dict()
		for term in terms:
			preprocessed_term = self.cleaner.preprocess_word(term)
			dic_terms[preprocessed_term] = self.index.get_occurrence_list(preprocessed_term)
		return dic_terms

	def get_docs_term(self, query:str) -> List[int]:
		"""
			A partir do indice, retorna a lista de ids de documentos desta consulta
			usando o modelo especificado pelo atributo ranking_model
		"""
		#Obtenha, para cada termo da consulta, sua ocorrencia por meio do método get_query_term_occurence
		dic_query_occur = self.get_query_term_occurence(query)

		#obtenha a lista de ocorrencia dos termos da consulta
		terms = query.split(' ')
		dic_occur_per_term_query = self.get_occurrence_list_per_term(terms)

		#utilize o ranking_model para retornar o documentos ordenados considrando dic_query_occur e dic_occur_per_term_query
		return self.ranking_model.get_ordered_docs(dic_query_occur, dic_occur_per_term_query)

	@staticmethod
	def runQuery(query:str, indice:Index, indice_pre_computado:IndexPreComputedVals , map_relevantes:Mapping[str,Set[int]]):
		"""
			Para um daterminada consulta `query` é extraído do indice `index` os documentos mais relevantes, considerando 
			um modelo informado pelo usuário. O `indice_pre_computado` possui valores précalculados que auxiliarão na tarefa. 
			Além disso, para algumas consultas, é impresso a precisão e revocação nos top 5, 10, 20 e 50. Essas consultas estão
			Especificadas em `map_relevantes` em que a chave é a consulta e o valor é o conjunto de ids de documentos relevantes
			para esta consulta.
		"""
		time_checker = CheckTime()

		#PEça para usuario selecionar entre Booleano ou modelo vetorial para intanciar o QueryRunner
		#apropriadamente. NO caso do booleano, vc deve pedir ao usuario se será um "and" ou "or" entre os termos.
		#abaixo, existem exemplos fixos.

		cl = Cleaner(stop_words_file="stopwords.txt",language="portuguese", perform_stop_words_removal=False,perform_accents_removal=False, perform_stemming=False)
		
		qr = QueryRunner(indice, VectorRankingModel(indice_pre_computado), cl)
		time_checker.print_delta("Query Creation")

		#Utilize o método get_docs_term para obter a lista de documentos que responde esta consulta
		resp_list, resp_map = qr.get_docs_term(query)

		time_checker.print_delta("anwered with {len(respostas)} docs")

		#nesse if, vc irá verificar se o termo possui documentos relevantes associados a ele
		#se possuir, vc deverá calcular a Precisao e revocação nos top 5, 10, 20, 50.
		#O for que fiz abaixo é só uma sugestao e o metododo countTopNRelevants podera auxiliar no calculo da revocacao e precisao

		if(query in map_relevantes.keys()):
			arr_top = [5,10,20,50]

			#imprima as top 10 respostas
			for n in arr_top:
				precision, recall = qr.compute_precision_recall(n, list(resp_list), set(map_relevantes[query]))

				print('precision #'+f'{n}'+': '+f'{precision}')
				print('Recall #'+f'{n}'+': '+f'{recall}')
		else:
			print('Termo não existe nos documentos!')

	@staticmethod
	def main():
		#leia o indice (base da dados fornecida)
		'''
			como montar o indice a partir da base de dados?
		'''
		idx = None

		idxPreCom = IndexPreComputedVals(idx)

		#Checagem se existe um documento (apenas para teste, deveria existir)
		print(f"Existe o doc? index.hasDocId(105047)")

		#Instancie o IndicePreCompModelo para pr ecomputar os valores necessarios para a query
		print("Precomputando valores atraves do indice...");
		check_time = CheckTime()
        
		check_time.print_delta("Precomputou valores")

		#encontra os docs relevantes
		'''
			como vai funcionar essa chamada do map relevance se get_relevance_per_query nao eh um metodo estatico?
		'''
		map_relevance = QueryRunner.get_relevance_per_query()
		
		print("Fazendo query...")
		#aquui, peça para o usuário uma query (voce pode deixar isso num while ou fazer um interface grafica se estiver bastante animado ;)
		query = "São Paulo"
		QueryRunner.runQuery(query, idx, idxPreCom, map_relevance)