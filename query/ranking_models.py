from typing import List
from abc import abstractmethod
from typing import List, Set,Mapping
from index.structure import HashIndex,FileIndex,TermOccurrence
import math
from enum import Enum

class IndexPreComputedVals():
    def __init__(self,index):
        self.index = index
        self.precompute_vals()

    def weight_dict(self) -> dict:
        dict_w = dict()
        for term in self.index.vocabulary:
            occur_list = self.index.get_occurrence_list(term)
            num_docs_with_term = len(occur_list)
            for occur in occur_list:
                if not(occur.doc_id in dict_w):
                    dict_w[occur.doc_id] = [VectorRankingModel.tf_idf(self.doc_count, occur.term_freq, num_docs_with_term)]
                else:
                    dict_w[occur.doc_id].append(VectorRankingModel.tf_idf(self.doc_count, occur.term_freq, num_docs_with_term))
        return dict_w

    def precompute_vals(self):
        """
        Inicializa os atributos por meio do indice (idx):
            doc_count: o numero de documentos que o indice possui
            document_norm: A norma por documento (cada termo é presentado pelo seu peso (tfxidf))
        """
        self.document_norm = dict()
        self.doc_count = self.index.document_count
        dict_w = self.weight_dict()
        for doc_id, w_list in dict_w.items():
            sum = 0
            for w in w_list:
                sum += w**2
            self.document_norm[doc_id] = math.sqrt(sum)      

class RankingModel():
    @abstractmethod
    def get_ordered_docs(self,query:Mapping[str,TermOccurrence],
                              docs_occur_per_term:Mapping[str,List[TermOccurrence]]) -> (List[int], Mapping[int,float]):
        raise NotImplementedError("Voce deve criar uma subclasse e a mesma deve sobrepor este método")

    def rank_document_ids(self,documents_weight):
        doc_ids = list(documents_weight.keys())
        doc_ids.sort(key= lambda x:-documents_weight[x])
        return doc_ids

class OPERATOR(Enum):
  AND = 1
  OR = 2
    
#Atividade 1
class BooleanRankingModel(RankingModel):
    def __init__(self,operator:OPERATOR):
        self.operator = operator

    def intersection_all(self,map_lst_occurrences:Mapping[str,List[TermOccurrence]]) -> List[int]:
        dict_ids = dict()
        for term, lst_occurrences in map_lst_occurrences.items():
            for occur in lst_occurrences:
                if not (occur.term_id in dict_ids):
                    dict_ids[occur.term_id] = [occur.doc_id]
                else:
                    dict_ids[occur.term_id].append(occur.doc_id)
        list_ids = list()
        old_list = None
        for term_id, doc_ids_list in dict_ids.items():
            if old_list == None:
                old_list = doc_ids_list
            else:
                for doc_id in doc_ids_list:
                    if doc_id in old_list:
                        list_ids.append(doc_id)
        set_ids = set(list_ids)
        return set_ids
        
    def union_all(self,map_lst_occurrences:Mapping[str,List[TermOccurrence]]) -> List[int]:
        dict_ids = dict()
        for term, lst_occurrences in map_lst_occurrences.items():
            for occur in lst_occurrences:
                if not (occur.term_id in dict_ids):
                    dict_ids[occur.term_id] = [occur.doc_id]
                else:
                    dict_ids[occur.term_id].append(occur.doc_id)
        list_ids = list()
        for term_id, doc_ids_list in dict_ids.items():
            list_ids += doc_ids_list
        set_ids = set(list_ids)
        return set_ids

    def get_ordered_docs(self,query:Mapping[str,TermOccurrence],
                              map_lst_occurrences:Mapping[str,List[TermOccurrence]]) -> (List[int], Mapping[int,float]):
        """Considere que map_lst_occurrences possui as ocorrencias apenas dos termos que existem na consulta"""
        if self.operator == OPERATOR.AND:
            return self.intersection_all(map_lst_occurrences),None
        else:
            return self.union_all(map_lst_occurrences),None

#Atividade 2
class VectorRankingModel(RankingModel):

    def __init__(self,idx_pre_comp_vals:IndexPreComputedVals):
        self.idx_pre_comp_vals = idx_pre_comp_vals

    @staticmethod
    def tf(freq_term:int) -> float:
        return 1 + math.log(freq_term, 2)

    @staticmethod
    def idf(doc_count:int, num_docs_with_term:int )->float:
        return math.log(doc_count/num_docs_with_term, 2)

    @staticmethod
    def tf_idf(doc_count:int, freq_term:int, num_docs_with_term) -> float:
        tf = VectorRankingModel.tf(freq_term)
        idf = VectorRankingModel.idf(doc_count, num_docs_with_term)
        #print(f"TF:{tf} IDF:{idf} n_i: {num_docs_with_term} N: {doc_count}")
        return tf*idf

    def get_ordered_docs(self,query:Mapping[str,TermOccurrence],
                              docs_occur_per_term:Mapping[str,List[TermOccurrence]]) -> (List[int], Mapping[int,float]):
            #print(query)
            #print('fhglfgh')
            #print(docs_occur_per_term)
            documents_weight = {}

            # pegando a norma dos documentos
            index = FileIndex()
            for term, occur_list in docs_occur_per_term.items():
                for occur in occur_list:
                    index.index(term, occur.doc_id, occur.term_freq)  
            index.finish_indexing()
            pre_comp_vals = IndexPreComputedVals(index)
            #print(pre_comp_vals.document_norm)

            # recalcular o peso por documento
            w_per_doc = pre_comp_vals.weight_dict()
            #print(w_per_doc)

            # encontrar o peso por query
            w_per_query = dict()
            for term, occur in query.items():
                occur_list = pre_comp_vals.index.get_occurrence_list(term)
                num_docs_with_term = len(occur_list)
                for lst in occur_list:
                    if not (lst.doc_id in w_per_query):
                        w_per_query[lst.doc_id] = [VectorRankingModel.tf_idf(pre_comp_vals.doc_count, occur.term_freq, num_docs_with_term)]
                    else:
                        w_per_query[lst.doc_id].append(VectorRankingModel.tf_idf(pre_comp_vals.doc_count, occur.term_freq, num_docs_with_term))
            #print(w_per_query)

            for doc in w_per_doc:
                documents_weight[doc] = 0
                for wd in w_per_doc[doc]:
                    for wq in w_per_query[doc]:
                        documents_weight[doc] += wd * wq
                if pre_comp_vals.document_norm[doc] != 0:
                    documents_weight[doc] /= pre_comp_vals.document_norm[doc]
                else:
                    documents_weight[doc] = 0

            #retona a lista de doc ids ordenados de acordo com o TF IDF
            return self.rank_document_ids(documents_weight),documents_weight

