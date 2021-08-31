from IPython.display import clear_output
from typing import List, Set, Union
from abc import abstractmethod
from functools import total_ordering
from os import path, write
import os
import pickle
import gc

class Index:
    def __init__(self):
        self.dic_index = {}
        self.set_documents = set()

    def index(self, term: str, doc_id: int, term_freq: int):
        if term not in self.dic_index:
            int_term_id = len(self.dic_index)
            self.dic_index[term] = self.create_index_entry(int_term_id)
        else:
            int_term_id = self.get_term_id(term)

        self.set_documents.add(doc_id)
        self.add_index_occur(self.dic_index[term], doc_id, int_term_id, term_freq)

    @property
    def vocabulary(self) -> List:
        return list(self.dic_index)

    @property
    def document_count(self) -> int:
        return len(self.set_documents)

    @abstractmethod
    def get_term_id(self, term: str):
        raise NotImplementedError("Voce deve criar uma subclasse e a mesma deve sobrepor este método")

    @abstractmethod
    def create_index_entry(self, termo_id: int):
        raise NotImplementedError("Voce deve criar uma subclasse e a mesma deve sobrepor este método")

    @abstractmethod
    def add_index_occur(self, entry_dic_index, doc_id: int, term_id: int, freq_termo: int):
        raise NotImplementedError("Voce deve criar uma subclasse e a mesma deve sobrepor este método")

    @abstractmethod
    def get_occurrence_list(self, term: str) -> List:
        raise NotImplementedError("Voce deve criar uma subclasse e a mesma deve sobrepor este método")

    @abstractmethod
    def document_count_with_term(self, term: str) -> int:
        raise NotImplementedError("Voce deve criar uma subclasse e a mesma deve sobrepor este método")

    def finish_indexing(self):
        pass

    def __str__(self):
        arr_index = []
        for str_term in self.vocabulary:
            arr_index.append(f"{str_term} -> {self.get_occurrence_list(str_term)}")

        return "\n".join(arr_index)

    def __repr__(self):
        return str(self)


@total_ordering
class TermOccurrence:
    def __init__(self, doc_id: int, term_id: int, term_freq: int):
        self.doc_id = doc_id
        self.term_id = term_id
        self.term_freq = term_freq

    def write(self, idx_file):
        pickle.dump(self, idx_file)

    def __hash__(self):
        return hash((self.doc_id, self.term_id))

    def __eq__(self, other_occurrence: "TermOccurrence"):
        if other_occurrence is None:
            return False
        return self.term_id == other_occurrence.term_id and self.doc_id == other_occurrence.doc_id

    def __lt__(self, other_occurrence: "TermOccurrence"):
        if other_occurrence is None:
            return False
        return self.term_id < other_occurrence.term_id if self.term_id != other_occurrence.term_id else self.doc_id < other_occurrence.doc_id
    
    def __gt__(self, other_occurrence: "TermOccurrence"):
        if other_occurrence is None:
            return False
        return self.term_id > other_occurrence.term_id if self.term_id != other_occurrence.term_id else self.doc_id > other_occurrence.doc_id
    
    def __str__(self):
        return f"(term_id:{self.term_id} doc: {self.doc_id} freq: {self.term_freq})"

    def __repr__(self):
        return str(self)


# HashIndex é subclasse de Index
class HashIndex(Index):
    def get_term_id(self, term: str):
        return self.dic_index[term][0].term_id

    def create_index_entry(self, term_id: int) -> List:
        return list()  # não entendi a necessidade do term_id, caso for necessário no futuro a gente corrige

    def add_index_occur(self, entry_dic_index: List[TermOccurrence], doc_id: int, term_id: int, term_freq: int):
        entry_dic_index.append(TermOccurrence(doc_id, term_id, term_freq))

    def get_occurrence_list(self, term: str) -> List:
        return list() if term not in self.dic_index else self.dic_index[term]

    def document_count_with_term(self, term: str) -> int:
        return len(self.get_occurrence_list(term))


class TermFilePosition:
    def __init__(self, term_id: int, term_file_start_pos: int = None, doc_count_with_term: int = None):
        self.term_id = term_id

        # a serem definidos após a indexação
        self.term_file_start_pos = term_file_start_pos
        self.doc_count_with_term = doc_count_with_term

    def __str__(self):
        return f"term_id: {self.term_id}, doc_count_with_term: {self.doc_count_with_term}, term_file_start_pos: {self.term_file_start_pos}"

    def __repr__(self):
        return str(self)


class FileIndex(Index):
    TMP_OCCURRENCES_LIMIT = 1000000

    def __init__(self):
        super().__init__()

        self.lst_occurrences_tmp = []
        self.idx_file_counter = 0
        self.str_idx_file_name = None # primeira vez é vazio, então ja cria como None

    def get_term_id(self, term: str):
        return self.dic_index[term].term_id

    def create_index_entry(self, term_id: int) -> TermFilePosition:
        return TermFilePosition(term_id)

    def add_index_occur(self, entry_dic_index: TermFilePosition, doc_id: int, term_id: int, term_freq: int):
        self.lst_occurrences_tmp.append(TermOccurrence(doc_id, term_id, term_freq))
        if len(self.lst_occurrences_tmp) >= FileIndex.TMP_OCCURRENCES_LIMIT:
            self.save_tmp_occurrences()

    def next_from_list(self) -> TermOccurrence or None:
        if len(self.lst_occurrences_tmp) == 0:
            return None
        return self.lst_occurrences_tmp.pop(0)

    def next_from_file(self, file_idx) -> TermOccurrence or None:
        try:
            next_occurrence = pickle.load(file_idx)
        except:
            return None
        else:
            if not next_occurrence:
                return None
            return TermOccurrence(next_occurrence.doc_id, next_occurrence.term_id, next_occurrence.term_freq)

    def save_tmp_occurrences(self):
        # ordena pelo term_id, doc_id
        # Para eficiencia, todo o codigo deve ser feito com o garbage
        # collector desabilitado
        gc.disable()

        # ordena pelo term_id
        self.lst_occurrences_tmp.sort()
        
        # self.idx_file_counter
        '''
            idx_file_counter: No código, você irá criar sempre novos indices, excluindo o antigo. Este atributo será útil para definirmos o nome do arquivo do índice. O novo arquivo do índice chamará occur_index_X em que  𝑋  é o número do mesmo.
        '''
        # self.str_idx_file_name
        '''
            str_idx_file_name: Atributo que armazena o arquivo indice atual. A primeira vez que executarmos save_tmp_occurrences não haverá arquivo criado e, assim str_idx_file_name = None
        '''

        if self.str_idx_file_name == None:
            # primeira vez acessando save_tmp_occurrences
            # nao existe arquivo antigo
            self.str_idx_file_name = f"occur_index_{self.idx_file_counter}.idx"
            file = None # nao existe arquivo antigo 
        else:
            # ja ocorreu acesso ao save_tmp_occurrences
            file = open(self.str_idx_file_name, "rb") # abrir o arquivo antigo
            self.idx_file_counter = self.idx_file_counter + 1 # atualiza contagem pra abrir novo
            self.str_idx_file_name = f"occur_index_{self.idx_file_counter}.idx" # novo nome
            
        ### Abra um arquivo novo faça a ordenação externa: compar sempre a primeira posição
        new_file = open(self.str_idx_file_name, 'wb')

        ### da lista com a primeira possição do arquivo usando os métodos next_from_list e next_from_file
        next_term_from_file = self.next_from_file(file)
        next_term_from_list = self.next_from_list()
        while next_term_from_file is not None or next_term_from_list is not None:
            if next_term_from_list < next_term_from_file or next_term_from_file is None:
                next_term_from_list.write(new_file)
                next_term_from_list = self.next_from_list()
            else:
                print(next_term_from_file)
                next_term_from_file.write(new_file)
                next_term_from_file = self.next_from_file(file)
            ### para armazenar no novo indice ordenado

        # limpar a lista e fechar o arquivo
        self.lst_occurrences_tmp = []
        try:
            file.close()
        except:
            pass
        new_file.close()
        gc.enable()

    def finish_indexing(self):
        if len(self.lst_occurrences_tmp) > 0:
            self.save_tmp_occurrences()

        # Sugestão: faça a navegação e obtenha um mapeamento
        # id_termo -> obj_termo armazene-o em dic_ids_por_termo
        dic_ids_por_termo = {} # ids sao as chaves e as palavras sao os itens
        for str_term, obj_term in self.dic_index.items():
            dic_ids_por_termo[obj_term.term_id] = str_term

        with open(self.str_idx_file_name, 'rb') as idx_file:
            # Usar o next_from_file pra ir pegando cada registro
            next_term_from_file = self.next_from_file(idx_file)
            seek_file = 0
            while next_term_from_file is not None:
                # pegando chave do dic_index
                str_term = dic_ids_por_termo[next_term_from_file.term_id]

                # Quantidade de vezes que a palavra aparece
                # lembrar que eles ja estao em ordem no arquivo
                qtde = self.dic_index[str_term].doc_count_with_term
                if qtde is None: 
                    qtde = 0
                # so ir somando 1
                self.dic_index[str_term].doc_count_with_term = qtde + 1
                # quando aparecer a ultima ocorrencia, vai chegar no valor correto

                # Atualizando a posicao de inicio
                # Tem que atualizar a pos so para a primeira ocorrencia daquela palavra
                if self.dic_index[str_term].term_file_start_pos is None:
                    self.dic_index[str_term].term_file_start_pos = seek_file
                
                # Chamar o proximo e andar com o seeker do arquivo
                next_term_from_file = self.next_from_file(idx_file)
                # Cada registro tem tamanho 94
                seek_file = seek_file + 94         

    def get_occurrence_list(self, term: str) -> List:
        occurrence_list = []
        # existe no dicionario?
        if term in self.dic_index.keys():
            # entao o termo existe e ele tem um id
            term_id = self.dic_index[term].term_id

            # pode occorrer mais de uma vez, por isso eh uma lista

            # abrir o arquivo e ir para a posicao de inicio do termo
            idx_file = open(self.str_idx_file_name, 'rb')
            idx_file.seek(self.dic_index[term].term_file_start_pos)

            # consumir a primeira ocorrencia
            next = self.next_from_file(idx_file)
            # enquanto o next tiver o mesmo term_id do passado na busca vai add
            while (next != None and next.term_id == term_id):
                occurrence_list.append(next)
                # ja ta em ordem, so chamar a proxima ocorrencia
                next = self.next_from_file(idx_file)

            # ja add todos
            idx_file.close()         
        else:
            pass # se nao ta no dicionario, o termo nao ocorre no arquivo
        return occurrence_list

    def document_count_with_term(self, term: str) -> int:
        if term in self.dic_index:
            return self.dic_index[term].doc_count_with_term
        return 0
