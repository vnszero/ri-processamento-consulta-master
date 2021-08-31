from datetime import datetime

from nltk.stem.snowball import SnowballStemmer
from bs4 import BeautifulSoup
import string
from nltk.tokenize import word_tokenize
import os


class Cleaner:
    def __init__(self, stop_words_file: str, language: str,
                 perform_stop_words_removal: bool, perform_accents_removal: bool,
                 perform_stemming: bool):
        self.set_stop_words = self.read_stop_words(stop_words_file)

        self.stemmer = SnowballStemmer(language)
        in_table = "áéíóúâêôçãẽõü"
        out_table = "aeiouaeocaeou"
        # altere a linha abaixo para remoção de acentos (Atividade 11)
        self.accents_translation_table = ''.maketrans(in_table, out_table)
        self.set_punctuation = set(string.punctuation)

        # flags
        self.perform_stop_words_removal = perform_stop_words_removal
        self.perform_accents_removal = perform_accents_removal
        self.perform_stemming = perform_stemming

    def html_to_plain_text(self, html_doc: str) -> str:
        soup = BeautifulSoup(html_doc, parser='lxml')
        return soup.get_text()

    def read_stop_words(self, str_file):
        set_stop_words = set()
        with open(str_file, encoding='utf-8') as stop_words_file:
            for line in stop_words_file:
                arr_words = line.split(",")
                [set_stop_words.add(word) for word in arr_words]
        return set_stop_words

    def is_stop_word(self, term: str):
        return term in self.set_stop_words

    def word_stem(self, term: str):
        return self.stemmer.stem(term)

    def remove_accents(self, term: str) -> str:
        return term.translate(self.accents_translation_table)

    @staticmethod
    def is_accent(term):
        accents = "áéíóúâêôçãẽõü"
        return term in accents

    def preprocess_word(self, term: str) -> str or None:
        if self.is_stop_word(term) or self.is_accent(term):
            return None
        term = term.lower()
        if self.perform_accents_removal:
            term = self.remove_accents(term)
        if self.perform_stemming:
            term = self.word_stem(term)
        return self.remove_dots(term)

    @staticmethod
    def remove_dots(word):
        invalid_dots = (';', '!', '?', ':', ',', '.')
        for dot in invalid_dots:
            word = word.replace(dot, '')
        return word


class HTMLIndexer:
    cleaner = Cleaner(stop_words_file="stopwords.txt",
                      language="portuguese",
                      perform_stop_words_removal=True,
                      perform_accents_removal=True,
                      perform_stemming=True)

    def __init__(self, index):
        self.index = index

    def text_word_count(self, plain_text: str):
        dic_word_count = {}
        token_list = word_tokenize(plain_text)
        for word in token_list:
            processed_word = self.cleaner.preprocess_word(word)
            if processed_word in dic_word_count.keys():
                dic_word_count[processed_word] += 1
            else:
                dic_word_count[processed_word] = 1
        try:
            del dic_word_count['']
            del dic_word_count[None]
        except KeyError:
            pass
        return dic_word_count

    def index_text(self, doc_id: int, text_html: str):
        text_plain = self.cleaner.html_to_plain_text(text_html)
        dict_text_word_count = self.text_word_count(text_plain)
        for key in dict_text_word_count:
            if key:
                self.index.index(key, doc_id, dict_text_word_count[key])

    def index_text_dir(self, path: str):
        for str_sub_dir in os.listdir(path):
            path_sub_dir = self.create_path(path, str_sub_dir)
            self.browse_in_directory(path_sub_dir)

    def browse_in_directory(self, path_sub_dir):
        for file_name in os.listdir(path_sub_dir):
            filename = self.create_path(path_sub_dir, file_name)
            self.index_file(file_name, filename)
            self.write_file(file_name)

    @staticmethod
    def create_path(path, str_sub_dir):
        return f'{path}/{str_sub_dir}'

    @staticmethod
    def write_file(file_name):
        with open("teste.txt", "a", encoding="utf-8") as file:
            file.write(file_name)

    def index_file(self, file_name, filename):
        with open(filename, "rb") as file:
            self.index_text(self.get_doc_id(file_name), file)

    def get_doc_id(self, file_name):
        return int(self.get_first(file_name))

    @staticmethod
    def get_first(file_name):
        return (file_name.split("."))[0]

