import pickle
import json


def read_input(file):
    data = []
    dict_term_id = {}
    while True:
        try:
            line = pickle.load(file)
        except EOFError:
            break
        else:
            value = {'doc_id': line.doc_id, 'term_freq': line.term_freq}
            key = f'term_id {line.term_id}'
            if key not in dict_term_id:
                if dict_term_id != {}:
                    data.append(dict_term_id.copy())
                dict_term_id = {key: [value]}
            else:
                dict_term_id[key].append(value)
    return data


def write_output(data: list):
    with open('data.json', 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def main():
    file_name = 'occur_index.idx'
    with open(file_name, "rb") as file:
        write_output(read_input(file))


if __name__ == '__main__':
    main()
