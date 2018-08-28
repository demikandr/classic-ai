# import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('averaged_perceptron_tagger_ru')
# nltk.download('universal_tagset')

import os
import copy

from utils import Phonetic, PoemTemplateLoader, Word2vecProcessor
import nltk.corpus

import numpy as np

# Каталог с общими наборами данных, доступный на проверяющем сервере
# Нет необходимости добавлять файлы из этого каталога в архив с решением
# (подробности см. в описании соревнования)
DATASETS_PATH = os.environ.get('DATASETS_PATH', '../../data')

# Шаблоны стихов: строим их на основе собраний сочинений от организаторов
template_loader = PoemTemplateLoader(os.path.join(DATASETS_PATH, 'classic_poems.json'))

# Word2vec модель для оценки схожести слов и темы: берем из каталога RusVectores.org
word2vec = Word2vecProcessor(os.path.join(DATASETS_PATH, 'rusvectores/web_upos_cbow_300_20_2017.bin.gz'))
# word2vec = Word2vecProcessor(os.path.join(DATASETS_PATH, 'wiki.ru/wiki.ru.vec'))
# word2vec = Word2vecProcessor(os.path.join('wiki.ru.vec'))
# word2vec = Word2vecProcessor(os.path.join(DATASETS_PATH, 'wiki.ru/wiki.ru'))
# word2vec = Word2vecProcessor(os.path.join(DATASETS_PATH, 'cc.ru.300'))

# Словарь ударений: берется из локального файла, который идет вместе с решением
# phonetic = Phonetic('data/words_accent.json.bz2')
phonetic = Phonetic('data/accents.json.bz2')

# Словарь слов-кандидатов по фонетическим формам: строится из набора данных SDSJ 2017
word_by_form = phonetic.form_dictionary_from_csv(os.path.join(DATASETS_PATH, 'sdsj2017_sberquad.csv'))
# word_by_form = phonetic.from_accents_dict()

stopwords = nltk.corpus.stopwords.words('russian')
accents_dict_keys = set(phonetic.accents_dict.keys())

tag_cache = {}

from pymystem3 import Mystem
mystem = Mystem()

# def tag_word(word):
#     if word in tag_cache:
#         return tag_cache[word]
#     tag_cache[word] = nltk.tag.pos_tag([word], lang="rus")[0][1]
#     return tag_cache[word]

def tag_word(word):
    if word in tag_cache:
        return tag_cache[word]
    tag_cache[word] = mystem.analyze(word)[0]['analysis'][0]['gr']
    return tag_cache[word]

def tag_sentence(sent):
    analysis = mystem.analyze(sent)
    print(analysis)
    result = []
    # raise KeyError()
    # try:
    for a in analysis:
            if 'analysis' in a:
            # print(a)
                result.append(a['analysis'][0]['gr'])
    # except Exception as e:
        # print(">>>>>>>>>>>>>>>>>>>>>", e)
        # raise

    return result

def generate_poem(seed, poet_id):
    """
    Алгоритм генерации стихотворения на основе фонетических шаблонов
    """

    # выбираем шаблон на основе случайного стихотворения из корпуса
    template = template_loader.get_random_template(poet_id)
    poem = copy.deepcopy(template)
    print('\n'.join((' '.join(line) for line in poem)))

    # оцениваем word2vec-вектор темы
    seed_vec = word2vec.text_vector(seed)

    # заменяем слова в шаблоне на более релевантные теме
    for li, line in enumerate(poem):
        # print(line)
        tagging = tag_sentence(" ".join(line))
        # tagging = nltk.tag.pos_tag(line, lang='rus')
        print(tagging)
        for ti, token in enumerate(line[:-1]):
            if not token.isalpha():
                continue

            word = token.lower()
            if word in stopwords:
                continue
            if word not in accents_dict_keys:
                print(word, " has no accent")
                continue
            # выбираем слова - кандидаты на замену: максимально похожие фонетически на исходное слово
            form = phonetic.get_form(token)
            # candidate_phonetic_distances = [
            #    (replacement_word, phonetic.sound_distance(replacement_word, word))
            #    for replacement_word in word_by_form[form]
            #    ]
            if not word_by_form[form] or form == (0, 0):
                continue
            # min_phonetic_distance = min(d for w, d in candidate_phonetic_distances)
            # replacement_candidates = [w for w, d in candidate_phonetic_distances if d == min_phonetic_distance]
            # print(ti)
            replacement_candidates = [w for w in word_by_form[form]]

            replacement_candidates.append(token)
            # из кандидатов берем максимально близкое теме слово
            replacement_vecs = (word2vec.word_vector(word) for word in replacement_candidates)
            replacement_candidates, replacement_vecs = list(zip(\
                *[(word, vec) for word, vec in zip(replacement_candidates, replacement_vecs) if vec is not None]))
            if len(replacement_candidates) == 0:
                break
            replacement_distances = word2vec.distances(np.vstack(replacement_vecs), np.array([seed_vec,]))
            word2vec_distances = [
                (replacement_word, replacement_distance)
                for replacement_word, replacement_distance in zip(replacement_candidates, replacement_distances)
                ]
            word2vec_distances.sort(key=lambda pair: pair[1])
            word_is_found = False
            for new_word, _ in word2vec_distances[:100]:
                new_tag = tag_word(new_word) 
                if new_tag == tagging[ti]:
                    print("Found ", new_word, "(", new_tag, " == ", tagging[ti], ")")
                    word_is_found = True
                    break
                else:
                    print("\tDiscarding ", new_word, "(", new_tag, " != ", tagging[ti], ")")
            if word_is_found:
                print(word, new_word)
                new_word = new_word.lower() # doesnt work
                poem[li][ti] = new_word

    # собираем получившееся стихотворение из слов
    generated_poem = '\n'.join([' '.join([token for token in line]) for line in poem])

    return generated_poem
