import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import nltk
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize.treebank import TreebankWordDetokenizer

detokenizer = TreebankWordDetokenizer()
stemmer = SnowballStemmer("english")

def get_synonyms(word):
    synonyms = []
    antonyms = []

    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonyms.append(l.name())
            if l.antonyms():
                antonyms.append(l.antonyms()[0].name())

    synonyms = list(set(synonyms))
    try:
        synonyms.remove(word)
    except:
        pass
    return synonyms

def punnify_sentence(sentence, use_synonyms=False, is_causal=False, destroy_punless=False, explain=False):
    sentence = sentence[::]
    has_pun = False
    
    for i in range(len(sentence)):
        tgt_word = sentence[i]
        for j in range(len(sentence)):
            src_word = sentence[j]
            # If causal is true, only punnified if we have already read the src word
            if j > i and is_causal:
                continue
            # Don't punnify when the source word is too short
            if len(src_word) <= 3:
                continue
            # Don't punnify when the words come from same root words
            if stemmer.stem(src_word) == stemmer.stem(tgt_word):
                continue
            ## TODO: Use part of speech tagging to skip certain parts of speech
#             if nltk.pos_tag([src_word])=="???":
#                 pass
            
            synonyms = get_synonyms(src_word)
            candidate_sources = [src_word] + (synonyms if use_synonyms else [])
            for candidate_source in candidate_sources:
                embeddable, result = embed_word(candidate_source, tgt_word)
                if embeddable:
                    print(src_word, tgt_word)
                    if candidate_source==src_word:
                        explanation = f" ((Orig={src_word}, Source={tgt_word})) "
                    else:
                        explanation = f" ((Orig={src_word}, Source={tgt_word}, Embedded={candidate_source})) "
                    sentence[i] = result + (explanation if explain else "")
                    has_pun = True
            if tgt_word in candidate_sources:
                embeddable, result = embed_word(src_word, tgt_word)
                if embeddable:
                    print(result)
                print("Intrasentence Synonym found:", src_word, tgt_word)
    
    if not has_pun and destroy_punless:
        return []
    return sentence

def embed_word(src, tgt, min_src_len=4):
    if len(src) >= min_src_len and len(tgt) > len(src):
        for i in range(len(tgt) - len(src) + 1):
            num_same_letters = 0
            for s in range(len(src)):
                if src[s]==tgt[s + i]:
                    num_same_letters += 1
            if num_same_letters >= len(src) - 1:
                print("Embed:", src, tgt, tgt[:i] + src.upper() + tgt[i + s + 1:])
                return True, tgt[:i] + src.upper() + tgt[i + s + 1:]

    return False, None

def flatten(listt):
    return [item for sublist in listt for item in sublist]

# sentence = ["The", "lad", "was", "a", "happy", "grad", "student", "who", "ate", "potatoes", "glad", "gladly"]

### Main Code
f=open("input.txt", "r", encoding="utf8")
string = f.read()

tokenized = [word_tokenize(sent) for sent in sent_tokenize(string)]
print(tokenized)

use_synonyms = False
is_causal = False
destroy_punless = False
explain = False

punnified = [punnify_sentence(sent,
                              use_synonyms=use_synonyms,
                              is_causal=is_causal,
                              destroy_punless=destroy_punless,
                              explain=explain) for sent in tokenized]

detokenized = "\n\n".join([detokenizer.detokenize(_) for _ in punnified])

f = open("output.txt", "w", encoding="utf8")
f.write(detokenized)
f.close()

print(detokenized)
