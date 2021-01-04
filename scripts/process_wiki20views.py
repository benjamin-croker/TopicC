import csv
import sys
import re
import os

INPUT_FILENAME_WIKI_20_VIEWS = os.path.join('data', 'documents_utf8_filtered_20pageviews.csv')
OUTPUT_DIR = os.path.join('data', 'wiki20views')

STRIP_TITLE_FROM_ARTICLE = True
ARTICLES_PER_FILE = 10000

# Needs to be set to > max article length in chars
csv.field_size_limit(10000000)


# reads one row of the csv file at a time
def read_wiki(filename, strip_title_from_article=False):
    with open(filename, encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=['id', 'text'])

        for i, row in enumerate(reader):
            # title is separated by two spaces
            title, article = row['text'].split('  ', 1)
            title = title.strip()
            article = article.strip()

            if strip_title_from_article:
                # Many articles start with the title, so exlude text at the
                # start matching the title, allowing for extra chars between words

                # This doesn't need to be perfect, but prevents the
                # algorithm from learning to just pick up the first phrase

                try:
                    match_string = title.replace(' ', '.{0,20}')
                    re_match = re.search(
                        match_string.lower(), article[0:2 * len(title)].lower()
                    )

                    if re_match:
                        article = article[re_match.end():].strip()

                # some characters are not handled by the RE
                except re.error:
                    pass

            yield title, article


def open_files(part):
    titles_fn = os.path.join(OUTPUT_DIR, 'titles{0:06d}.txt'.format(part))
    articles_fn = os.path.join(OUTPUT_DIR, 'articles{0:06d}.txt'.format(part))
    
    titles_f = open(titles_fn, 'w', encoding='utf-8')
    articles_f = open(articles_fn, 'w', encoding='utf-8')

    return titles_f, articles_f
    

def main():
    print("Opening {}".format(INPUT_FILENAME_WIKI_20_VIEWS))
    data_gen = read_wiki(INPUT_FILENAME_WIKI_20_VIEWS, STRIP_TITLE_FROM_ARTICLE)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    part = 0
    titles_f, articles_f = open_files(part=0)
    
    print("Processing part {}".format(part))
    
    for i, (title, article) in enumerate(data_gen):
        if i % ARTICLES_PER_FILE == 0 and i > 0:
            part += 1
            print("Processing part {}".format(part))
            titles_f.close()
            articles_f.close()
            titles_f, articles_f = open_files(part=part)
            
        titles_f.write(title + '\n')
        articles_f.write(article + '\n')

    titles_f.close()
    articles_f.close()
    print("Done")


if __name__ == '__main__':
    main()
