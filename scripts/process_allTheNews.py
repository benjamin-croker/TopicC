import pandas as pd
import os

df1 = pd.read_csv(os.path.join('data', 'articles1.csv'))
df2 = pd.read_csv(os.path.join('data', 'articles2.csv'))
df3 = pd.read_csv(os.path.join('data', 'articles3.csv'))
df = pd.concat([df1, df2, df3])

# some of the titles end with " - The New York Times"
df['title'] = df.title.str.replace(' - The New York Times', '', regex=False)
# remove line feeds etc
df['title'] = df.title.str.replace('[^\S ]+', '', regex=True)
df['content'] = df.content.str.replace('[^\S ]+', '', regex=True)
# replace two or more spaces with one space
df['title'] = df.title.str.replace('  +', ' ', regex=True)
df['content'] = df.content.str.replace('  +', ' ', regex=True)
# remove trailing and leading whitespace
df['title'] = df.title.str.strip()
df['content'] = df.content.str.strip()

# use tab separator (there's only one column anyway) to avoid
# text with commas being quoted
df.content.to_csv(
    os.path.join('data', 'allTheNews_articles.txt'),
    header=False, index=False, sep='\t'
)
df.title.to_csv(
    os.path.join('data', 'allTheNews_titles.txt'),
    header=False, index=False, sep='\t'
)