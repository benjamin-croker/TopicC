import pandas as pd
import os

df1 = pd.read_csv(os.path.join('data', 'articles1.csv'))
df2 = pd.read_csv(os.path.join('data', 'articles2.csv'))
df3 = pd.read_csv(os.path.join('data', 'articles3.csv'))
df = pd.concat([df1, df2, df3])

df['content'] = df.content.str.replace('[^a-zA-Z0-9 ]', '', regex=True)
df['title'] = df.title.str.replace('[^a-zA-Z0-9 ]', '', regex=True)

df.content.to_csv(
    os.path.join('data', 'allTheNews_articles.txt'),
    header=False, index=False
)
df.title.to_csv(
    os.path.join('data', 'allTheNews_titles.txt'),
    header=False, index=False
)