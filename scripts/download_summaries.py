from bs4 import BeautifulSoup
import urllib.request
import urllib.parse
import urllib.error
import json
import threading
import os

EXCLUDED_LINKS = (
    'Featured articles',
    'Good articles',
    'Template:Icon/doc',
    'Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5',
    '10', '100', '1000', '1,000', '10,000', '25,000',
    't', 'v',
    'Wikipedia:Vital articles',
    'Wikipedia vital articles',
    'About Wikipedia',
    'Wikidata item',
    'Wikipedia core topics',
    'Wikipedia article lists',
    'Talk', 'Help', 'Log in',
    'Disclaimers',
    'Learn to edit',
    'Download as PDF',
    'Project page',
    'Random article',
    'Wikipedia core topics',
    'Community portal',
    'Wikipedia article lists',
    'Page information',
    'Create account',
    'View history',
    'Upload file',
    'What links here'
)
# number of requests
N_REQ_LIMIT = 100
# per seconds
T_REQ_LIMT = 1
# URLS
WIKI_VA_L5_URL = 'https://en.wikipedia.org/wiki/Wikipedia:Vital_articles/Level/5'
WIKI_TITLE_SUMMARY_URL = 'https://en.wikipedia.org/api/rest_v1/page/summary/{}'
WIKI_URL = 'https://en.wikipedia.org'


def get_url(url, max_retries=2):
    # attempt twice
    retries = 0
    while retries < max_retries:
        try:
            req = urllib.request.urlopen(url)
            data = req.read()
            charset = req.headers.get_content_charset('utf-8')
            return data.decode(charset)

        except urllib.error.URLError:
            retries += 1

    raise urllib.error.URLError("Exceeded max retries")


# Gets the URLs for category subpages from the Level 5 Vital Articles Main Page
def get_va_lvl5_subpage_urls(base_url=WIKI_URL, va_lvl5_url=WIKI_VA_L5_URL):
    va_lvl5_soup = BeautifulSoup(get_url(va_lvl5_url), 'html.parser')
    va_lvl5_subpage_link_tags = va_lvl5_soup.find('table', attrs={'class': 'wikitable'}).find_all('a')
    va_lvl5_subpage_urls = [base_url + a['href'] for a in va_lvl5_subpage_link_tags]
    return va_lvl5_subpage_urls


# Extracts the article titles from the category subpage
def get_subpage_titles(va_lvl5_subpage_url, excluded_links=EXCLUDED_LINKS):
    print(f"Getting information from {va_lvl5_subpage_url}")
    va_lvl5_subpage_soup = BeautifulSoup(get_url(va_lvl5_subpage_url), 'html.parser')
    # Remove the footer table and sidebar
    va_lvl5_subpage_soup.find('div', {'aria-labelledby': 'Wikipedia_core_topic_lists'}).decompose()
    va_lvl5_subpage_soup.find('div', {'id': 'mw-navigation'}).decompose()
    va_lvl5_subpage_soup.find('footer', {'class': 'mw-footer'}).decompose()
    # Find all links to pages in the Vital Article list.
    # They are links within list elements.
    links = [li.find_all('a', title=True) for li in va_lvl5_subpage_soup.find_all('li')]
    # flatten the lists
    links = sum(links, [])
    # Get the page titles, but exclude links to other summary information/
    # Also, get the disambiguated title, rather than the text
    # E.g. extract "Stripping (chemistry)" from
    # <a href="/wiki/Stripping_(chemistry)" title="Stripping (chemistry)">Stripping</a>
    titles = [link['title'] for link in links if link.text not in excluded_links]
    # remove duplicates
    return list(set(titles))


# Uses the REST API to get the page summaries
def get_title_summary(title, api_summary_endpoint=WIKI_TITLE_SUMMARY_URL):
    # escape '/' chars in the title, by setting safe to an empty string
    q_url = api_summary_endpoint.format(urllib.parse.quote(title, safe=""))
    try:
        return json.loads(get_url(q_url))['extract']
    except:
        print(f"error extracting summary for {title}")
        return ''


# runs a batch of queries with a rate limiter
def rate_limit_query(qry_fun, args, n_req, t_limit):
    # timer is just used as a timer, no action performed
    def _f():
        pass

    results = []

    for i in range(0, len(args), n_req):
        print(f"Query {i} to {i + n_req} of {len(args)}")
        t_timer = threading.Timer(t_limit, _f)
        t_timer.start()
        results += [qry_fun(arg) for arg in args[i:(i + n_req)]]
        print("Waiting...")
        t_timer.join()

    return results


# Uses the REST API to get the page summaries for a list of titles
def get_subpage_summaries(titles, n_req_limit=N_REQ_LIMIT, t_req_limit=T_REQ_LIMT):
    summaries = rate_limit_query(get_title_summary, titles, n_req_limit, t_req_limit)
    summaries = [s.replace('\n', '') for s in summaries]

    if len(titles) != len(summaries):
        raise ValueError("titles and summaries should have the same length")

    return summaries


def write_files(categories, titles, summaries):
    with open(os.path.join('../data', 'wikiVAlvl5_categories.txt'), 'w') as f:
        f.writelines([(t + '\n') for t in categories])

    with open(os.path.join('../data', 'wikiVAlvl5_titles.txt'), 'w') as f:
        f.writelines([(t + '\n') for t in titles])

    with open(os.path.join('../data', 'wikiVAlvl5_summaries.txt'), 'w') as f:
        f.writelines([(s + '\n') for s in summaries])

    categories_unique = set(categories)
    category_labels = {c: i for i, c in enumerate(categories_unique)}
    with open(os.path.join('../data', 'wikiVAlvl5_category_labels.json'), 'w') as f:
        json.dump(category_labels, f, indent=2)


def get_all_summaries(va_lvl5_subpage_url, va_lvl5_url=WIKI_VA_L5_URL):
    titles = get_subpage_titles(va_lvl5_subpage_url)
    summaries = get_subpage_summaries(titles)
    # get the category from the url
    category = va_lvl5_subpage_url.replace(va_lvl5_url + '/', '').replace('/', '_').replace(',', '')
    # repeat category for each title
    categories = [category] * len(titles)
    return categories, titles, summaries


def main():
    categories = []
    titles = []
    summaries = []

    va_lvl5_subpage_urls = get_va_lvl5_subpage_urls()
    for va_lvl5_subpage_url in va_lvl5_subpage_urls:
        c, t, s = get_all_summaries(va_lvl5_subpage_url)
        categories += c
        titles += t
        summaries += s

    write_files(categories, titles, summaries)


if __name__ == '__main__':
    main()
