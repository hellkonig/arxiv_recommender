from urllib.request import urlopen
from bs4 import BeautifulSoup
import pickle

def onepage(paperurl):
    #read the data from the article page
    paper_data = urlopen(paperurl).read()

    # parser the html
    paper_soup = BeautifulSoup(paper_data,"html.parser")
    #print(paper_soup.findAll(attrs={"name":"citation_title"}))

    # title
    paper_title = paper_soup.findAll(
                  attrs={"name":"citation_title"})[0]['content']

    # authors
    paper_authors = []
    for authors in paper_soup.findAll(attrs={"name":"citation_author"}):
        paper_authors.append(authors['content'])

    # abstract
    paper_abstract = (paper_soup.find_all('blockquote'))[0].get_text()

    # subject
    paper_subject = paper_soup.find_all(
                    "span",class_="primary-subject")[0].get_text()

    item = {'title':paper_title,
            'authors':paper_authors,
            'abstract':paper_abstract,
            'subject':paper_subject}

    return item

def fetch_daily():

    # read local arxiv data
    try:
        arxiv_data = pickle.load(open('arxiv_daily.pkl','rb'))
    except:
        arxiv_data = {}

    # open a connection to a URL using urllib2
    weburl = urlopen("https://arxiv.org/list/astro-ph/new")

    # get the result code and print it
    print("result code: " + str(weburl.getcode()))

    # read the data from the URL and
    data = weburl.read()

    # parser the html
    soup = BeautifulSoup(data,"html.parser")
    #print(soup.prettify().encode('ascii','ignore'))
    for identifier in soup.find_all('span',class_='list-identifier'):
        # open connection to the article page
        paperurl = "https://arxiv.org"+identifier.\
                   find_all('a',title='Abstract')[0].get('href')
        print(paperurl)
        item_name = identifier.find_all(
                    'a',title='Abstract')[0].get_text()

        # read the data from the article page
        item_content = onepage(paperurl) 
 
        arxiv_data[item_name] = item_content

    # save updated arxiv data locally
    pickle.dump(arxiv_data,open('arxiv_daily.pkl','wb'))


if __name__ == "__main__":
    fetch_daily()
