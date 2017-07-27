from urllib.request import urlopen
from bs4 import BeautifulSoup
import os,pickle
import api_gen

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

def fetch_papers():

    # make directory
    try:
        os.mkdir('./data/')
        os.mkdir('./data/train/')
    except:
        pass

    # read local arxiv data
    try:
        arxiv_data = pickle.load(open('./data/train/arxiv_daily.pkl','rb'))
    except:
        arxiv_data = {}

    # get a url for api
    api_url_list = api_gen.api_url_gen()

    for api_url in api_url_list:
        # open a connection to a URL using urllib2
        weburl = urlopen(api_url)

        # get the result code and print it
        print("result code: " + str(weburl.getcode()))

        # read the data from the URL and
        data = weburl.read()

        # parser the html
        soup = BeautifulSoup(data,"html.parser")
        #print(soup.prettify().encode('ascii','ignore'))
        print(soup.find_all('id')[1:])
        for indentifier in soup.find_all('id')[1:]:
            # open connection to the article page
            paperurl = indentifier.get_text()
            print(paperurl)
            
            # extract the arxiv id
            item_name = paperurl.split('/')[4]
            print(item_name)

            # read the data from the article page
            item_content = onepage(paperurl) 
 
            arxiv_data[item_name] = item_content

    # save updated arxiv data locally
    pickle.dump(arxiv_data,open('./data/train/arxiv_daily.pkl','wb'))


if __name__ == "__main__":
    fetch_papers()
