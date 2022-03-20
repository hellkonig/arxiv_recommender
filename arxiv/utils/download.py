from asyncore import write
from urllib.request import urlopen
from bs4 import BeautifulSoup
import csv
import os

class api:
    '''
    Generate api query

    input:
    search_query: str
        default: 'all'
    total_results: int
        total number of papers fetched
        default: 10000
    max_results: int
        number of papers fetched per query
        default: 1
    data_dir: str
        the directory to save the data
        default: ./
    '''

    def __init__(
        self, 
        search_query='all', 
        total_results=10000, 
        max_results=1,
        data_dir='./'):

        self.search_query = search_query
        self.total_results = total_results
        self.max_results = max_results

        # initialize the saving csv filt
        try:
            os.mkdir('./data')
        except:
            pass
        self.data_file = data_dir + 'arxiv.csv'
        header = ['arxiv_id', 
                  'title',
                  'abstract', 
                  'primary_category', 
                  'categories']
        with open(self.data_file, 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(header)

    def api_urls(self):
        base_url = "http://export.arxiv.org/api/query?"

        length_api_url_list = self.total_results // self.max_results
        for start_id in range(length_api_url_list):
            start = start_id * self.max_results
            query = base_url + ("search_query=%s&"
                                "start=%i&"
                                "max_results=%i&"
                                "sortBy=lastUpdatedDate&"
                                "sortOrder=descending")%(
                                    self.search_query,
                                    start,
                                    self.max_results)
            parsed_paper_info = self.parse(query)

        # save the parsed paper info
        with open(self.data_file, 'a', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerows(parsed_paper_info)

    def parse(self, api_url):
        # open a connection to a URL using urllib2
        weburl = urlopen(api_url)

        # read the data from the URL and
        data = weburl.read()

        # parser the html
        soup = BeautifulSoup(data,"html.parser")
   
        # retrieve title
        titles = soup.find_all('title')
      
        # retrieve abstract
        abstracts = soup.find_all('summary')

        # retrieve category
        tag_primary_category = 'arxiv:primary_category'
        primary_category = soup.find(tag_primary_category)['term']
        category = []
        for cate_item in soup.find_all('category'):
            category.append(cate_item['term'])

        paper_info_list = []
        for idx, indentifier in enumerate(soup.find_all('id')[1:]):
            # open connection to the article page
            paperurl = indentifier.get_text()
            print(paperurl)
            
            # extract the arxiv id
            item_name = paperurl.split('/')[4]
            print(item_name)

            paper_info_list.append([item_name,  # arxiv id
                                    titles[idx].get_text(),
                                    abstracts[idx].get_text(),
                                    primary_category,
                                    category])

        return paper_info_list