from arxiv_recommender.arxiv_paper_fetcher import download

import os
import yaml

if __name__ == "__main__":
    config_path = os.getcwd() + '/config.yml'
    if os.path.exists(config_path):
        config_file = open(config_path, "r", encoding="utf-8")
        config_data = config_file.read()
        config_file.close()
        config = yaml.load(config_data, Loader=yaml.FullLoader)

        search_query = config['search_query']
        total_results = config['total_results']
        max_results = config['max_results']
        data_dir = config['data_dir']

        print(('The query information are:'
               'search query : %s\n'
               'total_results : %i\n'
               'max_results : %i\n'
               'data_dir : %s\n')%(search_query,
                                   total_results,
                                   max_results,
                                   data_dir))
        paper_downloader = download.api(search_query=search_query,
                                        total_results=total_results,
                                        max_results=max_results,
                                        data_dir=data_dir)
    else:
        print(('The query information are:'
               'search query : all\n'
               'total_results : 100\n'
               'max_results : 1\n'
               'data_dir : ./data/ \n'))
        paper_downloader = download.api()

    paper_downloader.fetch()
