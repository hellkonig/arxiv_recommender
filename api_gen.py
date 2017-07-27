def api_url_gen():
    # base api query url
    base_url = "http://export.arxiv.org/api/query?"

    # search parameters
    search_query = 'cat:astro-ph*'
    total_results = 30 # total number of papers fetched
    max_results = 10  # number of papers fetched per loop

    api_url_list = []

    for i in range(total_results//max_results):
        start = i*max_results
        query = "search_query=%s&start=%i&max_results=%i&sortBy=lastUpdatedDate&sortOrder=descending"%(search_query,start,max_results)
        print(base_url+query)
        api_url_list.append(base_url+query)

    return api_url_list
