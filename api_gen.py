def api_url_gen():
    # base api query url
    base_url = "http://export.arxiv.org/api/query?"

    # search parameters
    search_query = 'cat:astro-ph*'
    start = 0
    total_results = 20

    query = "search_query=%s&start=%i&max_results=%i&sortBy=lastUpdatedDate&sortOrder=descending"%(search_query,start,total_results)

    return base_url+query
