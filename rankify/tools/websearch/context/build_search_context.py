from typing import Any

from rankify.tools.websearch.content_scraping.crawler import WebScraper
from rankify.tools.websearch.content_scraping.scrapedResult import print_extracted_result
from rankify.tools.websearch.models.SerpResults import SerpResult
from typing import List, TypeVar

T = TypeVar('T')

async def build_search_context(sources:T, num_result: int = 2) ->T:
    sources = [(i,source) for i,source in enumerate(sources.data['organic']) if sources ][:num_result]
    urls = [source[1]['link'] for source in sources]
    print(f"urls : {urls}")
    scraper = WebScraper(strategies=['no_extraction'])
    multi_results = await scraper.scrape_many(urls)
    # Print multiple URL results
    for url, url_results in multi_results.items():
        print(f"\nResults for {url}:")
        for result in url_results.values():
            print_extracted_result(result)

    sources_with_html:List[T]= []
    for i, source in sources:
        source['html'] = multi_results[urls[i]]['no_extraction'].content
        sources_with_html.append(source)

    print(sources_with_html)

    return  sources_with_html
