
import os
import json
import subprocess

from pyserini.search.lucene import LuceneSearcher

from rankify.tools.websearch.content_scraping.crawler import WebScraper
from rankify.tools.websearch.content_scraping.scrapedResult import print_extracted_result
from typing import List, TypeVar
import tempfile
T = TypeVar('T')


async def build_search_context(sources:T) ->T:
    if sources is None:
        return []
    sources = [(i,source) for i,source in enumerate(sources.data['organic']) if sources ]
    urls = [source[1]['link'] for source in sources]
    print(f"urls : {urls}")
    scraper = WebScraper(strategies=['no_extraction'])
    multi_results = await scraper.scrape_many(urls)

    sources_with_html:List[T]= []
    for i, source in sources:
        source['html'] = multi_results[urls[i]]['no_extraction'].content
        sources_with_html.append(source)


    return  sources_with_html


