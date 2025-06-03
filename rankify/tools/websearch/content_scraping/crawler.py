"""
    Web scraper implementation using Crawl4AI
"""
import asyncio
from typing import Dict, List, Optional

from loguru import logger
from crawl4ai import BrowserConfig , CrawlerRunConfig , CacheMode , AsyncWebCrawler
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import PruningContentFilter

from rankify.tools.websearch.content_scraping.scrapedResult import ScrapedResult, print_extracted_result
from rankify.tools.websearch.content_scraping.strategyFactory import StrategyFactory
from rankify.tools.websearch.content_scraping.config import ExtractionConfig


class WebScraper:

    def __init__(self,browser_config:Optional[BrowserConfig]=None,strategies : List[str]= ['no_extraction'],llm_instruction:str="Extract relevant content from the provided text, only return the text, no markdown formatting, remove all footnotes, citations, and other metadata and only keep the main content",
                 query:Optional[str]=None,filter_content:bool=False):
        self.browser_config = browser_config
        self.strategies = strategies
        self.llm_instruction = llm_instruction
        self.query = query
        self.filter_content = filter_content
        self.strategy_factory = StrategyFactory()
        valid_strategies = {'markdown_llm', 'html_llm', 'fit_markdown_llm', 'css', 'xpath', 'no_extraction', 'cosine'}
        invalid_strategies = set(strategies) - valid_strategies
        if invalid_strategies:
            raise ValueError(f"Invalid strategies: {invalid_strategies}")
        self.strategy_map = {
            'markdown_llm': lambda : self.strategy_factory.create_llm_strategy(),
            'cosine': lambda:self.strategy_factory.create_cosine_strategy(),
            'no_extraction': lambda:self.strategy_factory.create_no_extraction_strategy(),
        }

    def _create_crawler_config(self):
        filter_content = PruningContentFilter(user_query=self.query) if self.query else PruningContentFilter()
        return CrawlerRunConfig(cache_mode=CacheMode.BYPASS,markdown_generator=DefaultMarkdownGenerator(content_filter=filter_content))

    async def extract(self, extraction_config: ExtractionConfig, url: str) -> ScrapedResult:
            """ Method to perform extraction using a strategy(e.g., markdown strategy)"""
            try:
                config = self._create_crawler_config()
                config.extraction_strategy = extraction_config.strategy

                async with AsyncWebCrawler(config=self.browser_config) as crawler:
                    if isinstance(url, list):
                        results = await crawler.arun_many(urls=url, config=config)
                    else:
                        results = await crawler.arun(url=url, config=config)


                content = None
                if results.success:
                    if extraction_config.name in ['no_extraction', 'cosine']:
                        if hasattr(results, 'markdown'):
                            content = results.markdown.raw_markdown
                        elif hasattr(results, 'raw_html'):
                            content = results.raw_html
                        elif hasattr(results, 'extracted_content') and results.extracted_content:
                            if isinstance(results.extracted_content, list):
                                content = '\n'.join(item.get('content', '') for item in results.extracted_content)
                            else:
                                content = results.extracted_content
                        if self.filter_content and content:
                            from rankify.tools.websearch.content_scraping.utils import filter_quality_content
                            content = filter_quality_content(content)

                    else:
                        content = results.extracted_content
                        from rankify.tools.websearch.content_scraping.utils import filter_quality_content
                        content = filter_quality_content(content)

                extracted_result = ScrapedResult(
                    name=extraction_config.name,
                    success=results.success,
                    content=content,
                    error=getattr(results, 'error', None)
                )
                if results.success:
                    extracted_result.raw_markdown_len = len(results.markdown.raw_markdown)
                    extracted_result.citations_markdown_len = len(results.markdown.markdown_with_citations)

                return extracted_result

            except Exception as e:
                return ScrapedResult(
                    name=extraction_config.name,
                    success=False,
                    error=str(e)
                )

    async def scrape(self, url: str) -> dict[str, ScrapedResult]:
        """ Scrape URL using the chosen strategy"""
        logger.info(f"Scraping {url}")
        results = {}
        # Handle Wikipedia URLs
        if 'wikipedia.org/wiki/' in url:
            from rankify.tools.websearch.content_scraping.utils import get_wikipedia_content
            try:
                content = get_wikipedia_content(url)
                results['no_extraction'] = ScrapedResult(name='no_extraction', success=True, content=content)

            except Exception as e:
                raise ValueError(f"Debug: Wikipedia extraction failed {str(e)}")

        for strategy_name in self.strategies:
            config = ExtractionConfig(
                name=strategy_name, strategy=self.strategy_map[strategy_name]()
            )
            result = await self.extract(config, url=url)
            results[strategy_name] = result

        return results

    async def scrape_many(self, urls: List[str]) -> Dict[str, Dict[str, ScrapedResult]]:
        """ Extracted multiple URLs concurrently using chosen strategies """
        tasks = [self.scrape(url) for url in urls]
        results_list = await asyncio.gather(*tasks)

        results = {}
        for url, result in zip(urls, results_list):
            results[url] = result

        return results


'''async def main():
    urls = [
        "https://en.wikipedia.org/wiki/Apple_Inc.",
        "https://python.org",
        "https://github.com"
    ]
    scraper = WebScraper(strategies=['no_extraction'])
    multi_results = await scraper.scrape_many(urls)

    # Print multiple URL results
    for url, url_results in multi_results.items():
        print(f"\nResults for {url}:")
        for result in url_results.values():
            print_extracted_result(result)


if __name__ == "__main__":
    asyncio.run(main())'''