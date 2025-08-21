"""
Web scraping functionality using crawl4ai with advanced crawling strategies.
"""

import asyncio
import logging
import re
from typing import Any
from urllib.parse import urljoin, urlparse

try:
    import httpx
except ImportError:
    raise ImportError("httpx is required. Install with: pip install httpx") from None

try:
    from crawl4ai import (
        AsyncWebCrawler,
        BrowserConfig,
        CacheMode,
        CrawlerRunConfig,
        MemoryAdaptiveDispatcher,
    )
    from crawl4ai.content_filter_strategy import PruningContentFilter
    from crawl4ai.deep_crawling import BestFirstCrawlingStrategy, BFSDeepCrawlStrategy
    from crawl4ai.deep_crawling.filters import FilterChain, URLPatternFilter
    from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer
    from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
except ImportError:
    raise ImportError("crawl4ai is required. Install with: pip install crawl4ai") from None

from ..config.config import WebVectorConfig
from ..errors.exceptions import ScrapingError

logger = logging.getLogger(__name__)


class WebScraper:
    """Advanced web scraper using Crawl4AI with concurrency control and deep crawling."""

    def __init__(self, config: WebVectorConfig):
        self.config = config
        self.crawler = None

    async def __aenter__(self):
        browser_cfg = BrowserConfig(
            headless=True,
            java_script_enabled=True
        )
        self.crawler = AsyncWebCrawler(config=browser_cfg, verbose=True)
        await self.crawler.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.crawler:
            await self.crawler.__aexit__(exc_type, exc_val, exc_tb)

    def _make_dispatcher(self) -> MemoryAdaptiveDispatcher:
        return MemoryAdaptiveDispatcher(
            memory_threshold_percent=80.0,
            check_interval=1.0,
            max_session_permit=10
        )

    def _make_run_config(self) -> CrawlerRunConfig:
        md_gen = DefaultMarkdownGenerator(
            content_filter=PruningContentFilter(threshold=0.2, threshold_type="fixed")
        )
        return CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            wait_for=None,
            markdown_generator=md_gen,
            stream=True
        )

    def _create_crawl_strategy(self):
        """Choose Best-First (keyword-based) or BFS deep crawling."""
        url_scorer = (
            KeywordRelevanceScorer(keywords=self.config.keywords, weight=0.7)
            if self.config.keywords else None
        )

        filters = []
        if self.config.url_patterns:
            filters.extend(URLPatternFilter(patterns=[p]) for p in self.config.url_patterns)
        filter_chain = FilterChain(filters) if filters else None

        params = dict(
            max_depth=self.config.max_depth,
            include_external=self.config.include_external,
            max_pages=self.config.max_pages,
            filter_chain=filter_chain
        )

        if url_scorer:
            return BestFirstCrawlingStrategy(url_scorer=url_scorer, **params)
        return BFSDeepCrawlStrategy(**params)

    async def scrape_website(self, url: str) -> list[dict[str, Any]]:
        """Scrape a single website with deep crawling enabled."""
        if not self.crawler:
            raise ScrapingError("Use async context manager to initialize the scraper.")

        dispatcher = self._make_dispatcher()
        run_cfg = self._make_run_config()
        strategy = self._create_crawl_strategy()

        scraped_data: list[dict[str, Any]] = []

        try:
            # Step 1: Crawl the root page to discover internal links (like the working example)
            root_res = await self.crawler.arun(url=url, config=run_cfg)

            discovered_urls: list[str] = []
            try:
                internal_links = []
                # crawl4ai exposes links as a dict with keys like "internal" / "external"
                if hasattr(root_res, 'links') and isinstance(root_res.links, dict):
                    internal_links = root_res.links.get("internal", []) or []
                logger.info(f"Root discovery: found {len(internal_links)} internal links from page HTML")

                # Build absolute internal URLs constrained to the final resolved netloc (normalize www)
                resolved_root = getattr(root_res, 'url', url) or url
                def _norm(n: str) -> str:
                    return n.lower().lstrip().rstrip().removeprefix("www.")
                base_netloc = _norm(urlparse(resolved_root).netloc)
                for link in internal_links:
                    href = link.get("href") if isinstance(link, dict) else None
                    if not href:
                        continue
                    abs_url = urljoin(resolved_root, href)
                    link_netloc = _norm(urlparse(abs_url).netloc)
                    if not link_netloc or link_netloc == base_netloc:
                        discovered_urls.append(abs_url)

                # Always include the homepage
                discovered_urls.append(url)
                logger.info(f"After adding homepage: {len(discovered_urls)} URLs so far")

                # Step 1b: merge in sitemap/robots discovered URLs
                try:
                    # Use resolved root URL as base for sitemap/robots discovery
                    sitemap_urls = await self._discover_urls(resolved_root)
                except Exception:
                    sitemap_urls = []
                logger.info(f"Sitemap/robots discovery: found {len(sitemap_urls)} URLs")

                # Filter sitemap URLs to same netloc just in case
                for s_url in sitemap_urls:
                    if not s_url:
                        continue
                    s_netloc = _norm(urlparse(s_url).netloc)
                    if not s_netloc or s_netloc == base_netloc:
                        discovered_urls.append(s_url)
                # Deduplicate while preserving order
                seen = set()
                deduped = []
                for u in discovered_urls:
                    if u not in seen:
                        seen.add(u)
                        deduped.append(u)
                logger.info(f"Total discovered (deduped) URLs: {len(deduped)}")

                # Respect max_pages if provided
                max_pages = self.config.max_pages or len(deduped)
                target_urls = deduped[:max_pages]
            except Exception:
                target_urls = [url]

            # If we discovered more than just the homepage, crawl that set directly
            if len(target_urls) > 1:
                logger.info(f"Discovered {len(target_urls)} URLs to crawl from root {url}.")
                logger.debug(f"First 10 target URLs: {target_urls[:10]}")
                async for result in await self.crawler.arun_many(
                    urls=target_urls,
                    config=run_cfg,
                    dispatcher=dispatcher,
                ):
                    if result.success and getattr(result, 'markdown', None):
                        content = getattr(
                            result.markdown,
                            'fit_markdown',
                            result.markdown.raw_markdown
                        )
                        scraped_data.append({
                            'url': result.url,
                            'content': content,
                            'status': result.status_code
                        })
                return scraped_data

            # Fallback: use deep crawling strategy if discovery found nothing meaningful
            async for result in await self.crawler.arun_many(
                urls=[url],
                config=run_cfg,
                dispatcher=dispatcher,
                strategy=strategy
            ):
                if result.success and getattr(result, 'markdown', None):
                    content = getattr(
                        result.markdown,
                        'fit_markdown',
                        result.markdown.raw_markdown
                    )
                    scraped_data.append({
                        'url': result.url,
                        'content': content,
                        'status': result.status_code
                    })
            return scraped_data
        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            raise ScrapingError(f"Failed to scrape {url}: {str(e)}") from e

    def _process_page_result(self, result: Any, url: str) -> dict[str, Any]:
        """Process a single page result into standardized format."""
        content = ""
        if hasattr(result, 'extracted_content') and result.extracted_content:
            content = result.extracted_content
        elif hasattr(result, 'markdown') and result.markdown:
            try:
                md = result.markdown
                content = getattr(md, 'fit_markdown', None) or getattr(md, 'raw_markdown', '')
            except Exception:
                content = str(result.markdown) if result.markdown else ""
        elif hasattr(result, 'cleaned_html') and result.cleaned_html:
            content = result.cleaned_html

        metadata = {
            'url': url,
            'title': getattr(result, 'title', ''),
            'description': getattr(result, 'description', ''),
            'keywords': getattr(result, 'keywords', []),
            'language': getattr(result, 'language', ''),
            'word_count': len(content.split()) if content else 0,
            'scraped_at': asyncio.get_event_loop().time(),
            'success': getattr(result, 'success', True),
            'status_code': getattr(result, 'status_code', 200)
        }

        links = []
        if hasattr(result, 'links') and result.links:
            for link in result.links:
                if isinstance(link, dict):
                    links.append(link)
                else:
                    absolute_url = urljoin(url, str(link))
                    links.append({
                        'url': absolute_url,
                        'text': '',
                        'internal': self._is_internal_link(url, absolute_url)
                    })

        return {
            'content': content,
            'metadata': metadata,
            'links': links,
            'url': url
        }

    async def _discover_urls(self, base_url: str) -> list[str]:
        """Discover URLs using sitemap.xml and robots.txt."""
        discovered: list[str] = []
        origin = f"{urlparse(base_url).scheme}://{urlparse(base_url).netloc}"

        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            # try common sitemap paths
            for sm_path in ("/sitemap.xml", "/wp-sitemap.xml"):
                sitemap_url = urljoin(origin, sm_path)
                try:
                    discovered += await self._fetch_sitemap_urls(client, origin, sitemap_url, depth=0)
                except Exception:
                    continue

            # try robots.txt for sitemap
            robots_url = urljoin(origin, "/robots.txt")
            try:
                resp = await client.get(robots_url)
                if resp.status_code == 200:
                    for line in resp.text.splitlines():
                        if line.lower().startswith("sitemap:"):
                            sm = line.split(":", 1)[1].strip()
                            try:
                                discovered += await self._fetch_sitemap_urls(client, origin, sm, depth=0)
                            except Exception:
                                continue
            except Exception:
                pass

        return list(dict.fromkeys(discovered))  # deduplicate

    async def _fetch_sitemap_urls(
        self,
        client: httpx.AsyncClient,
        origin: str,
        sitemap_url: str,
        depth: int = 0,
        max_depth: int = 2
    ) -> list[str]:
        """Fetch URLs from a sitemap or sitemap index recursively."""
        urls: list[str] = []
        if depth > max_depth:
            return urls

        resp = await client.get(sitemap_url)
        if resp.status_code != 200:
            return urls

        text_min = re.sub(r"\s+", " ", resp.text)

        if "<sitemapindex" in text_min:
            locs = re.findall(r"<loc>(.*?)</loc>", text_min)
            for loc in locs:
                child = urljoin(origin, loc.strip())
                try:
                    urls += await self._fetch_sitemap_urls(client, origin, child, depth + 1, max_depth)
                except Exception:
                    continue
        elif "<urlset" in text_min or "<url>" in text_min:
            locs = re.findall(r"<loc>(.*?)</loc>", text_min)
            for loc in locs:
                try:
                    urls.append(urljoin(origin, loc.strip()))
                except Exception:
                    continue
        return urls

    def _is_internal_link(self, base_url: str, link_url: str) -> bool:
        """Check if a link is internal to the base domain."""
        try:
            def _norm(n: str) -> str:
                return n.lower().lstrip().rstrip().removeprefix("www.")
            base_domain = _norm(urlparse(base_url).netloc)
            link_domain = _norm(urlparse(link_url).netloc)
            # If link has no netloc (relative), treat as internal
            if not link_domain:
                return True
            return base_domain == link_domain
        except Exception:
            return False

    async def scrape_multiple_sites(self, urls: list[str]) -> list[dict[str, Any]]:
        """Scrape multiple websites sequentially (to avoid overwhelming servers)."""
        all_results = []
        for url in urls:
            try:
                site_results = await self.scrape_website(url)
                all_results.extend(site_results)
            except ScrapingError as e:
                logger.error(f"Failed to scrape {url}: {e}")
                continue
        return all_results
