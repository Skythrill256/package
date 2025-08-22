#!/usr/bin/env python3
import asyncio
import os

from package.config.config import WebVectorConfig
from package.core.core import WebVectorClient

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "name"


async def run():
    openai_api_key = os.getenv("OPENAI_API_KEY") or input("Enter OPENAI_API_KEY: ")
    max_pages = int(os.getenv("MAX_PAGES", "10"))
    max_depth = int(os.getenv("MAX_DEPTH", "1"))

    url_input = input("Enter one or more site URLs (comma or space separated): ").strip()

    def parse_urls(s: str) -> list[str]:
        if "," in s:
            parts = [p.strip() for p in s.split(",")]
        else:
            parts = [p.strip() for p in s.split()]  
        return [p for p in parts if p]

    urls = parse_urls(url_input)
    if not urls:
        print("No URLs provided. Exiting.")
        return

    
    config = WebVectorConfig(
        openai_api_key=openai_api_key,
        qdrant_url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
        max_pages=max_pages,
        max_depth=max_depth,
       
    )

    client = WebVectorClient(config)

    
    print("\n[1/2] Scraping, embedding, and storing in vector DB... This may take a bit.\n")
    summary = await client.scrape_and_store(urls, recreate_collection=False)
    print("Done.")
    print(
        f"Processed URLs: {summary.get('urls_processed')} | Pages: {summary.get('pages_scraped')} | "
        f"Embeddings stored: {summary.get('embeddings_stored')} | Collection: {summary.get('collection_name')}"
    )

    
    while True:
        print("\n[2/2] Running query...\n")
        query_input = input("Enter your query (leave empty to exit): ").strip()
        if not query_input:
            print("Exiting.")
            break

        response = await client.query(query_input, top_k=5, similarity_threshold=0.7, include_sources=True)

        print("Answer:\n-------")
        print(response.get("answer", "<no answer>"))

        sources = response.get("sources") or []
        if sources:
            print("\nSources:\n--------")
            for i, s in enumerate(sources, 1):
                title = s.get("title") or s.get("metadata", {}).get("title") or "Untitled"
                url = s.get("url") or s.get("metadata", {}).get("url") or ""
                score = s.get("score")
                score_txt = f" (score: {score:.3f})" if isinstance(score, (int, float)) else ""
                print(f"{i}. {title}{score_txt}\n   {url}")

        again = input("\nWould you like to query again? [y/N]: ").strip().lower()
        if again not in ("y", "yes"):
            print("Goodbye.")
            break


if __name__ == "__main__":
    asyncio.run(run())
