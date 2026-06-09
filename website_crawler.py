"""Website-Crawling fuer die RAG-Wissensbasis.

Dieses Modul kapselt das Crawling ausserhalb des Notebooks. Dadurch bleibt
`rag.ipynb` leichter zu mergen und die Crawling-Logik kann später separat
getestet, erweitert oder ausgetauscht werden.
"""

from __future__ import annotations

import re
import time
from collections import deque
from html.parser import HTMLParser
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.request import Request, urlopen
from urllib.robotparser import RobotFileParser

from langchain_core.documents import Document


class _CleanHTMLTextExtractor(HTMLParser):
    """Extrahiert sichtbaren Text und Links aus HTML."""

    # Diese Bereiche enthalten meistens Navigation, Layout oder Code statt
    # fachlichem Inhalt und wuerden die Embeddings unnoetig verrauschen.
    SKIP_TAGS = {"script", "style", "noscript", "svg", "nav", "footer", "header", "aside"}
    BLOCK_TAGS = {
        "p",
        "br",
        "div",
        "section",
        "article",
        "main",
        "li",
        "tr",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
    }

    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.skip_depth = 0
        self.parts: list[str] = []
        self.links: set[str] = set()
        self.title_parts: list[str] = []
        self.in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_by_name = dict(attrs)

        if tag in self.SKIP_TAGS:
            self.skip_depth += 1
            return

        if tag == "title":
            self.in_title = True

        if tag == "a" and attrs_by_name.get("href"):
            # Relative Links werden absolut, Sprungmarken wie #kontakt werden entfernt.
            absolute_url, _ = urldefrag(urljoin(self.base_url, attrs_by_name["href"]))
            self.links.add(absolute_url)

        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP_TAGS and self.skip_depth:
            self.skip_depth -= 1
            return

        if tag == "title":
            self.in_title = False

        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self.skip_depth:
            return

        text = data.strip()
        if not text:
            return

        if self.in_title:
            self.title_parts.append(text)
        else:
            self.parts.append(text)


class WebsiteCrawler:
    """Crawlt Webseiten und gibt sie als LangChain-Documents zurueck."""

    def __init__(
        self,
        *,
        max_pages: int = 20,
        max_depth: int = 1,
        delay_seconds: float = 0.5,
        timeout: int = 10,
        same_domain_only: bool = True,
        user_agent: str = "mlwr-rag-notebook/1.0",
    ) -> None:
        # Diese Grenzen verhindern versehentlich grosse Crawls und halten das
        # Notebook reproduzierbarer.
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.delay_seconds = delay_seconds
        self.timeout = timeout
        self.same_domain_only = same_domain_only
        self.user_agent = user_agent

    def crawl(self, start_urls: list[str]) -> list[Document]:
        """Crawlt ab den Start-URLs und liefert RAG-faehige Dokumente."""
        normalized_starts = [url for url in (self._normalize_url(url) for url in start_urls) if url]
        allowed_domains = {urlparse(url).netloc.lower() for url in normalized_starts}
        robot_parsers = {
            urlparse(url).netloc.lower(): self._build_robot_parser(url)
            for url in normalized_starts
        }

        queue = deque((url, 0) for url in normalized_starts)
        visited: set[str] = set()
        website_docs: list[Document] = []

        while queue and len(website_docs) < self.max_pages:
            url, depth = queue.popleft()

            if url in visited:
                continue
            visited.add(url)

            if self.same_domain_only and not self._same_domain(url, allowed_domains):
                continue

            domain = urlparse(url).netloc.lower()
            robot_parser = robot_parsers.get(domain)
            if robot_parser and not robot_parser.can_fetch(self.user_agent, url):
                print(f"Uebersprungen wegen robots.txt: {url}")
                continue

            try:
                html = self._fetch_html(url)
            except Exception as exc:
                # Einzelne fehlerhafte Seiten sollen nicht den gesamten Crawl abbrechen.
                print(f"Konnte nicht geladen werden: {url} ({exc})")
                continue

            if not html:
                continue

            extractor = _CleanHTMLTextExtractor(base_url=url)
            extractor.feed(html)
            page_text = self._clean_text(" ".join(extractor.parts))
            page_title = self._clean_text(" ".join(extractor.title_parts))

            if page_text:
                website_docs.append(
                    Document(
                        page_content=page_text,
                        metadata={"source": url, "title": page_title, "source_type": "website"},
                    )
                )
                print(f"Geladen: {url}")

            if depth < self.max_depth:
                self._append_next_links(queue, extractor.links, visited, allowed_domains, depth)

            # Hoefliches Crawling: kurze Pause zwischen Requests.
            time.sleep(self.delay_seconds)

        return website_docs

    def _append_next_links(
        self,
        queue: deque[tuple[str, int]],
        links: set[str],
        visited: set[str],
        allowed_domains: set[str],
        depth: int,
    ) -> None:
        """Fuegt neue, erlaubte Links in die Crawl-Warteschlange ein."""
        for link in sorted(links):
            normalized_link = self._normalize_url(link)
            if not normalized_link or normalized_link in visited:
                continue
            if self.same_domain_only and not self._same_domain(normalized_link, allowed_domains):
                continue
            queue.append((normalized_link, depth + 1))

    def _fetch_html(self, url: str) -> str | None:
        """Laedt nur HTML-Seiten und ignoriert Downloads, Bilder oder PDFs."""
        request = Request(
            url,
            headers={"User-Agent": self.user_agent, "Accept": "text/html,application/xhtml+xml"},
        )

        with urlopen(request, timeout=self.timeout) as response:
            content_type = response.headers.get("Content-Type", "")
            if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
                return None

            charset = response.headers.get_content_charset() or "utf-8"
            return response.read().decode(charset, errors="replace")

    def _build_robot_parser(self, root_url: str) -> RobotFileParser | None:
        """Laedt robots.txt, damit die Regeln der Website beachtet werden."""
        parsed = urlparse(root_url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        robot_parser = RobotFileParser(robots_url)

        try:
            robot_parser.read()
        except Exception:
            # Ist robots.txt nicht erreichbar, crawlen wir vorsichtig weiter.
            return None

        return robot_parser

    @staticmethod
    def _normalize_url(url: str) -> str | None:
        """Vereinheitlicht URLs, damit dieselbe Seite nicht mehrfach gecrawlt wird."""
        normalized_url, _ = urldefrag(url.strip())
        parsed = urlparse(normalized_url)

        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return None

        path = parsed.path or "/"
        return parsed._replace(path=path, fragment="").geturl()

    @staticmethod
    def _same_domain(url: str, allowed_domains: set[str]) -> bool:
        """Prueft, ob eine URL innerhalb der erlaubten Domains liegt."""
        return urlparse(url).netloc.lower() in allowed_domains

    @staticmethod
    def _clean_text(text: str) -> str:
        """Reduziert Layout-Whitespace, damit Embeddings weniger Rauschen enthalten."""
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
