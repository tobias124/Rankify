import re
from bs4 import BeautifulSoup, NavigableString, Tag
from typing import List, Set
from urllib.parse import unquote

class WikipediaTextCleaner:
    """
    Advanced Wikipedia content cleaner that extracts only meaningful article text.
    Removes all navigation, templates, infoboxes, and metadata.
    """
    
    def __init__(self):
        # Patterns for content that should be completely removed
        self.remove_patterns = [
            # Navigation and UI
            r'^Contents?$',
            r'^Navigation menu$',
            r'^Personal tools$',
            r'^Namespaces$',
            r'^Views$',
            r'^More$',
            r'^Tools$',
            r'^Search$',
            r'^Appearance$',
            r'^\d+\s+languages?$',
            r'^Add topic$',
            r'^Toggle.*contents?$',
            
            # Wikipedia specific
            r'^From Wikipedia.*',
            r'^Jump to navigation',
            r'^This article.*',
            r'^Coordinates:.*',
            r'^\[show\]|\[hide\]|\[edit\]',
            r'^\[v\].*\[t\].*\[e\]',  # Template navigation
            
            # References and metadata
            r'^\d+\s+(References?|External links?|See also|Further reading|Sources?|Bibliography|Notes?)$',
            r'^Categories?:',
            r'^Hidden categories?:',
            r'^Authority control',
            r'^ISBN|OCLC|LCCN',
            
            # Templates and infoboxes
            r'^Films?(\s+directed\s+by|\s*\|)',
            r'^Soundtracks?\s*\|',
            r'^Video games?\s*\|',
            r'^Other media\s*\|',
            r'^Related\s*\|',
            r'^External links?$',
            
            # Navigation templates
            r'.*directed by.*---',
            r'.*\s+---\s+.*',
            
            # Maintenance
            r'^This section.*citation',
            r'^Please help.*',
            r'.*may be challenged.*',
            
            # Common navigation phrases
            r'move to sidebar hide',
            r'show.*hide.*',
            
            # Template markers
            r'^\*\s*[vetc]:\s*(View|Edit|Talk).*template',
        ]
        
        # Compiled regex patterns
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.remove_patterns]
        
        # Tags to completely remove
        self.remove_tags = {
            'script', 'style', 'nav', 'header', 'footer', 'aside',
            'table', 'ul', 'ol', 'dl'  # Remove all lists and tables
        }
        
        # CSS classes/IDs that indicate non-content
        self.remove_selectors = [
            # Navigation
            '[class*="nav"]', '[id*="nav"]',
            '[class*="menu"]', '[id*="menu"]',
            '[class*="sidebar"]', '[id*="sidebar"]',
            '[class*="toc"]', '[id*="toc"]',
            
            # Wikipedia specific
            '.mw-navigation', '.printfooter', '.catlinks',
            '.navbox', '.infobox', '.metadata',
            '.references', '.reflist',
            '.interlanguage-link', '.tools-list',
            '.mw-editsection', '.mw-footer',
            
            # Templates and boxes
            '[class*="template"]', '[class*="box"]',
            '[class*="portal"]', '[class*="sister"]',
            '[class*="authority"]', '[class*="external"]',
        ]

    def remove_unwanted_elements(self, soup: BeautifulSoup) -> None:
        """Remove all unwanted HTML elements."""
        
        # Remove by tag
        for tag_name in self.remove_tags:
            for element in soup.find_all(tag_name):
                element.decompose()
        
        # Remove by CSS selectors
        for selector in self.remove_selectors:
            for element in soup.select(selector):
                element.decompose()
        
        # Remove by class/id patterns
        for element in soup.find_all(attrs={"class": re.compile(r"(nav|footer|header|sidebar|toc|lang|tool|edit|portal|external|template|box|menu)", re.I)}):
            element.decompose()
            
        for element in soup.find_all(attrs={"id": re.compile(r"(nav|footer|header|sidebar|toc|lang|tool|edit|portal|external|template|box|menu)", re.I)}):
            element.decompose()

    def extract_main_content(self, soup: BeautifulSoup) -> BeautifulSoup:
        """Extract only the main article content."""
        
        # Try to find main content container
        main_selectors = [
            'div.mw-parser-output',
            'div#mw-content-text',
            'div#content',
            'div#bodyContent',
        ]
        
        main_content = None
        for selector in main_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                break
        
        if not main_content:
            main_content = soup
        
        return main_content

    def extract_paragraphs_only(self, content: BeautifulSoup) -> List[str]:
        """Extract only paragraph text, ignoring everything else."""
        
        paragraphs = []
        
        # Only look for paragraph tags
        for p_tag in content.find_all('p', recursive=True):
            # Skip if paragraph is inside unwanted containers
            if p_tag.find_parent(['table', 'nav', 'aside', 'footer', 'div'], 
                                 class_=re.compile(r'(nav|infobox|sidebar|template|box)', re.I)):
                continue
            
            # Extract text and clean it
            text = self.clean_paragraph_text(p_tag.get_text())
            
            if self.is_valid_paragraph(text):
                paragraphs.append(text)
        
        return paragraphs

    def clean_paragraph_text(self, text: str) -> str:
        """Clean individual paragraph text."""
        
        # Remove Wikipedia markup
        text = re.sub(r'\[edit\]', '', text)
        text = re.sub(r'\[\d+\]', '', text)  # Citation numbers
        text = re.sub(r'\[.*?\]', '', text)  # All remaining brackets
        
        # Remove URLs
        text = re.sub(r'https?://[^\s]+', '', text)
        text = re.sub(r'www\.[^\s]+', '', text)
        
        # Clean up special characters
        text = re.sub(r'[→←↑↓•]', '', text)
        text = re.sub(r'\|', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text

    def is_valid_paragraph(self, text: str) -> bool:
        """Check if text is a valid article paragraph."""
        
        if not text or len(text) < 50:
            return False
        
        # Check against removal patterns
        for pattern in self.compiled_patterns:
            if pattern.match(text):
                return False
        
        # Skip if too many special characters
        special_chars = sum(1 for c in text if c in '|[](){}*→←↑↓•')
        if special_chars > len(text) * 0.15:
            return False
        
        # Skip if mostly links or URLs
        if text.count('http') > 2 or text.count('www.') > 2:
            return False
        
        # Skip if starts with common navigation patterns
        nav_starts = [
            'Main page', 'Random article', 'About Wikipedia',
            'Create account', 'Log in', 'Donate',
            'Personal tools', 'Namespaces', 'Views',
            'Films directed by', 'Soundtracks',
            'Video games', 'Other media', 'Related',
            'External links', 'References', 'See also'
        ]
        
        for nav_start in nav_starts:
            if text.startswith(nav_start):
                return False
        
        # Must contain at least some regular words
        words = text.split()
        if len(words) < 8:
            return False
        
        # Check for actual sentences (should have some periods or other sentence endings)
        sentence_endings = text.count('.') + text.count('!') + text.count('?')
        if sentence_endings == 0 and len(text) > 100:
            return False
        
        return True

    def clean_wikipedia_html(self, html_content: str) -> List[str]:
        """Main method to clean Wikipedia HTML and return clean paragraphs."""
        
        if not html_content:
            return []
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Step 1: Remove unwanted elements
        self.remove_unwanted_elements(soup)
        
        # Step 2: Extract main content
        main_content = self.extract_main_content(soup)
        
        # Step 3: Extract only paragraphs
        paragraphs = self.extract_paragraphs_only(main_content)
        
        # Step 4: Final filtering
        clean_paragraphs = []
        for para in paragraphs:
            # Additional quality checks
            if self.is_quality_content(para):
                clean_paragraphs.append(para)
        
        return clean_paragraphs

    def is_quality_content(self, text: str) -> bool:
        """Final quality check for content."""
        
        # Must be substantial
        if len(text) < 100:
            return False
        
        # Should not be all caps (likely headers)
        if text.isupper():
            return False
        
        # Should contain common English words
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = set(text.lower().split())
        if len(words.intersection(common_words)) == 0:
            return False
        
        # Should not be mostly punctuation
        alpha_chars = sum(1 for c in text if c.isalpha())
        if alpha_chars < len(text) * 0.6:
            return False
        
        return True

    def extract_title_from_html(self, html_content: str) -> str:
        """Extract clean page title."""
        
        if not html_content:
            return "Unknown"
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Try Wikipedia-specific title elements
        title_elem = soup.find('h1', {'id': 'firstHeading'}) or soup.find('h1', {'class': 'firstHeading'})
        if title_elem:
            title = title_elem.get_text().strip()
            # Clean the title
            title = re.sub(r'\[edit\]', '', title)
            title = re.sub(r'\s+', ' ', title).strip()
            return title
        
        # Fallback to page title
        title_tag = soup.find('title')
        if title_tag:
            title = title_tag.get_text().strip()
            # Remove Wikipedia suffix
            title = re.sub(r'\s*-\s*Wikipedia.*$', '', title)
            return title
        
        return "Unknown"


# Updated processor using the new cleaner
class WikipediaContentProcessor:
    """
    Enhanced Wikipedia content processor using advanced text cleaning.
    """
    
    def __init__(self):
        self.cleaner = WikipediaTextCleaner()
    
    def is_wikipedia_url(self, url: str) -> bool:
        """Check if URL is from Wikipedia"""
        return 'wikipedia.org' in url.lower()
    
    def clean_html_content(self, html_content: str) -> str:
        """Clean HTML and return concatenated clean text."""
        paragraphs = self.cleaner.clean_wikipedia_html(html_content)
        return '\n\n'.join(paragraphs)
    
    def split_into_paragraphs(self, text: str, min_length: int = 100) -> List[str]:
        """Split already cleaned text into paragraphs."""
        if not text:
            return []
        
        paragraphs = text.split('\n\n')
        return [p.strip() for p in paragraphs if len(p.strip()) >= min_length]
    
    def extract_title_from_html(self, html_content: str) -> str:
        """Extract page title."""
        return self.cleaner.extract_title_from_html(html_content)


# Example usage function
def test_cleaner():
    """Test the cleaner with sample Wikipedia HTML."""
    
    # This would be your actual HTML content
    sample_html = """
    <div class="mw-parser-output">
        <p>This is a real paragraph about the topic that contains actual information.</p>
        <div class="navbox">Navigation stuff to remove</div>
        <p>Another paragraph with substantive content about the subject matter.</p>
        <ul class="interlanguage-links">Language links to remove</ul>
    </div>
    """
    
    cleaner = WikipediaTextCleaner()
    clean_paragraphs = cleaner.clean_wikipedia_html(sample_html)
    
    for i, para in enumerate(clean_paragraphs):
        print(f"Paragraph {i+1}: {para}")

if __name__ == "__main__":
    test_cleaner()