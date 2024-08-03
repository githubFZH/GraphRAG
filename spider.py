import scrapy
from scrapy.crawler import CrawlerProcess
from bs4 import BeautifulSoup
import html2text
import os
import json
from urllib.parse import urlparse

class ContentFocusedSpider(scrapy.Spider):
    name = 'content_focused_spider'
    start_urls = ['https://crawl4ai.com/mkdocs/']
    allowed_domains = ['crawl4ai.com']

    def __init__(self, *args, **kwargs):
        super(ContentFocusedSpider, self).__init__(*args, **kwargs)
        self.h = html2text.HTML2Text()
        self.h.ignore_links = True
        self.h.ignore_images = True
        self.h.ignore_emphasis = True
        self.h.body_width = 0
        self.results = []
        
        os.makedirs('.data', exist_ok=True)
        os.makedirs('.data/markdown_files', exist_ok=True)

    def parse(self, response):
        # 使用 BeautifulSoup 提取主要内容
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 移除导航栏、侧边栏、页脚等元素
        for elem in soup(['nav', 'header', 'footer', 'aside']):
            elem.decompose()
        
        # 尝试找到主要内容区域
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
        
        if main_content:
            content = str(main_content)
        else:
            content = str(soup.body)  # 如果找不到明确的主要内容，使用整个 body

        # 转换为 Markdown
        markdown_content = self.h.handle(content)

        # 生成文件名并保存
        parsed_url = urlparse(response.url)
        file_path = parsed_url.path.strip('/').replace('/', '_') or 'index'
        markdown_filename = f'.data/markdown_files/{file_path}.txt'
        
        with open(markdown_filename, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        result = {
            'url': response.url,
            'markdown_file': markdown_filename,
        }
        self.results.append(result)
        
        # 继续爬取其他链接
        for link in response.css('a::attr(href)').getall():
            yield response.follow(link, self.parse)

    def closed(self, reason):
        with open('.data/markdown_results.json', 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        print(f"爬取完成。总共爬取了 {len(self.results)} 个页面")
        print("结果元数据保存在 .data/markdown_results.json")
        print("Markdown 文件保存在 .data/markdown_files/ 目录下")

process = CrawlerProcess(settings={
    'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'ROBOTSTXT_OBEY': True,
    'CONCURRENT_REQUESTS': 1,
    'DOWNLOAD_DELAY': 2,
})

process.crawl(ContentFocusedSpider)
process.start()