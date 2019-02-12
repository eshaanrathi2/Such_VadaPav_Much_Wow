import sys
from icrawler.builtin import GoogleImageCrawler, BingImageCrawler

## uncomment for command line input and replace 'vada pav' in 
## "google_crawler.crawl(keyword='vada pav', max_num=500)" 
## with str(key)
## issue --> gives absurd images sometimes

# key = sys.argv

## replace 'vada pav' below with images for downloading random not_vada_pav images
google_crawler = GoogleImageCrawler(storage={r'root_dir': r'C:/Users/RS_Vulcan/Documents/vada_pav_bing'})
google_crawler.crawl(keyword='vada pav', max_num=500)