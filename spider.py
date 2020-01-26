# -*- coding: utf-8 -*-
import scrapy
from scrapy import Request

class ProjectSpider(scrapy.Spider):
    name = 'project'
    allowed_domains = ['craigslist.org']
    #start_urls = ['https://chicago.craigslist.org/d/general-for-sale/search/foa']    # for general data
    start_urls = ['https://chicago.craigslist.org/d/cars-trucks/search/cta']          #for cars data
    def parse(self, response):
        # titles = response.xpath('//a[@class="result-title hdrlnk"]/text()').extract()
        # print(titles)
        #
        # for i in titles:
        #     yield{'Title': i}
        deals = response.xpath('//p[@class="result-info"]')
        #print(adv)
        for deal in deals:
            #t = deal.xpath('//a[@class="result-title hdrlnk"]/text()').extract_first("")
            title = deal.xpath('a/text()').extract_first()
            #p = deal.xpath('//span[@class="result-price"]/text()').extract_first("")[1:]
            price = deal.xpath('span[@class="result-meta"]/span[@class="result-price"]/text()').extract_first()[1:]
            #yield {'Title': title, 'Price': price}

            lower_rel_url = deal.xpath('a/@href').extract_first()
            lower_url = response.urljoin(lower_rel_url)
            yield Request(lower_url, callback=self.parse_lower,meta={'Title': title, 'Price': price})
        next_rel_url = response.xpath('//a[@class="button next"]/@href').get()
        next_url = response.urljoin(next_rel_url)
        yield Request(next_url, callback=self.parse)

    def parse_lower(self, response):
        text = "".join(line for line in response.xpath \
            ('//*[@id="postingbody"]/text()').getall())
        response.meta['Text'] = text
        yield response.meta
