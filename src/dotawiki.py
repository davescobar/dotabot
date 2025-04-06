from bs4 import BeautifulSoup
import requests
import time
import json
from datetime import datetime
import re

class DotaWikiScraper:
    def __init__(self, start_url, base_url, sitemap_base):
        """
        Initializes the scraper with the necessary URLs.

        Args:
            start_url (str): The starting URL for scraping.
            base_url (str): The base URL of the website.
            sitemap_base (str): The base URL for the sitemap.
        """
        self.start_url = start_url
        self.base_url = base_url
        self.sitemap_base = sitemap_base
        self.all_urls = []

    @staticmethod
    def html_to_text(html_content):
        """
        Converts HTML content to plain text and cleans it up.

        Args:
            html_content (str): The HTML content as a string.

        Returns:
            str: The cleaned plain text extracted from the HTML.
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        text = soup.get_text()
        
        # Remove extra white spaces, newlines, tabs, and normalize spaces
        return re.sub(r'\s+', ' ', text).strip()

    def scrape_urls(self):
        """
        Scrapes all URLs from the sitemap starting from the initial URL.
        """
        self.first_scraped_time = datetime.now().isoformat()  # Record start time
        current_url = self.start_url
        previous_namefrom = None

        while current_url:
            print(f"Requesting: {current_url}")
            res = requests.get(current_url)
            soup = BeautifulSoup(res.text, "html.parser")

            chunk = soup.find("ul", class_="mw-allpages-chunk")
            if not chunk:
                break

            links = chunk.find_all("a")
            if not links:
                break

            page_urls = [self.base_url + a["href"] for a in links]
            self.all_urls.extend(page_urls)

            # Get last item's title to build next page URL
            last_title = links[-1]["href"].split("/wiki/")[-1]
            namefrom = last_title.replace("_", "+")

            # Stop if we've already seen this namefrom (end of list)
            if namefrom == previous_namefrom:
                print("Reached last page.")
                break

            previous_namefrom = namefrom
            current_url = self.sitemap_base + namefrom
            time.sleep(1)

        self.last_scraped_time = datetime.now().isoformat()  # Record end time
        print(f"Collected {len(self.all_urls)} URLs.")

    def group_urls(self):
        """
        Groups URLs by their base page.

        Returns:
            dict: A dictionary where keys are base pages and values are lists of URLs.
        """
        grouped_urls = {}
        for url in self.all_urls:
            base_page = url.split("/wiki/")[1].split("/")[0]
            if base_page not in grouped_urls:
                grouped_urls[base_page] = []
            grouped_urls[base_page].append(url)
        return grouped_urls

    def save_to_json(self, filename):
        """
        Saves the grouped URLs to a JSON file, including metadata.

        Args:
            filename (str): The name of the JSON file.
        """
        grouped_urls = self.group_urls()
        metadata = {
            "first_scraped": self.first_scraped_time,
            "last_scraped": self.last_scraped_time,
            "grouped_urls": grouped_urls
        }
        with open(filename, "w") as file:
            json.dump(metadata, file, indent=4)
        print(f"URLs saved to {filename}.")

    def parse_url(self, url):
        """
        Parses the HTML content of a given URL and extracts the relevant content.

        Args:
            url (str): The URL to parse.

        Returns:
            str: The cleaned and extracted content from the HTML.
        """
        print(f"Parsing URL: {url}")
        res = requests.get(url)
        soup = BeautifulSoup(res.text, "html.parser")

        # Define the beginning and ending signals
        start_signal = '<noscript><iframe src="https://www.googletagmanager.com/ns.html?id=GTM-N6XD44P"'
        end_signal = '<!-- Saved in parser cache with key'

        # Extract the HTML content between the signals
        html_content = res.text
        start_index = html_content.find(start_signal)
        end_index = html_content.find(end_signal)

        if start_index == -1 or end_index == -1:
            print(f"Could not find content signals for URL: {url}")
            return None

        # Extract and clean the content
        extracted_html = html_content[start_index:end_index]
        cleaned_text = self.html_to_text(extracted_html)
        return cleaned_text

    def process_urls(self, output_filename):
        """
        Iterates through each URL, parses its content, and updates the JSON with cleaned text.

        Args:
            output_filename (str): The name of the output JSON file to save the results.
        """
        results = {}
        for url in self.all_urls:
            print(f"Processing URL: {url}")
            cleaned_text = self.parse_url(url)
            if cleaned_text:
                results[url] = cleaned_text
            else:
                print(f"Failed to process URL: {url}")

        # Save the results to a JSON file
        with open(output_filename, "w") as file:
            json.dump(results, file, indent=4)
        print(f"Processed data saved to {output_filename}.")

    def scrape_and_process_all(self, output_filename):
        """
        Collects all URLs, scrapes each URL, and saves the cleaned text to a JSON file.

        Args:
            output_filename (str): The name of the output JSON file to save the results.
        """
        # Step 1: Scrape all URLs
        self.scrape_urls()

        # Step 2: Process each URL
        results = {}
        for url in self.all_urls:
            print(f"Processing URL: {url}")
            cleaned_text = self.parse_url(url)
            if cleaned_text:
                results[url] = cleaned_text
            else:
                print(f"Failed to process URL: {url}")
            time.sleep(1)
        # Step 3: Save the results to a JSON file
        with open(output_filename, "w") as file:
            json.dump(results, file, indent=4)
        print(f"Processed data saved to {output_filename}.")

# Example usage
if __name__ == "__main__":
    base_url = "https://dota2.fandom.com"
    sitemap_base = "https://dota2.fandom.com/wiki/Local_Sitemap?namefrom="
    start_url = sitemap_base + "%28monkey%29+Business"

    scraper = DotaWikiScraper(start_url, base_url, sitemap_base)
    scraper.scrape_and_process_all("processed_urls.json")
