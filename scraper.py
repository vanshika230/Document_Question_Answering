import requests
from bs4 import BeautifulSoup
from weasyprint import HTML

base_url = "https://sunbird.org"
page_urls = ["explore/sunbird-usecases", "explore/sunbird-ed-documentation", "explore/articles", "building-blocks", "about-us", "community/events"]  # Replace with the actual URLs of different pages


for index, page_url in enumerate(page_urls):
    full_url = f"{base_url}/{page_url}"
    response = requests.get(full_url)
    soup = BeautifulSoup(response.content, "html.parser")
    page_text = soup.prettify()  

    pdf_filename = f"page_{index+1}.pdf"
    HTML(string=page_text).write_pdf(pdf_filename)

    print(f"PDF created for {full_url} as {pdf_filename}")
