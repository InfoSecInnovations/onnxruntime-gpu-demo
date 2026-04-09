from tika import parser
from bs4 import BeautifulSoup

print("convert PDF")
parsed = parser.from_file('ingest_test.pdf', xmlContent=True)
print(parsed["metadata"])
print(parsed["content"])

print("convert docx")
parsed = parser.from_file('ingest_test.docx', xmlContent=True)
print(parsed["metadata"])
print(parsed["content"])

print("convert image")
parsed = parser.from_file('ingest_test_page-0001.jpg', xmlContent=True)
print(parsed["metadata"])
print(parsed["content"])


def page_split(raw_content):
    content = raw_content["content"]
    soup = BeautifulSoup(content, "html.parser")
    pages = [page.text for page in soup.find_all('div', {"class": "page"})]
    if len(pages):
      return pages
    return [soup.get_text()]

print("split pages PDF")
parsed = parser.from_file('ingest_test.pdf', xmlContent=True)
pages = page_split(parsed)
print(pages)

print("split pages DOCX (doesn't actually split)")
parsed = parser.from_file('ingest_test.docx', xmlContent=True)
pages = page_split(parsed)
print(pages)
