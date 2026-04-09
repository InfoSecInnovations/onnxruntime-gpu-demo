from tika import parser

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