import requests
from bs4 import BeautifulSoup
from langchain.schema import Document

def load_webpage(url):
    # Fetch the page
    response = requests.get(url)
    response.raise_for_status()

    # Parse the HTML
    soup = BeautifulSoup(response.text, "html.parser")

    # Remove scripts, styles, etc.
    for script in soup(["script", "style"]):
        script.decompose()

    # Extract text
    text = soup.get_text(separator="\n")
    text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])

    # Return as a LangChain Document
    return [Document(page_content=text, metadata={"source": url})]

# Ask for URL 
user_input = input("Enter the URL: ")

print(load_webpage(user_input))