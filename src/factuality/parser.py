from urllib.parse import urlparse

from courlan import extract_domain, get_base_url


def load_subdomains(file_path):
    with open(file_path, "r") as f:
        return set(line.strip().lower() for line in f if line.strip())


def normalize_url_for_framing_and_manipulation(url):
    subdomains = load_subdomains("subdomains-10000.txt")
    parsed_url = urlparse(url)
    domain = parsed_url.netloc.lower()
    parts = domain.split(".")
    while len(parts) > 2 and parts[0] in subdomains:
        parts.pop(0)
        main_domain = parts[0]

    return main_domain


def normalize_source_url_for_corpus(url):
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


if __name__ == "__main__":
    url = "https://www.amnesty.org/en"
    main_domain = normalize_url_for_framing_and_manipulation(url)
    print(main_domain)
    base_url = get_base_url(url)
    print(base_url)
    domain = extract_domain(url)
    print(domain)
