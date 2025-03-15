import pandas as pd
import re
from urllib.parse import urlparse, parse_qs
import whois
from datetime import datetime

def extract_url_features(url):
    features = {}

    # URL-based features
    features['length'] = len(url)
    features['num_dots'] = url.count('.')
    features['num_hyphens'] = url.count('-')
    features['num_slashes'] = url.count('/')
    features['num_subdomains'] = urlparse(url).netloc.count('.')

    # IP address detection
    features['contains_ip'] = 1 if re.search(r'\b\d{1,3}(?:\.\d{1,3}){3}\b', url) else 0

    # HTTPS presence
    features['https_token'] = 1 if url.startswith('https://') else -1

    # '@' symbol detection
    features['contains_at_symbol'] = 1 if '@' in url else -1

    # Double-slash redirecting
    features['double_slash_redirecting'] = 1 if url.count('//') > 1 else -1

    # Prefix-Suffix in domain
    features['prefix_suffix'] = 1 if '-' in urlparse(url).netloc else -1

    # --- Additional Features ---

    # Non-standard port: check if port exists and if it’s non-standard (80 for HTTP, 443 for HTTPS)
    parsed = urlparse(url)
    port = parsed.port
    if port and port not in [80, 443]:
        features['non_standard_port'] = 1
    else:
        features['non_standard_port'] = -1

    # URL shortening detection: check if the domain belongs to known shortening services
    shortening_services = ['bit.ly', 'tinyurl.com', 'goo.gl', 'ow.ly', 'is.gd', 'buff.ly']
    domain = parsed.netloc.lower()
    features['shortening_service'] = 1 if any(service in domain for service in shortening_services) else -1

    # Query string analysis: number of query parameters and query length
    query = parsed.query
    query_params = parse_qs(query)
    features['num_query_params'] = len(query_params)
    features['query_length'] = len(query)

    # Ratio of digits in the URL
    digits = sum(c.isdigit() for c in url)
    features['digit_ratio'] = digits / len(url) if len(url) > 0 else 0

    # Suspicious words: e.g., "login", "verify", "secure", etc.
    suspicious_words = ['login', 'verify', 'secure', 'account', 'update']
    features['suspicious_words'] = 1 if any(word in url.lower() for word in suspicious_words) else -1

    # File extension: check if URL ends with a suspicious extension
    suspicious_extensions = ['.exe', '.zip', '.scr', '.bat']
    features['suspicious_extension'] = 1 if any(url.lower().endswith(ext) for ext in suspicious_extensions) else -1

    # Domain TLD: get the top-level domain and flag if it is unusual (you can customize this list)
    tld = domain.split('.')[-1]
    common_tlds = ['com', 'org', 'net', 'edu']
    features['unusual_tld'] = 1 if tld not in common_tlds else -1

    # WHOIS-based features
    domain_info = get_domain_info(url)
    features['domain_age'] = domain_info.get('age', -1)
    features['domain_registration_length'] = domain_info.get('registration_length', -1)

    return features


def get_domain_info(url):
    try:
        domain = urlparse(url).netloc
        domain_info = whois.whois(domain)

        creation_date = domain_info.creation_date
        expiration_date = domain_info.expiration_date

        if isinstance(creation_date, list):
            creation_date = creation_date[0]
        if isinstance(expiration_date, list):
            expiration_date = expiration_date[0]

        today = datetime.today()

        age = (today - creation_date).days if creation_date else -1
        registration_length = (expiration_date - creation_date).days if creation_date and expiration_date else -1

        return {'age': age, 'registration_length': registration_length}
    except Exception as e:
        print(f"WHOIS lookup failed for {url}: {e}")
        return {'age': -1, 'registration_length': -1}


def preprocess_url_dataset(phishing_site_urls):
    df = pd.read_csv(phishing_site_urls)

    features_list = []
    for url in df['URL']:
        features = extract_url_features(url)
        features_list.append(features)

    features_df = pd.DataFrame(features_list)
    df = df.join(features_df)

    # Assuming 'Label' column contains 'bad' or 'good'
    df['label'] = df['Label'].apply(lambda x: 1 if x.lower() == 'bad' else -1)

    return df


if __name__ == "__main__":
    csv_path = "data/phishing_site_urls.csv"
    processed_df = preprocess_url_dataset(csv_path)
    print("Processed URL Dataset:")
    print(processed_df.head())
