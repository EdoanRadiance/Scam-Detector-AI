import pandas as pd
import re



def extract_url_features(url):
    features = {}
    features['length'] = len(url)
    features['num_dots'] = url.count('.')
    features['num_hyphens'] = url.count('-')
    features['contains_ip'] = 1 if re.search(r'\b\d{1,3}(?:\.\d{1,3}){3}\b', url) else 0
    return features


def preprocess_url_dataset(phishing_site_urls):
    df = pd.read_csv(phishing_site_urls)

    features_list = []
    for url in df['URL']:
        features = extract_url_features(url)
        features_list.append(features)

    features_df = pd.DataFrame(features_list)
    df = df.join(features_df)

    df['label'] = df['Label'].apply(lambda x: 1 if x.lower() == 'bad' else 0)

    return df



if __name__ == "__main__":
    csv_path = "data/phishing_site_urls.csv"  # Ensure this matches your file location
    processed_df = preprocess_url_dataset(csv_path)
    print("Processed URL Dataset:")
    print(processed_df.head())

    

    