import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os
from tqdm import tqdm
import time  # <-- Import time module

# === Config ===
INPUT_PARQUET = '../logos.snappy.parquet'  
LOGO_FOLDER = '../Logos_10000'               
RESULTS_CSV = '../extracted_logos.csv'

if not os.path.exists(LOGO_FOLDER):
    os.makedirs(LOGO_FOLDER)

# Function to extract logo URL
def extract_logo_url(html, base_url):
    soup = BeautifulSoup(html, 'html.parser')
    candidates = []

    for img in soup.find_all('img'):
        attrs = (img.get('alt', '') + img.get('src', '') + str(img.get('class', '')) + img.get('id', '')).lower()
        if 'logo' in attrs:
            src = img.get('src')
            if src:
                candidates.append(urljoin(base_url, src))

    if not candidates:
        for link in soup.find_all('link', rel=lambda x: x and 'icon' in x):
            href = link.get('href')
            if href:
                candidates.append(urljoin(base_url, href))

    return candidates[0] if candidates else None

# Function to download image
def download_image(img_url, domain):
    try:
        response = requests.get(img_url, timeout=10)
        if response.status_code == 200 and 'image' in response.headers['Content-Type']:
            ext = os.path.splitext(urlparse(img_url).path)[1] or '.png'
            filename = os.path.join(LOGO_FOLDER, f"{domain.replace('.', '_')}{ext}")
            with open(filename, 'wb') as f:
                f.write(response.content)
            return filename
    except Exception:
        pass
    return None

# Start timing
start_time = time.time()

# Read and limit domains
df = pd.read_parquet(INPUT_PARQUET).head(10000)
if 'domain' not in df.columns:
    raise ValueError("Fisierul Parquet trebuie sa contina o coloana numita 'domain'.")

results = []

# Crawl each domain
for domain in tqdm(df['domain']):
    try:
        url = domain.strip()
        if not url.startswith('http'):
            url = 'http://' + url

        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            logo_url = extract_logo_url(response.text, url)
            local_file = download_image(logo_url, urlparse(url).netloc) if logo_url else None
            results.append({'domain': domain, 'logo_url': logo_url, 'local_file': local_file})
        else:
            results.append({'domain': domain, 'logo_url': None, 'local_file': None})
    except Exception:
        results.append({'domain': domain, 'logo_url': None, 'local_file': None})

# Save results
pd.DataFrame(results).to_csv(RESULTS_CSV, index=False)

# End timing
end_time = time.time()
elapsed = end_time - start_time
print(f"Rezultatele au fost salvate in {RESULTS_CSV}")
print(f"Time elapsed: {elapsed:.2f} seconds")
