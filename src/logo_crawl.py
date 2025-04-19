import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os
from tqdm import tqdm

# === Config ===
INPUT_PARQUET = '../logos.snappy.parquet'  # Numele .parquet
LOGO_FOLDER = '../Logos'               # Folder unde salvezi logo-urile
RESULTS_CSV = '../extracted_logos.csv'

# Functie pentru extragerea URL-ului logo-ului din HTML
def extract_logo_url(html, base_url):
    soup = BeautifulSoup(html, 'html.parser')
    candidates = []

    # Cauta imagini cu "logo" in src/alt/class/id
    for img in soup.find_all('img'):
        attrs = (img.get('alt', '') + img.get('src', '') + str(img.get('class', '')) + img.get('id', '')).lower()
        if 'logo' in attrs:
            src = img.get('src')
            if src:
                candidates.append(urljoin(base_url, src))

    # Fallback: favicon sau alte link rel="icon" 
    if not candidates:
        for link in soup.find_all('link', rel=lambda x: x and 'icon' in x):
            href = link.get('href')
            if href:
                candidates.append(urljoin(base_url, href))

    return candidates[0] if candidates else None

# Functie pentru descarcarea logo-ului local
def download_image(img_url, domain):
    try:
        response = requests.get(img_url, timeout=10)
        if response.status_code == 200 and 'image' in response.headers['Content-Type']:
            ext = os.path.splitext(urlparse(img_url).path)[1] or '.png'
            filename = os.path.join(LOGO_FOLDER, f"{domain.replace('.', '_')}{ext}")
            with open(filename, 'wb') as f:
                f.write(response.content)
            return filename
    except Exception as e:
        pass
    return None

# Citeste domeniile din fisierul .parquet
df = pd.read_parquet(INPUT_PARQUET)
df = pd.read_parquet(INPUT_PARQUET).head(1000) #limita
if 'domain' not in df.columns:
    raise ValueError("Fisierul Parquet trebuie sa contina o coloana numita 'domain'.")

results = []

# Itereaza peste toate domeniile
for domain in tqdm(df['domain']):
    try:
        url = domain.strip()
        if not url.startswith('http'):
            url = 'http://' + url  # fallback

        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            logo_url = extract_logo_url(response.text, url)
            if logo_url:
                local_file = download_image(logo_url, urlparse(url).netloc)
                results.append({
                    'domain': domain,
                    'logo_url': logo_url,
                    'local_file': local_file
                })
            else:
                results.append({'domain': domain, 'logo_url': None, 'local_file': None})
        else:
            results.append({'domain': domain, 'logo_url': None, 'local_file': None})
    except Exception as e:
        results.append({'domain': domain, 'logo_url': None, 'local_file': None})

# Salveaza rezultatele in CSV
pd.DataFrame(results).to_csv(RESULTS_CSV, index=False)
print(f"Rezultatele au fost salvate in {RESULTS_CSV}")
