import requests
import zipfile
import io
import pandas as pd
from pathlib import Path

class StatsSAIngestor:
    def __init__(self, download_dir="../data/raw"):
        self.base_url = f"https://www.statssa.gov.za/timeseriesdata/Excel/"
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)

    def download_publication(self, pub_code="P0141",yyyymm="202601"):
        """
        Downloads a publication zip file by its code (e.g., 'P0141').
        Note: The URL structure can vary, but this is the common pattern for time series.
        """
        file_name = f"{pub_code}%20-%20CPI(COICOP)%20from%20Jan%202008%20({yyyymm}).zip"
        url = f"{self.base_url}{file_name}"
        
        print(f"Attempting to download {pub_code} from {url}...")

        #Provide full browser headers to evade Incapsula firewall which often blocks 
        # the default Python User-Agent
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.statssa.gov.za/',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }

        session = requests.Session()
        response = session.get(url, headers=headers,stream=True)

        if response.status_code == 200:
            # Extract zip in memory and save files
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                z.extractall(self.download_dir / pub_code)
            print(f"Successfully ingested {pub_code} to {self.download_dir / pub_code}")
            return True
        else:
            print(f"Failed to download {pub_code}. Status: {response.status_code}")
            return False


if __name__ == "__main__":
    ingestor = StatsSAIngestor()
    if ingestor.download_publication(pub_code="P0141", yyyymm="202601"):
        
        print("publication downloaded successfully. You can now load the data into a DataFrame using pandas.")