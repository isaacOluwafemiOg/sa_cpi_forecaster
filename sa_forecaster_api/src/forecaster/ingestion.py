import requests
import zipfile
import io
import pandas as pd
from pathlib import Path
from datetime import datetime
import shutil


class StatsSAIngestor:
    '''
    Ingests CPI data from Stats SA by downloading the latest publication zip file,
    extracting the relevant Excel file, and saving it to a standardized location for further processing.
    '''
    def __init__(self):
        # Get the absolute path of the directory containing the script file
        self.download_dir = Path(__file__).resolve().parent.parent.parent / "data" / "raw"

        self.base_url = "https://www.statssa.gov.za/timeseriesdata/Excel/"
        self.download_dir.mkdir(parents=True, exist_ok=True)

    def download_publication(self, pub_code="P0141",yyyymm="202601",is_latest=False):
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
                for filename in z.namelist():
                    if filename.endswith('.xlsx'):
                        z.extract(filename, path=self.download_dir)
                        #rename the extracted file to a consistent name for easier loading later
                        extracted_path = self.download_dir / filename
                        standardized_path = self.download_dir / f"CPI_{yyyymm}.xlsx"
                        try:
                            extracted_path.rename(standardized_path,exist_ok=True)
                            print(f"Successfully ingested {standardized_path} to {self.download_dir}")
                        except FileExistsError:
                            print(f"Error: '{destination}' already exists. Replacing instead.")
                            standardized_path.unlink()  # Remove the existing file
                            extracted_path.rename(standardized_path)
                        except OSError as e:
                            # Catch other potential OS errors
                            print(f"OSError: {e}")
                        
                        #if is_latest is True, make a copy of this file and save as "CPI_latest.xlsx" for easier access in the cleaning step
                        if is_latest:
                            latest_path = self.download_dir / "CPI_latest.xlsx"
                            #windowspath object has no attribute copy, use shutil instead
                            shutil.copy(standardized_path, latest_path)
                            print(f"Latest CPI file saved to {latest_path}")

                return True
        else:
            print(f"Failed to download CPI for {yyyymm}. Status: {response.status_code}")
            return False


if __name__ == "__main__":
    ingestor = StatsSAIngestor()
    # get current year and month in yyyymm format
    for month_offset in range(0, 6):  # Try current month and the previous 5 months
        target_yyyymm = (datetime.now() - pd.DateOffset(months=month_offset)).strftime("%Y%m")
        if ingestor.download_publication(pub_code="P0141", yyyymm=target_yyyymm,is_latest=True):
            print(f"Latest available Publication for {target_yyyymm} downloaded successfully.")
            break
    
        