import requests
import zipfile
import io
import pandas as pd
import logging
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StatsSAIngestor:
    """
    Ingests CPI data from Stats SA by downloading the latest publication zip file,
    extracting the relevant Excel file, and validating the content.
    """
    def __init__(self, raw_data_dir: Optional[Path] = None):
        if raw_data_dir:
            self.download_dir = raw_data_dir
        else:
            self.download_dir = Path(__file__).resolve().parent.parent.parent / "data" / "raw"
        
        self.base_url = "https://www.statssa.gov.za/timeseriesdata/Excel/"
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
        # Consistent headers to bypass WAF
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Referer': 'https://www.statssa.gov.za/',
            'Connection': 'keep-alive'
        }

    def _validate_excel(self, file_path: Path) -> bool:
        """Checks if the downloaded file is a valid Excel and has minimal expected content."""
        
        # Just read the first few rows to ensure it's not a corrupted file or HTML error page
        df = pd.read_excel(file_path, nrows=5)
        if df.empty:
            logger.warning("File %s is empty.", file_path)
            return False
        return True

    def download_publication(self, pub_code: str = "P0141", yyyymm: str = "202601",
                              is_latest: bool = False) -> bool:
        standardized_path = self.download_dir / f"CPI_{yyyymm}.xlsx"
        
        # Step 0: Check if file already exists to avoid redundant calls
        if standardized_path.exists():
            logger.info("File for %s already exists at %s. Skipping download.", yyyymm, standardized_path)
            if is_latest:
                self._update_latest_link(standardized_path)
            return True

        file_name = f"{pub_code}%20-%20CPI(COICOP)%20from%20Jan%202008%20({yyyymm}).zip"
        url = f"{self.base_url}{file_name}"
        
        logger.info("Downloading from: %s", url)

        try:
            session = requests.Session()
            response = session.get(url, headers=self.headers, timeout=30)
            
            if response.status_code != 200:
                logger.warning("Failed download for %s. HTTP Status: %s", yyyymm, response.status_code)
                return False

            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                # Find the .xlsx file inside the zip
                excel_files = [f for f in z.namelist() if f.endswith('.xlsx')]
                if not excel_files:
                    logger.error("No Excel file found in zip for %s", yyyymm)
                    return False
                
                # Extract the first excel found
                filename = excel_files[0]
                temp_extracted_path = self.download_dir / filename
                z.extract(filename, path=self.download_dir)
                
                # Standardize naming
                if temp_extracted_path.exists():
                    shutil.move(str(temp_extracted_path), str(standardized_path))
                
            # Step 2: Validate the content
            if self._validate_excel(standardized_path):
                logger.info("Successfully validated and saved: %s", standardized_path)
                if is_latest:
                    self._update_latest_link(standardized_path)
                return True
            else:
                standardized_path.unlink(missing_ok=True)
                return False

        except Exception as e:
            logger.error("An unexpected error occurred during ingestion: %s", e)
            return False

    def _update_latest_link(self, source_path: Path):
        """Standardizes the 'CPI_latest.xlsx' file for downstream cleaning scripts."""
        latest_path = self.download_dir / "CPI_latest.xlsx"
        shutil.copy2(source_path, latest_path) # copy2 preserves metadata
        logger.info("Updated latest pointer to: %s", source_path.name)

    def find_and_ingest_latest(self, lookback_months: int = 6):
        """Iteratively looks for the most recent available file on Stats SA."""
        for month_offset in range(0, lookback_months):
            target_date = datetime.now() - pd.DateOffset(months=month_offset)
            target_yyyymm = target_date.strftime("%Y%m")
            
            logger.info("Checking availability for %s...", target_yyyymm)
            if self.download_publication(yyyymm=target_yyyymm, is_latest=True):
                logger.info("Successfully synchronized data up to %s", target_yyyymm)
                return target_yyyymm
        
        logger.error("Could not find any available data in the last 6 months.")
        return None

if __name__ == "__main__":
    ingestor = StatsSAIngestor()
    ingestor.find_and_ingest_latest()