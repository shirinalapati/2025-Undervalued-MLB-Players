"""
Fetch MLB player salary data for hitters.

This module loads salary data from CSV files (primarily from USA TODAY database)
and merges it with player statistics. The salary data is essential for calculating
value metrics like WAR per $1M and the Salary Efficiency component of UVS.

The module expects salary data in data/processed/salaries_2025.csv with columns:
- Player name (various formats supported)
- Salary in millions

Note: The project uses pre-collected salary data rather than live scraping
to ensure reliability and avoid rate limiting issues.
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import logging
from pathlib import Path
import sys
from typing import Optional, Dict, List
import re

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


def fetch_player_salary_from_spotrac(player_name: str, delay: float = 1.0) -> Optional[float]:
    """
    Fetch a single player's 2025 salary from Spotrac.
    
    Args:
        player_name: Player name (e.g., "Mike Trout" or "Trout, Mike")
        delay: Delay between requests in seconds
        
    Returns:
        2025 salary in dollars, or None if not found
    """
    time.sleep(delay)  # Rate limiting
    
    # Convert name format if needed (handle "Last, First" -> "First Last")
    if ', ' in player_name:
        parts = player_name.split(', ')
        search_name = f"{parts[1]} {parts[0]}"
    else:
        search_name = player_name
    
    # Try direct contract page URL first (faster)
    # Format: https://www.spotrac.com/mlb/[team]/[firstname-lastname]/contract/
    # But we don't know team, so we'll search
    
    # Spotrac search URL
    search_url = f"https://www.spotrac.com/search/?q={search_name.replace(' ', '+')}"
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        response = requests.get(search_url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Look for player contract page link - try multiple patterns
        player_links = []
        
        # Pattern 1: Links with /contract in href
        links = soup.find_all('a', href=re.compile(r'/mlb/.*/contract'))
        player_links.extend(links)
        
        # Pattern 2: Links with player name in text and /mlb/ in href
        if not player_links:
            all_links = soup.find_all('a', href=re.compile(r'/mlb/'))
            for link in all_links:
                link_text = link.get_text(strip=True).lower()
                if search_name.lower().split()[0] in link_text or search_name.lower().split()[-1] in link_text:
                    player_links.append(link)
        
        if not player_links:
            logger.warning(f"Could not find contract page link for {player_name}")
            return None
        
        # Get the first result (most likely match)
        contract_url = player_links[0].get('href')
        if not contract_url.startswith('http'):
            contract_url = f"https://www.spotrac.com{contract_url}"
        
        # If it's not a contract page, try to make it one
        if '/contract' not in contract_url:
            contract_url = contract_url.rstrip('/') + '/contract/'
        
        # Fetch contract page
        time.sleep(delay)
        contract_response = requests.get(contract_url, headers=headers, timeout=15)
        contract_response.raise_for_status()
        
        contract_soup = BeautifulSoup(contract_response.content, 'html.parser')
        
        # Look for 2025 salary in contract table - try multiple methods
        salary = None
        
        # Method 1: Look in contract tables
        salary_tables = contract_soup.find_all('table')
        for table in salary_tables:
            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 2:
                    year_text = cells[0].get_text(strip=True)
                    if '2025' in year_text or '2025' in str(cells[0]):
                        salary_text = cells[-1].get_text(strip=True)
                        # Extract dollar amount (handle $X,XXX,XXX or $X.XXM format)
                        # Try millions first
                        if 'M' in salary_text.upper():
                            match = re.search(r'\$?([\d.]+)\s*M', salary_text, re.IGNORECASE)
                            if match:
                                salary = float(match.group(1)) * 1_000_000
                                break
                        else:
                            # Try regular dollar amount
                            match = re.search(r'\$?([\d,]+)', salary_text.replace(',', ''))
                            if match:
                                salary = int(match.group(1).replace(',', ''))
                                break
                if salary:
                    break
            if salary:
                break
        
        # Method 2: Look for salary in page text
        if not salary:
            page_text = contract_soup.get_text()
            # Look for "2025" followed by salary
            pattern = r'2025[^\$]*\$?([\d,]+)'
            matches = re.findall(pattern, page_text)
            if matches:
                salary = int(matches[0].replace(',', ''))
        
        if salary:
            return int(salary)
        else:
            logger.warning(f"Could not find 2025 salary for {player_name} on {contract_url}")
            return None
        
    except requests.exceptions.RequestException as e:
        logger.warning(f"Network error fetching salary for {player_name}: {e}")
        return None
    except Exception as e:
        logger.warning(f"Error fetching salary for {player_name}: {e}")
        return None


def fetch_salaries_from_spotrac_batch(player_names: List[str], delay: float = 2.0) -> pd.DataFrame:
    """
    Fetch salaries for multiple players from Spotrac.
    
    Args:
        player_names: List of player names
        delay: Delay between requests in seconds
        
    Returns:
        DataFrame with 'name' and 'salary_2025' columns
    """
    results = []
    
    logger.info(f"Fetching salaries for {len(player_names)} players from Spotrac...")
    logger.info("WARNING: This will take a while due to rate limiting!")
    logger.info("Estimated time: ~{:.1f} minutes".format(len(player_names) * delay / 60))
    
    for i, name in enumerate(player_names, 1):
        logger.info(f"[{i}/{len(player_names)}] Fetching salary for {name}...")
        salary = fetch_player_salary_from_spotrac(name, delay=delay)
        results.append({
            'name': name,
            'salary_2025': salary
        })
        
        if i % 10 == 0:
            logger.info(f"Progress: {i}/{len(player_names)} players processed")
    
    df = pd.DataFrame(results)
    logger.info(f"Successfully fetched {df['salary_2025'].notna().sum()} salaries out of {len(df)} players")
    
    return df


def fetch_salaries_from_csv(csv_path: Path) -> pd.DataFrame:
    """
    Load salary data from a manually created CSV file.
    
    CSV should have columns: 'name' (or 'Name'), 'salary_2025' (or 'Salary')
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        DataFrame with salary information
    """
    try:
        df = pd.read_csv(csv_path)
        
        # Normalize column names
        if 'Name' in df.columns and 'name' not in df.columns:
            df['name'] = df['Name']
        if 'Salary' in df.columns and 'salary_2025' not in df.columns:
            df['salary_2025'] = df['Salary']
        
        # Ensure we have required columns
        if 'name' not in df.columns or 'salary_2025' not in df.columns:
            logger.error(f"CSV must have 'name' and 'salary_2025' columns")
            return pd.DataFrame()
        
        logger.info(f"Loaded {len(df)} salary records from {csv_path}")
        return df[['name', 'salary_2025']]
        
    except Exception as e:
        logger.error(f"Error loading salary CSV: {e}")
        return pd.DataFrame()


def get_salary_data(year: int = 2025, source: str = 'csv', csv_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Get salary data from the specified source.
    
    Args:
        year: Season year
        source: 'csv' (manual CSV file) or 'spotrac' (web scraping)
        csv_path: Path to CSV file if source is 'csv'
        
    Returns:
        DataFrame with salary information
    """
    if source == 'csv':
        if csv_path is None:
            csv_path = DATA_PROCESSED / f"salaries_{year}.csv"
        
        if csv_path.exists():
            return fetch_salaries_from_csv(csv_path)
        else:
            logger.warning(f"Salary CSV not found at {csv_path}")
            logger.info("Creating template CSV file...")
            # Create template
            template_df = pd.DataFrame(columns=['name', 'salary_2025'])
            template_df.to_csv(csv_path, index=False)
            logger.info(f"Template created at {csv_path}. Please fill in salary data manually.")
            return pd.DataFrame()
    
    elif source == 'spotrac':
        # This would require a list of player names
        logger.warning("Spotrac scraping requires a list of player names.")
        logger.info("Use fetch_salaries_from_spotrac_batch() with your player list.")
        return pd.DataFrame()
    
    else:
        logger.error(f"Unknown source: {source}. Use 'csv' or 'spotrac'")
        return pd.DataFrame()


def merge_salary_with_player_data(player_df: pd.DataFrame, salary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge salary data with player statistics.
    
    Args:
        player_df: DataFrame with player statistics
        salary_df: DataFrame with salary data (columns: 'name', 'salary_2025')
        
    Returns:
        Merged DataFrame
    """
    # Normalize player names for merging
    player_df = player_df.copy()
    salary_df = salary_df.copy()
    
    # Create merge key from player names
    def normalize_name(name):
        if pd.isna(name):
            return None
        name_str = str(name).strip()
        # Handle "Last, First" format
        if ', ' in name_str:
            parts = name_str.split(', ')
            return f"{parts[1]} {parts[0]}".lower().strip()
        return name_str.lower().strip()
    
    player_df['merge_name'] = player_df.get('name', player_df.get('Name', '')).apply(normalize_name)
    salary_df['merge_name'] = salary_df['name'].apply(normalize_name)
    
    # Merge
    merged = pd.merge(
        player_df,
        salary_df[['merge_name', 'salary_2025']],
        on='merge_name',
        how='left'
    )
    
    # Clean up merge column
    merged = merged.drop(columns=['merge_name'], errors='ignore')
    
    logger.info(f"Merged salary data: {merged['salary_2025'].notna().sum()} players have salary info")
    
    return merged


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch MLB player salary data")
    parser.add_argument('--source', choices=['csv', 'spotrac'], default='csv',
                       help='Data source: csv (manual file) or spotrac (web scraping)')
    parser.add_argument('--csv-path', type=str, help='Path to salary CSV file')
    parser.add_argument('--year', type=int, default=2025, help='Season year')
    
    args = parser.parse_args()
    
    salary_df = get_salary_data(year=args.year, source=args.source, 
                               csv_path=Path(args.csv_path) if args.csv_path else None)
    
    if not salary_df.empty:
        print(f"\n✅ Loaded {len(salary_df)} salary records")
        print(salary_df.head())
    else:
        print("\n⚠️ No salary data loaded. Check the source or create a CSV file.")

