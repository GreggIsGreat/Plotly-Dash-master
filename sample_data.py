import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_sample_data(n_samples=1000):
    """
    Generate sample data that mimics the structure of the yokyo.log file
    but is much smaller in size.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        DataFrame with sample data
    """
    # Generate timestamps
    start_date = datetime(2021, 7, 23)  # Olympics start date
    end_date = datetime(2021, 8, 8)     # Olympics end date
    
    timestamps = [
        (start_date + timedelta(
            days=random.randint(0, 16),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59)
        )).strftime("%d/%b/%Y:%H:%M:%S")
        for _ in range(n_samples)
    ]
    
    # Generate IP addresses
    ip_addresses = [
        f"{random.randint(1, 255)}.{random.randint(0, 255)}."
        f"{random.randint(0, 255)}.{random.randint(0, 255)}"
        for _ in range(n_samples)
    ]
    
    # HTTP methods
    http_methods = np.random.choice(['GET', 'POST', 'PUT', 'DELETE'], n_samples, p=[0.7, 0.2, 0.05, 0.05])
    
    # Paths
    paths = np.random.choice([
        '/', '/results', '/schedule', '/athletes', '/countries', 
        '/sports', '/medals', '/news', '/highlights', '/about'
    ], n_samples)
    
    # Status codes
    status_codes = np.random.choice([200, 404, 500], n_samples, p=[0.9, 0.08, 0.02])
    
    # HTTP versions
    http_versions = np.random.choice(['HTTP/1.1', 'HTTP/2.0'], n_samples, p=[0.8, 0.2])
    
    # Traffic sources
    traffic_sources = np.random.choice([
        'Direct', 'Google', 'Twitter', 'Facebook', 'Instagram',
        'YouTube', 'Bing', 'Yahoo', 'Email', 'Referral'
    ], n_samples)
    
    # User agents (simplified)
    user_agents = np.random.choice([
        'Windows Chrome', 'Windows Edge', 'Mac Chrome', 'Mac Safari',
        'Windows Firefox', 'Android 11 - Samsung Galaxy S10', 'iOS 14 - iPhone',
        'Android 11 - Samsung Galaxy S20', 'iOS 14 - iPad', 'Samsung Smart TV - Tizen 5.0'
    ], n_samples)
    
    # Countries
    countries = np.random.choice([
        'United States', 'Japan', 'China', 'Russia', 'Great Britain',
        'Australia', 'Germany', 'France', 'Italy', 'Canada',
        'Brazil', 'South Korea', 'Netherlands', 'New Zealand', 'Kenya'
    ], n_samples)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Timestamp': timestamps,
        'IP Address': ip_addresses,
        'HTTP Method': http_methods,
        'Path': paths,
        'Status Code': status_codes,
        'HTTP Version': http_versions,
        'Traffic Source': traffic_sources,
        'User Agent': user_agents,
        'Country': countries
    })
    
    return df

if __name__ == "__main__":
    # Generate sample data
    sample_df = generate_sample_data(1000)
    
    # Save to CSV for testing
    sample_df.to_csv("sample_yokyo.log", sep=" ", index=False, header=False)
    
    print(f"Sample data generated with {len(sample_df)} rows")
    print(f"File size: {sample_df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
