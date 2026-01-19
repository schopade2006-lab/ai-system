import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_aadhaar_data():
    # Setup Parameters
    states = ['Maharashtra', 'Uttar Pradesh', 'Karnataka', 'Tamil Nadu', 'West Bengal', 
              'Gujarat', 'Rajasthan', 'Bihar', 'Madhya Pradesh', 'Kerala']
    
    # Generate dates from 2018 to 2025
    start_date = datetime(2018, 1, 1)
    date_list = [start_date + timedelta(days=x) for x in range(0, 2900, 30)] # Monthly data points

    enrolment_rows = []
    biometric_rows = []

    for dt in date_list:
        for state in states:
            # --- Enrolment Data ---
            # Simulate a declining trend as saturation is reached
            base_val = 50000 if dt.year < 2021 else 20000
            
            enrol_row = {
                'date': dt.strftime('%Y-%m-%d'),
                'state': state,
                'age_0_5': np.random.randint(5000, 15000),
                'age_5_17': np.random.randint(2000, 8000),
                'age_18_greater': np.random.randint(1000, 5000)
            }
            # Inject an anomaly for the Isolation Forest to find (e.g., Year 2022 Spike)
            if dt.year == 2022 and state == 'Maharashtra':
                enrol_row['age_18_greater'] += 100000 
                
            enrolment_rows.append(enrol_row)

            # --- Biometric Data ---
            # Biometric updates usually increase over time
            bio_row = {
                'date': dt.strftime('%Y-%m-%d'),
                'state': state,
                'bio_age_5_17': np.random.randint(3000, 10000),
                'bio_age_17_': np.random.randint(5000, 20000)
            }
            biometric_rows.append(bio_row)

    # Create DataFrames
    df_enrol = pd.DataFrame(enrolment_rows)
    df_bio = pd.DataFrame(biometric_rows)

    # Save to CSV
    df_enrol.to_csv('enrolment_clean.csv', index=False)
    df_bio.to_csv('biometric_clean.csv', index=False)
    
    print("Files 'enrolment_clean.csv' and 'biometric_clean.csv' generated successfully!")

if __name__ == "__main__":
    generate_aadhaar_data()
