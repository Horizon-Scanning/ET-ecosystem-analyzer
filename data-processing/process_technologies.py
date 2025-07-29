import pandas as pd
import numpy as np

def process_csv():
    # Read the original CSV file
    print("Reading the original CSV file...")
    df = pd.read_csv('spider_ui/emerging_technologies_processed.csv')
    
    # Display the first few rows to understand the structure
    print("Original CSV structure:")
    print(df.head())
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"Shape: {df.shape}")
    
    # Select the required columns
    required_columns = ['OECD_Research_Area', 'Year_Range', 'Tech_Level', 'TRLxAdoption', 'Technology_Name']
    df_selected = df[required_columns].copy()
    
    # Check if Technology_Name contains multiple values (separated by commas, semicolons, or pipes)
    print("\nSample Technology_Name values:")
    print(df_selected['Technology_Name'].head(10).tolist())
    
    # Create a new dataframe to store the processed data
    processed_rows = []
    
    for index, row in df_selected.iterrows():
        # Split Technology_Name by common separators (comma, semicolon, pipe)
        tech_names = str(row['Technology_Name'])
        
        # Try different separators
        if ',' in tech_names:
            tech_list = [name.strip() for name in tech_names.split(',')]
        elif ';' in tech_names:
            tech_list = [name.strip() for name in tech_names.split(';')]
        elif '|' in tech_names:
            tech_list = [name.strip() for name in tech_names.split('|')]
        else:
            tech_list = [tech_names.strip()]
        
        # Create a new row for each technology
        for tech_name in tech_list:
            if tech_name and tech_name != 'nan':  # Skip empty or NaN values
                new_row = {
                    'OECD_Research_Area': row['OECD_Research_Area'],
                    'Year_Range': row['Year_Range'],
                    'Tech_Level': row['Tech_Level'],
                    'TRLxAdoption': row['TRLxAdoption'],
                    'Technology_Name': tech_name
                }
                processed_rows.append(new_row)
    
    # Create the new dataframe
    new_df = pd.DataFrame(processed_rows)
    
    print(f"\nProcessed data shape: {new_df.shape}")
    print("\nSample of processed data:")
    print(new_df.head(10))
    
    # Save to new CSV file
    output_file = 'spider_ui/emerging_technologies_separated.csv'
    new_df.to_csv(output_file, index=False)
    print(f"\nNew CSV file created: {output_file}")
    
    # Display statistics
    print(f"\nStatistics:")
    print(f"Original rows: {len(df_selected)}")
    print(f"New rows: {len(new_df)}")
    print(f"Unique technologies: {new_df['Technology_Name'].nunique()}")

if __name__ == "__main__":
    process_csv() 