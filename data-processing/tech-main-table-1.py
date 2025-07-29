import pandas as pd
import numpy as np
from pathlib import Path
import re

input_file = "filtered_technologies.xlsx"
# output_file = "emerging_technologies_expanded.csv"

# Canonical mapping for OECD research areas (numbers stripped)
_OECD_CANONICAL_MAP = {
    # Natural Sciences
    "natural sciences": "Natural Sciences",
    "mathematics": "Mathematics",
    "computer and information sciences": "Computer and information sciences",
    "physical sciences": "Physical sciences",
    "chemical sciences": "Chemical sciences",
    "earth and related environmental sciences": "Earth and related environmental sciences",
    "biological sciences": "Biological sciences",
    "other natural sciences": "Other natural sciences",
    # Engineering & Technology
    "engineering and technology": "Engineering and Technology",
    "civil engineering": "Civil engineering",
    "electrical engineering": "Electrical engineering, electronic engineering, information engineering",
    "electronic engineering": "Electrical engineering, electronic engineering, information engineering",
    "information engineering": "Electrical engineering, electronic engineering, information engineering",
    "mechanical engineering": "Mechanical engineering",
    "chemical engineering": "Chemical engineering",
    "materials engineering": "Materials engineering",
    "medical engineering": "Medical engineering",
    "environmental engineering": "Environmental engineering",
    "environmental biotechnology": "Environmental biotechnology",
    "industrial biotechnology": "Industrial biotechnology",
    "nano-technology": "Nano-technology",
    "other engineering and technologies": "Other engineering and technologies",
    # Medical & Health Sciences
    "medical and health sciences": "Medical and Health Sciences",
    "basic medical research": "Basic medical research",
    "clinical medicine": "Clinical medicine",
    "health sciences": "Health sciences",
    "medical biotechnology": "Medical biotechnology",
    "other medical science": "Other medical science",
    # Agricultural & Veterinary Sciences
    "agricultural and veterinary sciences": "Agricultural and Veterinary Sciences",
    "agriculture, forestry, fisheries": "Agriculture, forestry, fisheries",
    "agriculture": "Agriculture, forestry, fisheries",
    "animal and dairy science": "Animal and dairy science",
    "veterinary science": "Veterinary science",
    "other agricultural science": "Other agricultural science",
    # Social Sciences
    "social sciences": "Social Sciences",
    "psychology and cognitive science": "Psychology and cognitive science",
    "economics and business": "Economics and business",
    "educational sciences": "Educational sciences",
    "sociology": "Sociology",
    "law": "Law",
    "political science": "Political science",
    "social and economic geography": "Social and economic geography",
    "media and communication": "Media and communication",
    "other social sciences": "Other social sciences",
    # Humanities & the Arts
    "humanities and the arts": "Humanities and the arts",
    "history and archeology": "History and archeology",
    "languages and literature": "Languages and literature",
    "philosophy, ethics and religion": "Philosophy, ethics and religion",
    "art": "Art",
    "other humanities": "Other Humanities",
}

# Fallback keyword-based rules (pattern fragment -> canonical)
_OECD_KEYWORD_RULES = [
    # Agricultural & Veterinary
    (r"(agric|crop|plant|soil|horticul|agronom|forestr|fish|aquatic)", "Agriculture, forestry, fisheries"),
    (r"(animal|dairy)", "Animal and dairy science"),
    (r"veterinar", "Veterinary science"),
    # Natural sciences
    (r"(math|statistics)", "Mathematics"),
    (r"computer", "Computer and information sciences"),
    (r"(physics|physical)", "Physical sciences"),
    (r"chemic", "Chemical sciences"),
    (r"earth|geolog|geophys|ocean|environ", "Earth and related environmental sciences"),
    (r"(biolog|genom|microbiolog|botany|zoolog)", "Biological sciences"),
    # Engineering & Tech
    (r"civil", "Civil engineering"),
    (r"electrical|electronic", "Electrical engineering, electronic engineering, information engineering"),
    (r"mechanical", "Mechanical engineering"),
    (r"materials", "Materials engineering"),
    (r"nano", "Nano-technology"),
    (r"medical eng", "Medical engineering"),
    (r"industrial bio", "Industrial biotechnology"),
    (r"environmental eng", "Environmental engineering"),
    (r"environmental bio", "Environmental biotechnology"),
    # Medical & Health
    (r"basic medical", "Basic medical research"),
    (r"clinical", "Clinical medicine"),
    (r"health", "Health sciences"),
    (r"medical bio", "Medical biotechnology"),
    # Social sciences
    (r"psycholog", "Psychology and cognitive science"),
    (r"econom", "Economics and business"),
    (r"educat", "Educational sciences"),
    (r"sociolog", "Sociology"),
    (r"law", "Law"),
    (r"political", "Political science"),
    (r"geograph", "Social and economic geography"),
    (r"media|communication", "Media and communication"),
    # Humanities
    (r"history|archaeolog", "History and archeology"),
    (r"language|literature", "Languages and literature"),
    (r"philosophy|ethic|religion", "Philosophy, ethics and religion"),
    (r"art", "Art"),
]

def get_trl_adoption_combinations(row):
    """
    Extract TRL and adoption stage combinations from the original row data.
    
        Parameters:
    row (pandas.Series): Original row data
    
    Returns:
    list: List of tuples (year_range, trl_adoption_value)
    """
    # TRL column mapping - each column represents a TRL stage
    trl_stages = {
        'TRL_1': 'TRL 1-2',
        'TRL_2': 'TRL 1-2', 
        'TRL_3': 'TRL 3-4',
        'TRL_4': 'TRL 3-4',
        'TRL_5': 'TRL 5-6',
        'TRL_6': 'TRL 5-6',
        'TRL_7': 'TRL 7-8',
        'TRL_8': 'TRL 7-8',
        'TRL_9': 'TRL 9'
    }
    
    # Adoption stages mapping
    adoption_stages = {
        'innovators': 'Innovators',
        'early_adopters': 'Early Adopters', 
        'early_majority': 'Early Majority',
        'late_majority': 'Late Majority',
        'laggards': 'Laggards'
    }
    
    # TRL stage hierarchy for determining progression
    trl_hierarchy = ['TRL 1-2', 'TRL 3-4', 'TRL 5-6', 'TRL 7-8', 'TRL 9']
    adoption_hierarchy = ['Innovators', 'Early Adopters', 'Early Majority', 'Late Majority', 'Laggards']
    
    # Collect TRL achievements by time period
    trl_by_time = {}
    for trl_col, trl_stage in trl_stages.items():
        if trl_col in row and pd.notna(row[trl_col]):
            time_period = str(row[trl_col]).strip()
            if time_period not in trl_by_time:
                trl_by_time[time_period] = []
            trl_by_time[time_period].append(trl_stage)
    
    # Collect adoption achievements by time period
    adoption_by_time = {}
    for adoption_col, adoption_stage in adoption_stages.items():
        if adoption_col in row and pd.notna(row[adoption_col]):
            time_period = str(row[adoption_col]).strip()
            if time_period not in adoption_by_time:
                adoption_by_time[time_period] = []
            adoption_by_time[time_period].append(adoption_stage)
    
    # Generate all TRL×Adoption combinations
    combinations = []
    all_time_periods = set(list(trl_by_time.keys()) + list(adoption_by_time.keys()))
    
    # Track highest TRL achieved so far
    highest_trl_achieved = None
    time_order = ['0 years', '1-3 years', '3-5 years', '5-7 years', '7-10 years', '10-15 years', '15+ years']
    
    for time_period in time_order:
        if time_period not in all_time_periods:
            continue
            
        current_trl = None
        current_adoptions = []
        
        # Determine TRL for this period
        if time_period in trl_by_time:
            trl_options = trl_by_time[time_period]
            current_trl = max(trl_options, key=lambda x: trl_hierarchy.index(x))
            highest_trl_achieved = current_trl
        
        # Determine adoptions for this period
        if time_period in adoption_by_time:
            current_adoptions = adoption_by_time[time_period]
        
        # Generate combinations
        if current_trl and current_adoptions:
            # Both TRL and adoption happen in this period
            for adoption in current_adoptions:
                # Filter out unwanted combinations
                trl_adoption = f"{current_trl} {adoption}"
                if trl_adoption not in ['TRL 9 Late Majority', 'TRL 9 Laggards']:
                    combinations.append((time_period, trl_adoption))
        elif current_trl and not current_adoptions:
            # Only TRL happens in this period
            combinations.append((time_period, current_trl))
        elif current_adoptions and highest_trl_achieved:
            # Only adoption happens, use highest TRL achieved so far
            for adoption in current_adoptions:
                trl_adoption = f"{highest_trl_achieved} {adoption}"
                if trl_adoption not in ['TRL 9 Late Majority', 'TRL 9 Laggards']:
                    combinations.append((time_period, trl_adoption))
        elif current_adoptions and not highest_trl_achieved:
            # Adoption but no prior TRL - assume appropriate TRL
            for adoption in current_adoptions:
                assumed_trl = "TRL 7-8" if adoption == "Innovators" else "TRL 9"
                trl_adoption = f"{assumed_trl} {adoption}"
                if trl_adoption not in ['TRL 9 Late Majority', 'TRL 9 Laggards']:
                    combinations.append((time_period, trl_adoption))
    
    return combinations

# -------------------------------------------------------------
# Helper to standardise OECD Research Area labels (remove numbers)
# -------------------------------------------------------------

def clean_oecd_category(category: str) -> str:
    """Return OECD Research Area label without leading numeric codes.

    Examples
    ---------
    >>> clean_oecd_category('1.02 Computer and information sciences')
    'Computer and information sciences'
    >>> clean_oecd_category('3 Medical and Health Sciences')
    'Medical and Health Sciences'
    """
    if not isinstance(category, str):
        return category

    # Remove leading digits (and sub-section digits) plus whitespace
    cleaned = re.sub(r"^\s*\d+(\.\d+)*\s+", "", category).strip()
    return cleaned

def canonical_oecd_category(category: str) -> str:
    """Map various cleaned OECD research area strings to a canonical label."""
    if not isinstance(category, str):
        return category

    cleaned = clean_oecd_category(category)
    key = cleaned.lower()
    for pattern, canonical in _OECD_CANONICAL_MAP.items():
        if key.startswith(pattern):
            return canonical
    # keyword fallback
    for regex, canonical in _OECD_KEYWORD_RULES:
        if re.search(regex, key):
            return canonical
    # contains check – if cleaned text contains an entire canonical label token sequence
    for canonical in set(_OECD_CANONICAL_MAP.values()):
        if canonical.lower() in key:
            return canonical
    # final fallback
    return "Other (unmapped)"

def process_technologies_data(input_file, output_file=None):
    """
    Process the technologies data to create the specified output format.
    
    Parameters:
    input_file (str): Path to the input Excel file
    output_file (str, optional): Path to the output CSV file
    
    Returns:
    pandas.DataFrame: The processed dataframe
    """
    
    # Read the Excel file
    print(f"📖 Reading Excel file: {input_file}")
    try:
        df = pd.read_excel(input_file)
        print(f"✅ Successfully loaded {len(df)} rows")
    except Exception as e:
        print(f"❌ Error reading Excel file: {e}")
        return None
    
    # Check required columns
    required_columns = ['OECD_Research_Area', 'technology_name', 'Impact_score', 'Tech_Level']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"❌ Error: Missing required columns: {missing_columns}")
        print(f"Available columns: {list(df.columns)}")
        return None
    
    # No need for description column anymore
    print("ℹ️  Technology descriptions will not be included in output.")
    
    print(f"\n📊 Original data:")
    print(f"   - Rows: {len(df)}")
    print(f"   - Columns: {len(df.columns)}")
    
    # Create expanded dataset
    expanded_rows = []
    
    print("\n🔄 Processing rows...")
    for idx, row in df.iterrows():
        # Split OECD Research Areas and standardise labels (remove numeric prefixes)
        if pd.notna(row['OECD_Research_Area']):
            raw_categories = [cat.strip() for cat in str(row['OECD_Research_Area']).split(';')]
            categories = list({canonical_oecd_category(cat) for cat in raw_categories if canonical_oecd_category(cat)})
        else:
            categories = ['Unknown']
        
        # Get TRL×Adoption combinations for this technology
        trl_adoption_combinations = get_trl_adoption_combinations(row)
        
        # If no combinations found (or all were "0 years"), create a default entry
        if not trl_adoption_combinations or all(combo[0] == "0 years" for combo in trl_adoption_combinations):
            trl_adoption_combinations = [('1-3 years', 'TRL 1-2')]  # Default to earliest non-zero period
        
        # Create rows for each category × TRL×Adoption combination
        for category in categories:
            for year_range, trl_adoption in trl_adoption_combinations:
                # Skip "0 years" entries
                if year_range == "0 years":
                    continue
                    
                new_row = {
                    'OECD_Research_Area': category,
                    'Year_Range': year_range,
                    'Tech_Level': row['Tech_Level'] if 'Tech_Level' in row and pd.notna(row['Tech_Level']) else '',
                    'TRLxAdoption': trl_adoption,
                    'Impact': row['Impact_score'] if pd.notna(row['Impact_score']) else 0,
                    'Tech_Count': 1,  # Will be calculated later
                    'Technology_Name': row['technology_name']
                }
                expanded_rows.append(new_row)
        
        # Progress indicator
        if (idx + 1) % 50 == 0:
            print(f"   Processed {idx + 1}/{len(df)} rows...")
    
    # Create expanded dataframe
    expanded_df = pd.DataFrame(expanded_rows)
    
    print(f"\n📈 Expansion results:")
    print(f"   - Original rows: {len(df)}")
    print(f"   - Expanded rows: {len(expanded_df)}")
    print(f"   - Expansion factor: {len(expanded_df)/len(df):.2f}x")
    
    # Calculate aggregated metrics
    print(f"\n🔢 Calculating aggregated metrics...")
    
    # Group by the key dimensions and aggregate technology names into lists
    grouping_cols = ['OECD_Research_Area', 'Year_Range', 'Tech_Level', 'TRLxAdoption']
    
    # Calculate Impact (average), Tech_Count, and Technology_Name list for each group
    grouped = expanded_df.groupby(grouping_cols).agg({
        'Impact': 'mean',  # Average impact score
        'Technology_Name': lambda x: list(x.unique())  # List of unique technology names
    }).reset_index()
    
    # Add Tech_Count based on the length of technology name lists
    grouped['Tech_Count'] = grouped['Technology_Name'].apply(len)
    
    # Convert technology name lists to semicolon-separated strings
    grouped['Technology_Name'] = grouped['Technology_Name'].apply(lambda x: '; '.join(sorted(x)))
    
    # This is our final dataframe - no need for individual technology rows
    final_df = grouped.copy()
    
    # Sort by category, year range, and TRL×Adoption (excluding "0 years")
    year_range_order = ['1-3 years', '3-5 years', '5-7 years', '7-10 years', '10-15 years', '15+ years', 'Unknown']
    trl_adoption_order = ['TRL 1-2', 'TRL 3-4', 'TRL 5-6', 'TRL 5-6 Innovators', 'TRL 7-8', 
                         'TRL 7-8 Innovators', 'TRL 9 Innovators', 'TRL 7-8 Early Adopters', 
                         'TRL 7-8 Early Majority']
    
    final_df['Year_Range_Sort'] = final_df['Year_Range'].map({yr: i for i, yr in enumerate(year_range_order)})
    final_df['TRLxAdoption_Sort'] = final_df['TRLxAdoption'].map({trl: i for i, trl in enumerate(trl_adoption_order)})
    final_df = final_df.sort_values(['OECD_Research_Area', 'Year_Range_Sort', 'TRLxAdoption_Sort'])
    
    # Drop sorting columns and round Impact values
    final_df = final_df.drop(['Year_Range_Sort', 'TRLxAdoption_Sort'], axis=1)
    final_df['Impact'] = final_df['Impact'].round(3)
    
    # Display final results
    print(f"\n📊 Final dataset:")
    print(f"   - Rows: {len(final_df)} (one per unique combination)")
    print(f"   - Unique research areas: {final_df['OECD_Research_Area'].nunique()}")
    print(f"   - Unique year ranges: {final_df['Year_Range'].nunique()}")
    print(f"   - Unique TRL×Adoption combinations: {final_df['TRLxAdoption'].nunique()}")
    print(f"   - Total unique technologies: {sum(final_df['Tech_Count'])}")
    
    # Show distribution of TRL×Adoption values
    print(f"\n📈 TRL×Adoption distribution:")
    trl_adoption_counts = final_df['TRLxAdoption'].value_counts()
    for trl_adoption, count in trl_adoption_counts.head(10).items():
        print(f"   - {trl_adoption}: {count} entries")
    
    # Show year range distribution
    print(f"\n📅 Year Range distribution:")
    year_range_counts = final_df['Year_Range'].value_counts()
    for year_range, count in year_range_counts.items():
        print(f"   - {year_range}: {count} entries")
    
    # Show impact statistics
    print(f"\n📊 Impact statistics:")
    print(f"   - Mean impact: {final_df['Impact'].mean():.3f}")
    print(f"   - Median impact: {final_df['Impact'].median():.3f}")
    print(f"   - Impact range: {final_df['Impact'].min():.3f} - {final_df['Impact'].max():.3f}")
    
    # Show tech count statistics
    print(f"\n🔢 Tech Count statistics:")
    tech_count_stats = final_df['Tech_Count'].value_counts().sort_index()
    print(f"   - Groups with 1 technology: {tech_count_stats.get(1, 0)}")
    print(f"   - Groups with 2-5 technologies: {tech_count_stats.loc[(tech_count_stats.index >= 2) & (tech_count_stats.index <= 5)].sum()}")
    print(f"   - Groups with 6+ technologies: {tech_count_stats.loc[tech_count_stats.index >= 6].sum()}")
    
    # Show example entries
    print(f"\n🔍 Example entries:")
    sample_df = final_df.head(3)
    for idx, row in sample_df.iterrows():
        print(f"   Research Area: {row['OECD_Research_Area']}")
        print(f"   Year Range: {row['Year_Range']}")
        print(f"   TRL×Adoption: {row['TRLxAdoption']}")
        print(f"   Impact: {row['Impact']}")
        print(f"   Tech Count: {row['Tech_Count']}")
        print(f"   Technologies: {row['Technology_Name'][:100]}{'...' if len(row['Technology_Name']) > 100 else ''}")
        print()
    
    # Save to file
    if output_file is None:
        output_file = input_file.replace('.xlsx', '_processed.csv').replace('.xls', '_processed.csv')
    
    print(f"\n💾 Saving processed dataset to: {output_file}")
    try:
        final_df.to_csv(output_file, index=False)
        print("✅ File saved successfully!")
    except Exception as e:
        print(f"❌ Error saving file: {e}")
        return final_df
    
    print(f"\n💡 Notes:")
    print(f"   - Tech_Level values are populated from the input file")
    print(f"   - Impact values are averaged across technologies in the same group")
    print(f"   - Tech_Count shows how many technologies belong to each group")
    print(f"   - Technology_Name contains semicolon-separated list of all technologies in the group")
    print(f"   - Each row represents one unique combination with aggregated data")
    print(f"   - Filtered out 'TRL 9 Late Majority' and 'TRL 9 Laggards' combinations")
    print(f"   - Excluded all '0 years' entries from the dataset")
    
    return final_df

def main():
    """
    Main function to run the script
    """
    print("🚀 Technology Data Processor - New Format")
    print("=" * 60)
    
    # File paths (modify these as needed)
    input_file = "filtered_technologies.xlsx"  # Change this to your file path
    output_file = "emerging_technologies_processed.csv"    # Change this to your desired output path
    
    # Check if input file exists
    if not Path(input_file).exists():
        print(f"❌ Input file not found: {input_file}")
        print("\n💡 Instructions:")
        print("1. Place your Excel file in the same directory as this script")
        print("2. Update the 'input_file' variable with the correct filename")
        print("3. Run the script again")
        return
    
    # Process the file
    result_df = process_technologies_data(input_file, output_file)
    
    if result_df is not None:
        print(f"\n🎉 Processing complete!")
        print(f"📁 Output file: {output_file}")
        print(f"📊 Final dataset: {len(result_df)} rows × {len(result_df.columns)} columns")
        print(f"\n📋 Output format:")
        print(f"   1. OECD_Research_Area")
        print(f"   2. Year_Range") 
        print(f"   3. Tech_Level (from input file)")
        print(f"   4. TRLxAdoption")
        print(f"   5. Impact (average by group)")
        print(f"   6. Tech_Count (count by group)")
        print(f"   7. Technology_Name (semicolon-separated list)")

if __name__ == "__main__":
    # Required packages check
    try:
        import pandas as pd
        import openpyxl  # Required for reading Excel files
    except ImportError as e:
        print("❌ Required packages not installed.")
        print("Please install them with:")
        print("pip install pandas openpyxl")
        exit(1)
    
    main()