import pandas as pd
import numpy as np

input_file = 'emerging_technologies_evaluation_oecd.xlsx'

def filter_technologies_by_categories(
    input_file: str,
    output_file: str | None = None,
    debug_mode: bool = True,
    target_categories: set[str] | None = None,
):
    """
    Filter technologies based on specified **OECD_Research_Area** categories.
    If no `target_categories` are supplied (default), ALL rows that contain at
    least one OECD category will be kept (i.e. the function behaves as an
    extractor/analysis tool rather than a filter).

    Parameters
    ----------
    input_file : str
        Path to the input Excel file.
    output_file : str | None, optional
        Path to save the filtered results (optional).
    debug_mode : bool, default=True
        If True, prints detailed analysis of categories.
    target_categories : set[str] | None, optional
        Categories to filter by. Provide an empty set or None to keep all.

    Returns
    -------
    pandas.DataFrame
        The (optionally) filtered dataframe, with **one row per OECD category**
        (column `OECD_Category`).
    """
    
    # ------------------------------------------------------------------
    # Handle default: if *no* target categories specified, we take **all** of
    # them (i.e. no filtering by category name). When a non-empty set is
    # provided, we use it to filter.
    # ------------------------------------------------------------------
    if target_categories is None:
        target_categories = set()
    
    # Read the Excel file
    print(f"Reading data from {input_file}...")
    df = pd.read_excel(input_file, sheet_name='Sheet1')
    
    print(f"Total rows in dataset: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    if debug_mode:
        # Analyze all unique categories in the dataset
        print("\n=== DEBUGGING: Analyzing all categories in your dataset ===")
        all_categories = set()
        for _, row in df.iterrows():
            if pd.notna(row['OECD_Research_Area']):
                categories = [cat.strip() for cat in str(row['OECD_Research_Area']).split(';')]
                all_categories.update(categories)
        
        print(f"Total unique categories found: {len(all_categories)}")
        
        # If a list of target categories is supplied, show presence stats
        if target_categories:
            print("\n--- Target categories found in your data ---")
        found_targets = []
        present_targets = set()
        
        # Collect all categories across all rows
        for _, row in df.iterrows():
            if pd.notna(row['OECD_Research_Area']):
                categories = [cat.strip() for cat in str(row['OECD_Research_Area']).split(';')]
                present_targets.update(categories)
        
        if target_categories:
            for target in sorted(target_categories):
                if target in present_targets:
                    found_targets.append(target)
                    print(f"✓ {target}")
        
        if target_categories:
            print(f"\nFound {len(found_targets)} out of {len(target_categories)} target categories in the data")
        
        # Show target categories NOT found
        if target_categories:
            missing_targets = target_categories - present_targets
            if missing_targets:
                print(f"\n--- Target categories NOT found in your data ---")
                for missing in sorted(missing_targets):
                    print(f"✗ {missing}")
        
        # Show a sample of all categories in your data
        print(f"\n--- Sample of OECD categories in your data (first 50 unique) ---")
        sorted_all_categories = sorted(present_targets)
        for i, cat in enumerate(sorted_all_categories[:50]):
            print(f"{i+1:2d}. {cat}")
        
        if len(sorted_all_categories) > 50:
            print(f"... and {len(sorted_all_categories) - 50} more categories")
        
        # Look for potential matches using partial string matching across all categories
        if target_categories:
            print(f"\n--- Potential similar categories you might want to include ---")
            potential_matches = set()
            for target in target_categories:
                for actual in present_targets:
                    # Check if there's significant overlap
                    target_words = set(target.lower().split())
                    actual_words = set(actual.lower().split())
                    if len(target_words & actual_words) >= 2:  # >=2 words in common
                        if actual not in target_categories:
                            potential_matches.add((target, actual))
            
            for target, actual in sorted(potential_matches):
                print(f"Target: '{target}' → Similar category: '{actual}'")
    
    # Function to check if ANY of the categories matches a target category
    def contains_target_category(categories_str):
        if pd.isna(categories_str):
            return False
        
        # Split categories by semicolon and strip whitespace
        categories = [cat.strip() for cat in str(categories_str).split(';')]
        
        # If target list is empty → keep ALL rows that contain at least one
        # OECD category. Otherwise, keep rows having ANY target category.
        if not target_categories:
            return True

        return any(cat in target_categories for cat in categories)
    
    # Apply the filter
    print("\n=== APPLYING FILTER ===")
    if target_categories:
        print("Keeping rows that contain AT LEAST ONE of the specified target categories…")
    else:
        print("No target categories supplied → keeping ALL rows (with a non-blank OECD category)…")

    mask = df['OECD_Research_Area'].apply(contains_target_category)
    filtered_df = df[mask].copy()
    
    # ------------------------------------------------------------------
    # EXPLODE ROWS: create one row per OECD category
    # ------------------------------------------------------------------
    # Keep original column for reference but create a new column that holds
    # individual categories. We will explode on that.
    filtered_df['OECD_Category'] = (
        filtered_df['OECD_Research_Area']
        .apply(lambda x: [c.strip() for c in str(x).split(';') if c.strip()])
    )

    # Perform explode so each category is its own row
    filtered_df = filtered_df.explode('OECD_Category').reset_index(drop=True)
    
    if target_categories:
        print(f"Rows where ANY category matches targets: {len(filtered_df)}")
    else:
        print(f"Rows with a non-empty OECD category: {len(filtered_df)}")
    
    # Show some statistics about the filtered data
    if len(filtered_df) > 0:
        print("\nSample of filtered technologies:")
        # Detect tech-level column if present
        tech_level_cols = [c for c in filtered_df.columns if 'tech' in c.lower() and 'level' in c.lower()]
        sample_cols = ['technology_name', 'OECD_Category']
        if tech_level_cols:
            sample_cols.append(tech_level_cols[0])  # take the first match

        print(filtered_df[sample_cols].head(10))
        
        # Count frequency of each target category in the filtered results
        if target_categories:
            print("\nFrequency of target categories in filtered results:")
        else:
            print("\nFrequency of OECD categories in filtered results:")
        
        category_counts = {}
        for _, row in filtered_df.iterrows():
            c = row['OECD_Category']
            if not target_categories or c in target_categories:
                category_counts[c] = category_counts.get(c, 0) + 1
        
        # Sort by frequency
        sorted_counts = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        for category, count in sorted_counts:
            print(f"  {category}: {count}")
    
    # Save to file if output_file is specified
    if output_file:
        print(f"\nSaving filtered results to {output_file}...")
        filtered_df.to_excel(output_file, index=False)
        print("File saved successfully!")
    
    return filtered_df

def main():
    """
    Main function to run the filtering process
    """
    # Input file path - CHANGE THIS to your actual file name
    # input_file = 'your_large_file.xlsx'  # Replace with your actual filename
    
    # Output file path (optional)
    output_file = 'filtered_technologies.xlsx'
    
    try:
        # Run the filtering with debug mode enabled
        print("Running in DEBUG mode to analyze your data...")
        filtered_data = filter_technologies_by_categories(input_file, output_file, debug_mode=True)
        
        print(f"\n=== RESULTS ===")
        print(f"Filtering completed successfully!")
        print(f"Filtered dataset contains {len(filtered_data)} technologies")
        
        # Additional analysis
        if len(filtered_data) > 0:
            print(f"\nImpact score statistics for filtered technologies:")
            print(f"  Mean: {filtered_data['Impact_score'].mean():.2f}")
            print(f"  Median: {filtered_data['Impact_score'].median():.2f}")
            print(f"  Min: {filtered_data['Impact_score'].min():.2f}")
            print(f"  Max: {filtered_data['Impact_score'].max():.2f}")
            
            print(f"\nImpact rank distribution:")
            print(filtered_data['Impact_rank_category'].value_counts())
        else:
            print("\n⚠️  NO RESULTS FOUND!")
            print("This suggests the FIRST categories in your file don't match the target categories.")
            print("Check the debug output above to see what FIRST categories are actually in your data.")
            print("You may need to adjust your target categories based on the 'Similar FIRST categories' suggestions.")
        
    except FileNotFoundError:
        print(f"Error: Could not find the input file '{input_file}'")
        print("Please make sure the Excel file is in the same directory as this script.")
        print("And update the 'input_file' variable in the main() function with your actual filename.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()