#!/usr/bin/env python3
"""
Convert various data formats (TSV, CSV, TXT) to Parquet format for better compression and faster processing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import Optional, Dict
import warnings

warnings.filterwarnings('ignore')


class DataToParquetConverter:
    """Convert data files to Parquet format"""
    
    def __init__(self, base_dir: str = "data"):
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / "raw"
        self.processed_dir = self.base_dir / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
    def convert_open_food_facts(self, input_file: str, output_file: Optional[str] = None,
                                chunksize: int = 100000) -> bool:
        """
        Convert Open Food Facts TSV to Parquet with chunked processing for large files
        
        Args:
            input_file: Path to input TSV file
            output_file: Path to output Parquet file (optional)
            chunksize: Number of rows to process at a time
            
        Returns:
            True if successful, False otherwise
        """
        input_path = Path(input_file)
        
        if not input_path.exists():
            print(f"Error: File not found: {input_path}")
            return False
        
        if output_file is None:
            output_file = self.processed_dir / f"{input_path.stem}.parquet"
        else:
            output_file = Path(output_file)
        
        print(f"Converting Open Food Facts data: {input_path.name}")
        print(f"  Input size: {input_path.stat().st_size / (1024**3):.2f} GB")
        print(f"  Processing in chunks of {chunksize:,} rows...")
        
        try:
            # Key columns for nutrition classification
            columns_to_load = [
                'product_name',
                'brands',
                'categories',
                'countries',
                'energy_100g',
                'fat_100g',
                'saturated-fat_100g',
                'carbohydrates_100g',
                'sugars_100g',
                'fiber_100g',
                'proteins_100g',
                'salt_100g',
                'sodium_100g',
                'nutrition_grade_fr',
            ]
            
            # Process in chunks to handle large files
            chunks = []
            total_rows = 0
            
            for i, chunk in enumerate(pd.read_csv(input_path, sep='\t', 
                                                   usecols=columns_to_load,
                                                   chunksize=chunksize,
                                                   low_memory=False,
                                                   on_bad_lines='skip')):
                total_rows += len(chunk)
                chunks.append(chunk)
                print(f"  Processed chunk {i+1}: {total_rows:,} rows", end='\r')
            
            print(f"\n  Concatenating {len(chunks)} chunks...")
            df = pd.concat(chunks, ignore_index=True)
            
            # Save to Parquet with compression
            print(f"  Writing to Parquet with snappy compression...")
            df.to_parquet(output_file, engine='pyarrow', compression='snappy', index=False)
            
            output_size = output_file.stat().st_size / (1024**3)
            input_size = input_path.stat().st_size / (1024**3)
            compression_ratio = (1 - output_size / input_size) * 100
            
            print(f"\n✓ Successfully converted!")
            print(f"  Output file: {output_file}")
            print(f"  Output size: {output_size:.2f} GB")
            print(f"  Compression: {compression_ratio:.1f}% reduction")
            print(f"  Total rows: {len(df):,}")
            print(f"  Columns: {len(df.columns)}")
            
            return True
            
        except Exception as e:
            print(f"\n✗ Error converting file: {e}")
            return False
    
    def convert_usda_sr28(self, data_dir: str, output_dir: Optional[str] = None) -> bool:
        """
        Convert USDA SR28 database files to Parquet format
        
        Args:
            data_dir: Directory containing SR28 ASCII files
            output_dir: Output directory for Parquet files (optional)
            
        Returns:
            True if successful, False otherwise
        """
        data_path = Path(data_dir)
        
        if not data_path.exists():
            print(f"Error: Directory not found: {data_path}")
            return False
        
        if output_dir is None:
            output_dir = self.processed_dir / "usda_sr28"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nConverting USDA SR28 data from: {data_path}")
        
        # Define file structures based on SR28 documentation
        file_specs = {
            'FOOD_DES.txt': {
                'names': ['NDB_No', 'FdGrp_Cd', 'Long_Desc', 'Shrt_Desc', 'ComName', 
                         'ManufacName', 'Survey', 'Ref_desc', 'Refuse', 'SciName',
                         'N_Factor', 'Pro_Factor', 'Fat_Factor', 'CHO_Factor'],
                'delimiter': '^',
                'quotechar': '~'
            },
            'FD_GROUP.txt': {
                'names': ['FdGrp_Cd', 'FdGrp_Desc'],
                'delimiter': '^',
                'quotechar': '~'
            },
            'NUT_DATA.txt': {
                'names': ['NDB_No', 'Nutr_No', 'Nutr_Val', 'Num_Data_Pts', 'Std_Error',
                         'Src_Cd', 'Deriv_Cd', 'Ref_NDB_No', 'Add_Nutr_Mark', 'Num_Studies',
                         'Min', 'Max', 'DF', 'Low_EB', 'Up_EB', 'Stat_cmt', 'AddMod_Date',
                         'CC'],
                'delimiter': '^',
                'quotechar': '~'
            },
            'NUTR_DEF.txt': {
                'names': ['Nutr_No', 'Units', 'Tagname', 'NutrDesc', 'Num_Dec', 'SR_Order'],
                'delimiter': '^',
                'quotechar': '~'
            },
            'WEIGHT.txt': {
                'names': ['NDB_No', 'Seq', 'Amount', 'Msre_Desc', 'Gm_Wgt', 'Num_Data_Pts', 'Std_Dev'],
                'delimiter': '^',
                'quotechar': '~'
            },
        }
        
        converted_files = []
        
        for filename, spec in file_specs.items():
            file_path = data_path / filename
            
            if not file_path.exists():
                print(f"  Skipping {filename} (not found)")
                continue
            
            try:
                print(f"\n  Converting {filename}...")
                
                # Read the text file with proper delimiter
                df = pd.read_csv(
                    file_path,
                    sep=spec['delimiter'],
                    names=spec['names'],
                    quotechar=spec['quotechar'],
                    encoding='latin-1',
                    on_bad_lines='skip'
                )
                
                # Clean up any remaining delimiters in text
                for col in df.select_dtypes(include=['object']).columns:
                    df[col] = df[col].str.strip().str.replace('^', '', regex=False)
                
                # Save to Parquet
                output_file = output_dir / f"{filename.replace('.txt', '.parquet')}"
                df.to_parquet(output_file, engine='pyarrow', compression='snappy', index=False)
                
                input_size = file_path.stat().st_size / 1024  # KB
                output_size = output_file.stat().st_size / 1024  # KB
                compression_ratio = (1 - output_size / input_size) * 100
                
                print(f"    ✓ {filename}: {len(df):,} rows, {len(df.columns)} columns")
                print(f"      Size: {input_size:.1f}KB → {output_size:.1f}KB ({compression_ratio:.1f}% reduction)")
                
                converted_files.append(filename)
                
            except Exception as e:
                print(f"    ✗ Error converting {filename}: {e}")
                continue
        
        if converted_files:
            print(f"\n✓ Successfully converted {len(converted_files)} USDA SR28 files")
            print(f"  Output directory: {output_dir}")
            return True
        else:
            print("\n✗ No files were converted")
            return False
    
    def convert_csv_to_parquet(self, input_file: str, output_file: Optional[str] = None,
                               **read_csv_kwargs) -> bool:
        """
        Convert a generic CSV file to Parquet
        
        Args:
            input_file: Path to input CSV file
            output_file: Path to output Parquet file (optional)
            **read_csv_kwargs: Additional arguments for pd.read_csv
            
        Returns:
            True if successful, False otherwise
        """
        input_path = Path(input_file)
        
        if not input_path.exists():
            print(f"Error: File not found: {input_path}")
            return False
        
        if output_file is None:
            output_file = self.processed_dir / f"{input_path.stem}.parquet"
        else:
            output_file = Path(output_file)
        
        print(f"\nConverting CSV: {input_path.name}")
        
        try:
            df = pd.read_csv(input_path, **read_csv_kwargs)
            df.to_parquet(output_file, engine='pyarrow', compression='snappy', index=False)
            
            input_size = input_path.stat().st_size / 1024  # KB
            output_size = output_file.stat().st_size / 1024  # KB
            compression_ratio = (1 - output_size / input_size) * 100
            
            print(f"  ✓ Converted: {len(df):,} rows, {len(df.columns)} columns")
            print(f"    Size: {input_size:.1f}KB → {output_size:.1f}KB ({compression_ratio:.1f}% reduction)")
            print(f"    Output: {output_file}")
            
            return True
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return False


def main():
    """Main conversion script"""
    converter = DataToParquetConverter()
    
    print("="*80)
    print("DATA TO PARQUET CONVERTER")
    print("="*80)
    
    # Convert Open Food Facts
    print("\n1. Converting Open Food Facts dataset...")
    print("-"*80)
    off_file = "data/raw/en.openfoodfacts.org.products.tsv"
    if Path(off_file).exists():
        converter.convert_open_food_facts(off_file)
    else:
        print(f"  Skipped: {off_file} not found")
    
    # Convert USDA SR28
    print("\n2. Converting USDA SR28 database files...")
    print("-"*80)
    usda_dir = "data/raw/25060841/sr28asc"
    if Path(usda_dir).exists():
        converter.convert_usda_sr28(usda_dir)
    else:
        print(f"  Skipped: {usda_dir} not found")
    
    print("\n" + "="*80)
    print("CONVERSION COMPLETE!")
    print("="*80)
    print(f"\nProcessed files are in: {converter.processed_dir}")
    print("\nTo load Parquet files in your code:")
    print("  df = pd.read_parquet('data/processed/filename.parquet')")
    print("\nBenefits of Parquet format:")
    print("  • 50-80% smaller file size (compressed)")
    print("  • 5-10x faster to read")
    print("  • Preserves data types")
    print("  • Column-based storage (efficient for analytics)")
    

if __name__ == "__main__":
    main()

