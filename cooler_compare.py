import argparse
import cooler
import numpy as np
import pandas as pd
from tqdm import tqdm

def remap_pixels(query_cooler, target_bins_df, agg_func='mean', chrom=None):
    """
    Remaps pixels from a query cooler to a new set of bins.

    This function takes a cooler object and a DataFrame of target bins. It iterates
    through each chromosome, mapping the pixels from the query cooler onto the
    binning scheme provided by `target_bins_df`.

    Args:
        query_cooler (cooler.Cooler): The cooler object whose pixels will be remapped.
        target_bins_df (pd.DataFrame): A DataFrame with 'chrom', 'start', 'end' columns
                                       defining the target binning.
        agg_func (str): The function to use for aggregating multiple query pixels that
                        map to the same target pixel ('mean', 'sum', 'max', 'min').

    Returns:
        pd.DataFrame: A DataFrame of pixels ('bin1_id', 'bin2_id', 'count') remapped
                      to the target binning scheme.
    """
    query_bins = query_cooler.bins()[:]
    
    # Pre-calculate midpoints for efficient mapping
    query_bins['mid'] = (query_bins['start'] + query_bins['end']) // 2
    target_bins_df['mid'] = (target_bins_df['start'] + target_bins_df['end']) // 2

    all_remapped_pixels = []
    
    # Process one chromosome at a time for memory efficiency
    chroms_to_process = [
        c for c in query_cooler.chromnames 
        if c in target_bins_df['chrom'].unique()
    ]

    print(f"Found {len(chroms_to_process)} common chromosomes to process.")
    print(chroms_to_process)
    if chrom:
        chroms_to_process = [chrom] if chrom in chroms_to_process else []
        if not chroms_to_process:
            print(f"Warning: Chromosome '{chrom}' not found in both coolers.")
            return pd.DataFrame(columns=['bin1_id', 'bin2_id', 'count'])
        print(f"Processing only chromosome: {chrom}")
    query_pixels = query_cooler.pixels()[:]
    for chrom in tqdm(chroms_to_process, desc="Remapping chromosomes"):
        # Fetch data for the current chromosome
        query_bins_chrom = query_bins[query_bins['chrom'] == chrom]
        query_pixels_chrom = query_pixels[query_pixels['bin1_id'].isin(query_bins_chrom.index) &
                                         query_pixels['bin2_id'].isin(query_bins_chrom.index)]
        if query_pixels_chrom.empty:
            continue

        
        target_bins_chrom = target_bins_df[target_bins_df['chrom'] == chrom]
        
        if target_bins_chrom.empty:
            continue

        # Get the global index offset for the target bins on this chromosome
        target_chrom_offset = target_bins_chrom.index.min()

        # Get the midpoints of the source pixels' bins
        bin1_mids = query_bins_chrom.loc[query_pixels_chrom['bin1_id']]['mid'].values
        bin2_mids = query_bins_chrom.loc[query_pixels_chrom['bin2_id']]['mid'].values

        # Use searchsorted to find the new bin indices for each pixel's midpoint
        # `searchsorted` finds the index where an element would be inserted to maintain order.
        # This effectively tells us which target bin each query midpoint falls into.
        new_bin1_indices = np.searchsorted(target_bins_chrom['mid'].values, bin1_mids, side='right') - 1
        new_bin2_indices = np.searchsorted(target_bins_chrom['mid'].values, bin2_mids, side='right') - 1

        # Create a DataFrame for the new pixels, converting local indices to global ones
        chrom_remapped_pixels = pd.DataFrame({
            'bin1_id': new_bin1_indices + target_chrom_offset,
            'bin2_id': new_bin2_indices + target_chrom_offset,
            'count': query_pixels_chrom['count'].values
        })
        
        # Ensure bin1_id is always less than or equal to bin2_id (upper triangle)
        mask = chrom_remapped_pixels['bin1_id'] > chrom_remapped_pixels['bin2_id']
        chrom_remapped_pixels.loc[mask, ['bin1_id', 'bin2_id']] = \
            chrom_remapped_pixels.loc[mask, ['bin2_id', 'bin1_id']].values
        
        all_remapped_pixels.append(chrom_remapped_pixels)

    if not all_remapped_pixels:
        print("Warning: No pixels were remapped. Check if chromosomes match.")
        return pd.DataFrame(columns=['bin1_id', 'bin2_id', 'count'])

    # Concatenate all chromosomal pixel chunks
    new_pixels_df = pd.concat(all_remapped_pixels, ignore_index=True)
    
    # Aggregate pixels that fall into the same new bin
    print(f"Aggregating remapped pixels using '{agg_func}'...")
    aggregated_pixels = new_pixels_df.groupby(['bin1_id', 'bin2_id'])['count'].agg(agg_func).reset_index()

    return aggregated_pixels


def main():
    """
    Main function to compare two cooler files.
    """
    parser = argparse.ArgumentParser(
        description="""
        Compares two cooler files by remapping one to the other's binning
        and calculating the Spearman correlation of their contact matrices.
        """
    )
    parser.add_argument(
        "--ref-cooler",
        type=str,
        required=True,
        help="Path to the reference cooler file. Its binning will be used as the target.",
    )
    parser.add_argument(
        "--query-cooler",
        type=str,
        required=True,
        help="Path to the query cooler file, which will be remapped to the reference binning.",
    )
    parser.add_argument(
        "--out-rebinned",
        type=str,
        default=None,
        help="(Optional) Path to save the remapped query cooler file for inspection."
    )
    parser.add_argument(
        "--chrom",
        type=str,
        default=None,
        help="(Optional) Specific chromosome to process. If not provided, all chromosomes will be processed."
    )

    args = parser.parse_args()

    # --- 1. Load Coolers and Define Target Bins ---
    print(f"Loading reference cooler: {args.ref_cooler}")
    ref_cooler = cooler.Cooler(args.ref_cooler)

    print(f"Loading query cooler: {args.query_cooler}")
    query_cooler = cooler.Cooler(args.query_cooler)
    
    ref_bins = ref_cooler.bins()[:]
    
    # --- 2. Remap Query Cooler to Reference Bins ---
    print("\nStarting pixel remapping process...")
    remapped_query_pixels = remap_pixels(query_cooler, ref_bins.copy(), chrom=args.chrom)
    
    if remapped_query_pixels.empty:
        print("Could not compute correlation because no common pixels were found after remapping.")
        return

    # --- 3. (Optional) Save the remapped cooler ---
    if args.out_rebinned:
        print(f"\nSaving remapped query cooler to: {args.out_rebinned}")
        cooler.create_cooler(
            args.out_rebinned,
            bins=ref_bins,
            pixels=remapped_query_pixels,
            dtypes={'count': np.float64}
        )
    
    # --- 4. Fetch Pixel Data for Correlation ---
    print("\nFetching pixel data for correlation analysis...")
    value_col = 'count'
    ref_pixels = ref_cooler.pixels()[:][['bin1_id', 'bin2_id', value_col]]
    if args.chrom:
        ref_pixels = ref_pixels[ref_pixels['bin1_id'].isin(ref_bins[ref_bins['chrom'] == args.chrom].index) &
                                ref_pixels['bin2_id'].isin(ref_bins[ref_bins['chrom'] == args.chrom].index)]
    ref_pixels = ref_pixels.rename(columns={value_col: 'count_ref'})
    query_pixels_final = remapped_query_pixels.rename(columns={'count': 'count_query'})
    if args.chrom:
        query_pixels_final = query_pixels_final[query_pixels_final['bin1_id'].isin(ref_bins[ref_bins['chrom'] == args.chrom].index) &
                                                query_pixels_final['bin2_id'].isin(ref_bins[ref_bins['chrom'] == args.chrom].index)]

    print(f"Reference pixels: {len(ref_pixels)}")
    print(ref_pixels.head())
    print(f"Query pixels after remapping: {len(query_pixels_final)}")
    print(query_pixels_final.head())

    # --- 5. Merge and Compute Correlation ---
    print("Aligning pixel dataframes...")
    merged_df = pd.merge(
        ref_pixels,
        query_pixels_final,
        on=['bin1_id', 'bin2_id'],
        how='inner' # Only compare pixels present in both datasets?
    )
    merged_df.dropna(inplace=True)

    if merged_df.empty:
        print("Correlation cannot be computed: The two datasets have no overlapping pixels.")
        return

    print(f"Calculating Spearman correlation on {len(merged_df)} common pixels.")
    correlation = merged_df['count_ref'].corr(merged_df['count_query'], method='spearman')

    print("\n" + "="*40)
    print(f"Spearman Correlation: {correlation:.6f}")
    print("="*40 + "\n")


if __name__ == "__main__":
    main()
