
import pandas as pd
try:
    df = pd.read_csv('../../fragmentation_results.csv')
    # Filter for non-empty nodes
    df = df[df['compute_util'] > 0]
    
    print("Dataset Summary:")
    print(df.groupby('method')[['compute_util', 'mem_util']].describe())
    
    print("\nSample Data (Cardinal):")
    print(df[df['method'].str.contains('cardinal')].head(10))
    
    print("\nUnique Value Pairs (Cardinal):")
    card_df = df[df['method'].str.contains('cardinal')]
    print(card_df.groupby(['compute_util', 'mem_util']).size())

except Exception as e:
    print(f"Error: {e}")
