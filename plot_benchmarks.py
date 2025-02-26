import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_benchmarks(batch_size=1):
    plt.style.use('default')
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Define results directory based on batch size
    results_dir = os.path.join('results', f'bsz-{batch_size}')
    
    # Read all CSV files from the batch-specific directory
    #df_compressed = pd.read_csv(os.path.join(results_dir, 'benchmark_results_cachecompressed.csv'))
    #df_decompressed = pd.read_csv(os.path.join(results_dir, 'benchmark_results_cachedecompressed.csv'))
    #df_absorbed = pd.read_csv(os.path.join(results_dir, 'benchmark_results_absorbed_cachecompressed.csv'))
    #df_materialized = pd.read_csv(os.path.join(results_dir, 'benchmark_results_absorbedmaterialized_cachecompressed_moveelision.csv'))
    #df_moveelision = pd.read_csv(os.path.join(results_dir, 'benchmark_results_absorbed_cachecompressed_moveelision.csv'))
    df_simple = pd.read_csv(os.path.join(results_dir, 'benchmark_results_simpleattention.csv'))
    df_simple_compressed = pd.read_csv(os.path.join(results_dir, 'benchmark_results_simplecompressedattention.csv'))
    df_simple_absorbed = pd.read_csv(os.path.join(results_dir, 'benchmark_results_simpleabsorbedattention.csv'))
    # Plot compressed data
#     plt.plot(df_compressed['kv_length'], df_compressed['Median'], 
#             marker='o', 
#             linewidth=2,
#             markersize=8,
#             color='blue',
#             label='Compressed KV Cache non-absorbed MLA')
    
#     # Add error bands for compressed
#     plt.fill_between(df_compressed['kv_length'], 
#                     df_compressed['P25'], 
#                     df_compressed['P75'], 
#                     alpha=0.2,
#                     color='blue')
    
#     # Plot decompressed data
#     plt.plot(df_decompressed['kv_length'], df_decompressed['Median'], 
#             marker='s',  # square marker to differentiate
#             linewidth=2,
#             markersize=8,
#             color='red',
#             label='Decompressed KV Cache non-absorbed MLA')
    
#     # Add error bands for decompressed
#     plt.fill_between(df_decompressed['kv_length'], 
#                     df_decompressed['P25'], 
#                     df_decompressed['P75'], 
#                     alpha=0.2,
#                     color='red')
    
#     # Plot attention-absorbed data
#     plt.plot(df_absorbed['kv_length'], df_absorbed['Median'], 
#             marker='^',  # triangle marker to differentiate
#             linewidth=2,
#             markersize=8,
#             color='green',
#             label='Attention-Absorbed')
    
#     # Add error bands for attention-absorbed
#     plt.fill_between(df_absorbed['kv_length'], 
#                     df_absorbed['P25'], 
#                     df_absorbed['P75'], 
#                     alpha=0.2,
#                     color='green')
    
#     # Plot attention-absorbed data with materialization
#     plt.plot(df_materialized['kv_length'], df_materialized['Median'], 
#             marker='d',  # diamond marker to differentiate
#             linewidth=2,
#             markersize=8,
#             color='purple',
#             label='Materialized Attention-Absorbed')
    
#     # # Add error bands for materialized attention-absorbed
#     plt.fill_between(df_materialized['kv_length'], 
#                     df_materialized['P25'], 
#                     df_materialized['P75'], 
#                     alpha=0.2,
#                     color='purple')
    
#     # Plot move-elision data
#     plt.plot(df_moveelision['kv_length'], df_moveelision['Median'], 
#             marker='*',  # star marker to differentiate
#             linewidth=2,
#             markersize=8,
#             color='orange',
#             #label='Attention-Absorbed with Move Elision')
#             label='Absorbed attention')
    
#     # Add error bands for move-elision
#     plt.fill_between(df_moveelision['kv_length'], 
#                     df_moveelision['P25'], 
#                     df_moveelision['P75'], 
#                     alpha=0.2,
#                     color='orange')

    # Plot simple attention data
    plt.plot(df_simple['kv_length'], df_simple['Median'], 
            marker='o', 
            linewidth=2,
            markersize=8,
            color='blue',
            label='Simple Attention')
    
    # Add error bands for simple attention
    plt.fill_between(df_simple['kv_length'], 
                    df_simple['P25'], 
                    df_simple['P75'], 
                    alpha=0.2,
                    color='blue')
    
    # Plot simple compressed attention data
    plt.plot(df_simple_compressed['kv_length'], df_simple_compressed['Median'], 
            marker='s',  # square marker to differentiate
            linewidth=2,
            markersize=8,
            color='red',
            label='Simple Compressed Attention')
    
    # Add error bands for simple compressed attention
    plt.fill_between(df_simple_compressed['kv_length'], 
                    df_simple_compressed['P25'], 
                    df_simple_compressed['P75'], 
                    alpha=0.2,
                    color='red')

    # Plot simple absorbed attention data
    plt.plot(df_simple_absorbed['kv_length'], df_simple_absorbed['Median'], 
            marker='^',  # triangle marker to differentiate
            linewidth=2,
            markersize=8,
            color='green',
            label='Simple Absorbed Attention')
    
    # Add error bands for simple absorbed attention
    plt.fill_between(df_simple_absorbed['kv_length'], 
                    df_simple_absorbed['P25'], 
                    df_simple_absorbed['P75'], 
                    alpha=0.2,
                    color='green')

    # Configure plot
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('KV Cache Length', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.title(f'Cache Performance Comparison batch size = {batch_size}', fontsize=14)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    
    # Save plot in the batch-specific directory
    plot_filename = os.path.join('results', f'bsz-{batch_size}', 'benchmark_comparison.png')
    os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # You can specify different batch sizes here
    plot_benchmarks(batch_size=4)
    print(f"Plot saved  ") 