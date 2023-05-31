import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

plt.switch_backend('agg')

def initialize_centers(data, n_clusters):
  idx = np.random.randint(0, data.shape[0], n_clusters)
  centers = data[idx]
  return centers

def calculate_membership(data, centers, m, distances=None):
  if distances is None:
    distances = np.linalg.norm(data[:, None, :] - centers, axis=2)
  
  membership_matrix = 1 / (distances ** (2 / (m - 1)) + 1e-8)
  membership_matrix = membership_matrix / np.sum(membership_matrix, axis=1)[:, None]
  return membership_matrix

def calculate_centers(data, membership_matrix, m):
  numerator = np.dot(membership_matrix.T, data)
  denominator = np.sum(membership_matrix.T, axis=1)[:, None]
  centers = numerator / (denominator + 1e-8)
  return centers

def fuzzy_c_means(data, n_clusters, m, epsilon=0.001, max_iter=10):
  centers = initialize_centers(data, n_clusters)
  distances = None
  
  for iter in range(max_iter):
    prev_centers = centers.copy()
    membership_matrix = calculate_membership(data, centers, m, distances=distances)
    centers = calculate_centers(data, membership_matrix ** m, m)
    distances = np.linalg.norm(data[:, None, :] - centers, axis=2)
    
    if np.linalg.norm(centers - prev_centers) < epsilon:
      break

  # Calculate FPC
  fpc = np.sum(membership_matrix ** 2) / (data.shape[0] * n_clusters)
  return centers, membership_matrix

def preprocessing(file):
  df = pd.read_excel(file)
  df.columns = [col.strip().lower().replace(' ', '') for col in df.columns]
  selected_columns = ['invoiceno', 'stokawal', 'stokakhir', 'namabarang', 'jenisbarang', 'kodebarang', 'kuantitasbarang', 'invoicedate']
  df = df[selected_columns]
  df.drop_duplicates(inplace=True)
  df.dropna(axis=0, inplace = True)
  return df

def concatenate(df1, df2, df3):
  df = pd.concat([df1, df2, df3])
  df.reset_index(drop=True, inplace=True)
  
  df['invoice_date'] = pd.to_datetime(df["invoicedate"], format="%Y/%m/%d", errors='coerce')
  df.dropna(inplace = True)
  df['month'] = df['invoice_date'].dt.month.astype('int')
  df['day'] = df['invoice_date'].dt.day.astype('int')
  df['year'] = df['invoice_date'].dt.year.astype('int')
  df.drop(columns=['invoice_date', 'invoicedate'], axis=1, inplace=True)

  return df

def fuzzy_clustering(df):
  df['jenisbarang'] = pd.factorize(df['jenisbarang'])[0]
  df_value = df[['stokawal', 'stokakhir', 'kuantitasbarang', 'jenisbarang']]
  scaler = MinMaxScaler()
  df_scaled = scaler.fit_transform(df_value)
  pca = PCA(n_components=3)
  df_pca = pca.fit_transform(df_scaled)

  m = 2
  n_clusters = 3
  epsilon = 0.0001
  max_iter = 100

  centers, membership_matrix = fuzzy_c_means(df_pca, n_clusters, m, epsilon=epsilon, max_iter=max_iter)

  cluster_membership = np.argmax(membership_matrix, axis=1)
  cluster_labels = ['Sangat Laris', 'Laris', 'Tidak Laris', ]
  df['cluster'] = [cluster_labels[c] for c in cluster_membership]

  fig, ax = plt.subplots(figsize=(10,6))
  scatter = ax.scatter(df_pca[:,0], df_pca[:,1], c=cluster_membership, cmap='rainbow')
  
  legend = ax.legend(*scatter.legend_elements(), loc="upper right", title="Clusters")
  legend.set_title("Clusters")

  for i, label in enumerate(cluster_labels):
    legend.get_texts()[i].set_text(label)

  for i in range(n_clusters):
    plt.scatter(centers[i][0], centers[i][1], marker='x', s=100, linewidths=2, color='black')

  plt.xlabel('PC 1')
  plt.ylabel('PC 2')

  plt.savefig('static/files/clusters.png')

  df_grouped = df.groupby(['cluster', 'namabarang']).size().reset_index(name='jumlah')
  df_grouped_sl = df_grouped[df_grouped['cluster'] == 'Sangat Laris'].sort_values(by='jumlah', ascending=False)[:10]
  fig, ax = plt.subplots(figsize=(10,10))

  ax.bar(df_grouped_sl['namabarang'], df_grouped_sl['jumlah'])
  ax.set_xticks(range(len(df_grouped_sl['namabarang'])))
  ax.set_xticklabels(df_grouped_sl['namabarang'], rotation=90)
  ax.set_title('10 Barang Sangat Laris')
  ax.set_xlabel('Nama Barang')
  ax.set_ylabel('Jumlah')

  plt.tight_layout()
  plt.savefig('files/bsl.png', bbox_inches='tight')

  df.to_csv('files/new_clustering.csv', index=False)

  cluster_count = df['cluster'].value_counts()
  data = cluster_count.to_dict()
  
  n_samples = df_pca.shape[0]
  pc = np.sum(membership_matrix ** 2) / n_samples
  mpc = (pc - 1/n_clusters) / (1 - 1/n_clusters)
  data['mpc'] = mpc
  return data