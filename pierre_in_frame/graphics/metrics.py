import matplotlib
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.interpolate import interpn

from settings.path_dir_file import PathDirFile

matplotlib.use('Agg')

import matplotlib.pyplot as plt

matplotlib.style.use('ggplot')
from scipy.stats import gaussian_kde
import numpy as np


def map_mrmc_performance_graphic(data, dataset_name):
    # Ordenação dos dados pelo eixo y
    df = data.sort_values(by=['PERFORMANCE'], ascending=[True])
    # Obtenção dos dados do eixo x e do eixo y
    x_data = df['MAP'].tolist()
    y_data = df['MRMC'].tolist()
    z_data = df['PERFORMANCE'].tolist()
    # Figura iniciada e atribuição das labels dos eixos x e y
    matplotlib.use('Agg')
    matplotlib.style.use('ggplot')
    plt.figure()
    plt.grid(True)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    plt.xlabel('MAP', fontsize=14)
    plt.ylabel('MRMC', fontsize=14)
    plt.xticks(rotation=30)
    # Dados desenhados na figura
    plt.scatter(x=x_data, y=y_data, c=z_data,
                alpha=0.5, label='AAA', cmap='viridis')
    plt.colorbar(label='Performance')
    # Pasta para salvar a figura
    file_dir = PathDirFile.set_graphics_file(dataset_name, 'map_mrmc_performance.png')
    # Salvar figura no disco
    plt.savefig(
        file_dir,
        format='png',
        dpi=400,
        quality=100,
        bbox_inches='tight'
    )
    # Figura fechada
    plt.close('all')


def map_mace_performance_graphic(data, dataset_name):
    # Ordenação dos dados pelo eixo y
    df = data.sort_values(by=['PERFORMANCE'], ascending=[True])
    # Obtenção dos dados do eixo x e do eixo y
    x_data = df['MAP'].tolist()
    y_data = df['MACE'].tolist()
    z_data = df['PERFORMANCE'].tolist()
    # Figura iniciada e atribuição das labels dos eixos x e y
    matplotlib.use('Agg')
    matplotlib.style.use('ggplot')
    plt.figure()
    plt.grid(True)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    plt.xlabel('MAP', fontsize=14)
    plt.ylabel('MACE', fontsize=14)
    plt.xticks(rotation=30)
    # Dados desenhados na figura
    plt.scatter(x=x_data, y=y_data, c=z_data,
                alpha=0.5, label='AAA', cmap='viridis')
    plt.colorbar(label='Performance')
    # Pasta para salvar a figura
    file_dir = PathDirFile.set_graphics_file(dataset_name, 'map_mace_performance.png')
    # Salvar figura no disco
    plt.savefig(
        file_dir,
        format='png',
        dpi=400,
        quality=100,
        bbox_inches='tight'
    )
    # Figura fechada
    plt.close('all')


def map_mrr_graphic(data, dataset_name):
    # Ordenação dos dados pelo eixo y
    df = data.sort_values(by=['MAP'], ascending=[True])
    # Obtenção dos dados do eixo x e do eixo y
    x_data = df['MAP'].tolist()
    y_data = df['MRR'].tolist()
    xy = np.vstack([x_data, y_data])
    z_data = gaussian_kde(xy)(xy)

    # Figura iniciada e atribuição das labels dos eixos x e y
    matplotlib.use('Agg')
    matplotlib.style.use('ggplot')
    plt.figure()
    plt.grid(True)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    plt.xlabel('MAP', fontsize=14)
    plt.ylabel('MRR', fontsize=14)
    plt.xticks(rotation=30)
    # Dados desenhados na figura
    plt.scatter(x=x_data, y=y_data, c=z_data,  # s=10,
                alpha=0.5, label='AAA', cmap='viridis')
    plt.colorbar(label='Density')
    # Pasta para salvar a figura
    file_dir = PathDirFile.set_graphics_file(dataset_name, 'map_mrr_density.png')
    # Salvar figura no disco
    plt.savefig(
        file_dir,
        format='png',
        dpi=400,
        quality=100,
        bbox_inches='tight'
    )
    # Figura fechada
    plt.close('all')


def map_mrr_graphic_new(data, dataset_name):
    # Ordenação dos dados pelo eixo y
    df = data.sort_values(by=['MAP'], ascending=[True])
    # Obtenção dos dados do eixo x e do eixo y
    x_data = df['MAP'].tolist()
    y_data = df['MRR'].tolist()

    def density_scatter(x, y, ax=None, sort=False, bins=20, **kwargs):
        """
        Scatter plot colored by 2d histogram
        """
        if ax is None:
            fig, ax = plt.subplots()
            plt.rc('xtick', labelsize=16)
            plt.rc('ytick', labelsize=16)
            plt.xlabel('MAP', fontsize=14)
            plt.ylabel('MRR', fontsize=14)
        data, x_e, y_e = np.histogram2d(x, y, bins=bins, density=True)
        z = interpn((0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])), data, np.vstack([x, y]).T,
                    method="splinef2d", bounds_error=False)

        # To be sure to plot all data
        z[np.where(np.isnan(z))] = 0.0

        # Sort the points by density, so that the densest points are plotted last
        if sort:
            idx = z.argsort()
            x, y, z = x[idx], y[idx], z[idx]

        ax.scatter(x, y, c=z, **kwargs)

        norm = Normalize(vmin=np.min(z), vmax=np.max(z))
        cbar = fig.colorbar(cm.ScalarMappable(norm=norm), ax=ax)
        cbar.ax.set_ylabel('Density')

        return ax

    density_scatter(x=x_data, y=y_data)
    # Pasta para salvar a figura
    file_dir = PathDirFile.set_graphics_file(dataset_name, 'map_mrr_density_new.png')
    # Salvar figura no disco
    plt.savefig(
        file_dir,
        format='png',
        dpi=400,
        quality=100,
        bbox_inches='tight'
    )
    # Figura fechada
    plt.close('all')
