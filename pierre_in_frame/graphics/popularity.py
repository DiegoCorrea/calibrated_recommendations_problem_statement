import matplotlib

from settings.charts import ChartsConfig
from settings.labels import Label
from settings.path_dir_file import PathDirFile

import matplotlib.pyplot as plt


def long_tail_graphic(data, dataset):
    matplotlib.use('Agg')
    matplotlib.style.use('ggplot')
    # Ordenação dos dados pelo eixo y invertido
    df = data.sort_values(by=[Label.TOTAL_TIMES], ascending=[False])
    # Obtenção dos dados do eixo x e do eixo y
    x_data = [i + 1 for i in range(len(df))]
    y_data = df[Label.TOTAL_TIMES].tolist()
    # Figura iniciada e atribuição das labels dos eixos x e y
    plt.figure()
    plt.grid(True)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    plt.xlabel("Number of items", fontsize=ChartsConfig.FONT_SIZE_VALUE)
    plt.ylabel("Item in user profile", fontsize=ChartsConfig.FONT_SIZE_VALUE)
    # Dados desenhados na figura
    plt.scatter(x_data, y_data, alpha=0.5,
                c=ChartsConfig.SCATTER_BUBBLE_COLOR)
    # Linha na horizontal
    short_tail_sum = 0
    medium_tail_sum = 99999
    cut_value = 0
    while short_tail_sum < medium_tail_sum:
        cut_value += 1
        short_tail_sum = (df.iloc[:cut_value])[Label.TOTAL_TIMES].sum()
        medium_tail_sum = (df.iloc[cut_value:])[Label.TOTAL_TIMES].sum()
    cuted_df = df.iloc[:cut_value]
    cut_value = cuted_df[Label.TOTAL_TIMES].min()
    plt.axhline(y=cut_value)
    # Pasta para salvar a figura
    saving_file = PathDirFile.preprocessing_graphics_file(dataset, 'long_tail.eps')
    # Salvar figura no disco
    plt.savefig(
        saving_file,
        format='eps',
        dpi=ChartsConfig.DPI_VALUE,
        bbox_inches='tight'
    )
    # Pasta para salvar a figura
    saving_file = PathDirFile.preprocessing_graphics_file(dataset, 'long_tail.png')
    # Salvar figura no disco
    plt.savefig(
        saving_file,
        format='png',
        dpi=ChartsConfig.DPI_VALUE,
        bbox_inches='tight'
    )
    # Closing figure
    plt.close('all')


def popularity_group_graphic(data, dataset):
    matplotlib.use('Agg')
    matplotlib.style.use('ggplot')

    plt.figure(figsize=(10, 5))

    plt.bar(data['group'], data['values'])

    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    plt.xlabel("Popularity Groups", fontsize=ChartsConfig.FONT_SIZE_VALUE)
    plt.ylabel("Percent of Each Group", fontsize=ChartsConfig.FONT_SIZE_VALUE)

    # Pasta para salvar a figura
    saving_file = PathDirFile.preprocessing_graphics_file(dataset, 'popularity_group.eps')
    # Salvar figura no disco
    plt.savefig(
        saving_file,
        format='eps',
        dpi=ChartsConfig.DPI_VALUE,
        bbox_inches='tight'
    )
    # Pasta para salvar a figura
    saving_file = PathDirFile.preprocessing_graphics_file(dataset, 'popularity_group.png')
    # Salvar figura no disco
    plt.savefig(
        saving_file,
        format='png',
        dpi=ChartsConfig.DPI_VALUE,
        bbox_inches='tight'
    )
    # Closing figure
    plt.close('all')

