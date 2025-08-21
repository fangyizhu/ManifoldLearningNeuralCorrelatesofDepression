from torch import cdist
from torch.nn.functional import cosine_similarity

def figures_to_html(figs, filename="dashboard.html"):
    with open(filename, 'w', encoding='utf-8') as dashboard:
        dashboard.write("<html><head></head><body>" + "\n")
        for fig in figs:
            inner_html = fig.to_html().split('<body>')[1].split('</body>')[0]
            dashboard.write(inner_html)
        dashboard.write("</body></html>" + "\n")

def euc(emb1, emb2):
    return cdist(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()

def cos_dist(emb1, emb2):
    return 1 - cosine_similarity(emb1, emb2, dim=0).item()