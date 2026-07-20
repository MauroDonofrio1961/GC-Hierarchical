
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def _save(fig,out,name,book):
    fig.tight_layout()
    fig.savefig(out/f"{name}.png",dpi=180,bbox_inches="tight")
    fig.savefig(out/f"{name}.pdf",bbox_inches="tight")
    book.savefig(fig,bbox_inches="tight"); plt.close(fig)

def make_plots(cfg,science,background,theta,membership,out,chain=None):
    d=science.copy()
    d["P_blue"]=membership[:,0]; d["P_red"]=membership[:,1]; d["P_background"]=membership[:,2]
    d["P_GC"]=d["P_blue"]+d["P_red"]
    d.to_csv(out/"object_membership_probabilities.csv",index=False)

    with PdfPages(out/"all_figures_v5.pdf") as book:
        fig,ax=plt.subplots(figsize=(7,6))
        sc=ax.scatter(d.color,d.rmag,c=d.P_background,s=14,vmin=0,vmax=1)
        ax.invert_yaxis(); ax.set(xlabel="g − r (AB)",ylabel="r (AB)",
            title="Posterior background probability")
        fig.colorbar(sc,ax=ax,label="P(background | data)")
        _save(fig,out,"01_background_membership_cmd",book)

        fig,ax=plt.subplots(figsize=(7,6))
        rgb=np.column_stack([d.P_red,d.P_blue,np.zeros(len(d))])
        ax.scatter(d.color,d.rmag,c=np.clip(rgb,0,1),s=14)
        ax.invert_yaxis(); ax.set(xlabel="g − r (AB)",ylabel="r (AB)",
            title="Blue and red GC membership")
        _save(fig,out,"02_blue_red_membership_cmd",book)

        for col,xlabel,name,rng in [
          ("rmag","r (AB)","03_r_membership",cfg["sample"]["r_range"]),
          ("color","g − r","04_color_membership",cfg["sample"]["color_range"]),
          ("radius_arcmin","Radius (arcmin)","05_radius_membership",cfg["sample"]["radius_range_arcmin"])]:
            fig,ax=plt.subplots(figsize=(7,5)); bins=np.linspace(*rng,26)
            ax.hist(d[col],bins=bins,weights=d.P_blue,density=True,histtype="step",linewidth=2,label="Blue GC")
            ax.hist(d[col],bins=bins,weights=d.P_red,density=True,histtype="step",linewidth=2,label="Red GC")
            ax.hist(d[col],bins=bins,weights=d.P_background,density=True,histtype="step",linewidth=1.5,label="Background")
            if col=="rmag": ax.invert_xaxis()
            ax.set(xlabel=xlabel,ylabel="Density",title=xlabel+" posterior decomposition"); ax.legend()
            _save(fig,out,name,book)

        fig,ax=plt.subplots(figsize=(7,5))
        ax.hist(d.P_background,bins=np.linspace(0,1,21),histtype="step",linewidth=2)
        ax.set(xlabel="P(background | data)",ylabel="Objects",title="Background-membership distribution")
        _save(fig,out,"06_background_probability_histogram",book)

        fig,ax=plt.subplots(figsize=(7,5))
        ax.scatter(d.TotL,d.P_background,s=10,alpha=.45)
        ax.set(xlabel="Catalogue TotL",ylabel="P(background | data)",title="Background probability versus TotL")
        _save(fig,out,"07_background_vs_TotL",book)

        if chain is not None:
            try:
                import corner
                fig=corner.corner(chain,labels=cfg["parameters"]["names"],show_titles=True)
                _save(fig,out,"08_corner",book)
            except Exception:
                pass
