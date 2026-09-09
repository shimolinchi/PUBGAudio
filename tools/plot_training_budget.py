"""Plot recorded losses and class F1, without smoothing away late improvements."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def draw(axes,path,title):
    records=[json.loads(s) for s in (path/'metrics.jsonl').read_text().splitlines()]
    epochs=[r['epoch']+1 for r in records]
    train=[r['train_loss'] for r in records]
    validation=[r['validation_loss'] for r in records]
    macro=[sum(v['f1'] or 0 for v in r['validation_diagnostics']['external']['per_class'].values())/3
           for r in records]
    low=min(range(len(records)),key=lambda i:validation[i])
    high=max(range(len(records)),key=lambda i:macro[i])
    ax=axes[0]
    ax.plot(epochs,train,label='Training loss',color='#2563eb',linewidth=1.7)
    ax.plot(epochs,validation,label='Validation loss',color='#dc2626',linewidth=1.7)
    ax.scatter([epochs[low]],[validation[low]],s=50,color='#dc2626',zorder=5)
    ax.axvline(epochs[low],color='#dc2626',alpha=.3,linestyle='--')
    ax.set_title(f'{title}\nLowest validation loss: epoch {epochs[low]}',loc='left',fontsize=11)
    ax.set_ylabel('Combined loss');ax.legend(frameon=False,fontsize=8)
    ax=axes[1]
    for cls,color in [('footsteps','#2563eb'),('vehicle','#d97706'),('gunfire','#059669')]:
        values=[r['validation_diagnostics']['external']['per_class'][cls]['f1'] for r in records]
        ax.plot(epochs,values,label=cls,color=color,alpha=.8,linewidth=1.2)
    ax.plot(epochs,macro,label='Macro F1',color='#111827',linewidth=1.8)
    ax.scatter([epochs[high]],[macro[high]],s=50,color='#111827',zorder=5)
    ax.axvline(epochs[high],color='#111827',alpha=.3,linestyle='--')
    ax.set_title(f'{title}\nHighest external macro F1: epoch {epochs[high]}',loc='left',fontsize=11)
    ax.set_ylabel('Validation frame F1 (threshold 0.5)');ax.set_ylim(0,1.02)
    ax.legend(frameon=False,fontsize=8,ncol=2,loc='lower right')
    for ax in axes:
        ax.set_xlabel('Epoch');ax.grid(alpha=.15)
        ax.spines[['top','right']].set_visible(False)
        if len(records)>30:
            ax.axvline(30,color='#64748b',alpha=.55,linestyle=':')
            ax.text(32,.97,'30-epoch budget',transform=ax.get_xaxis_transform(),
                    ha='left',va='top',fontsize=8,color='#475569')
    return dict(epochs=len(records),best_loss_epoch=epochs[low],best_macro_epoch=epochs[high])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--long-run',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    count=2 if args.long_run else 1
    fig,axes=plt.subplots(count,2,figsize=(11,3.8*count),squeeze=False,layout='constrained')
    summary={'baseline':draw(axes[0],args.baseline,'Original 30-epoch run')}
    if args.long_run:summary['long_run']=draw(axes[1],args.long_run,'Fixed 250-epoch run')
    fig.suptitle('Training budget: lower combined loss and higher detection F1 are different objectives',fontsize=12)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=160,facecolor='white')
    plt.close(fig)
    print(json.dumps(dict(output=str(args.output),**summary)))


if __name__=='__main__':main()
