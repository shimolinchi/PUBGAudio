"""Plot the completed F034 combined update, without changing evaluation rules."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    report=json.loads(args.report.read_text(encoding='utf-8'))
    assert report['status']=='verified_completed_combined_update'
    fig,axes=plt.subplots(2,2,figsize=(13,8.7))
    colors=('#64748b','#007e87');labels=('Parent: 250 epochs','Expanded + augmented: 400 epochs')
    for i,item in enumerate(report['comparisons']):
        folder=Path(item['run'])
        rows=[json.loads(s) for s in (folder/'metrics.jsonl').read_text().splitlines()]
        updates=np.cumsum([r['train_batches'] for r in rows])/1000
        values=[r['validation_external_macro_f1'] for r in rows]
        axes[0,0].plot(updates,values,color=colors[i],alpha=.22,lw=.65)
        axes[0,0].plot(updates,np.maximum.accumulate(values),color=colors[i],lw=2,label=labels[i])
        chosen=item['reports']['detection']['selected_epoch']-1
        axes[0,0].scatter(updates[chosen],values[chosen],color=colors[i],s=35,zorder=4)
        axes[0,1].plot(updates,[r['train_loss'] for r in rows],color=colors[i],lw=1.2,label=labels[i]+' / train')
        axes[0,1].plot(updates,[r['validation_loss'] for r in rows],color=colors[i],ls='--',lw=.8,alpha=.75,label=labels[i]+' / validation')
        test=item['reports']['detection']['test']['diagnostics']['external']['per_class']
        classes=('footsteps','vehicle','gunfire');x=np.arange(3)+(i-.5)*.34
        f1=[test[c]['f1']*100 for c in classes]
        bars=axes[1,0].bar(x,f1,width=.32,color=colors[i],label=labels[i])
        axes[1,0].bar_label(bars,fmt='%.1f',fontsize=8,padding=3)
        angles=[test[c]['conditional_localization']['angular_mae_degrees'] for c in classes]
        bars=axes[1,1].bar(x,angles,width=.32,color=colors[i],label=labels[i])
        axes[1,1].bar_label(bars,fmt='%.1f',fontsize=8,padding=3)
    axes[0,0].set(title='Validation macro F1 (bold: best so far)',ylabel='F1',xlabel='Optimizer updates (thousands)',ylim=(0,1))
    axes[0,0].legend(fontsize=8,loc='lower right')
    axes[0,1].set(title='Training and validation loss',ylabel='Loss (log scale)',xlabel='Optimizer updates (thousands)',yscale='log')
    axes[0,1].legend(fontsize=7)
    for ax in axes[1]:
        ax.set_xticks(np.arange(3),['Footsteps','Vehicle','Gunfire'])
    axes[1,0].set(title='Fixed synthetic test: external detection',ylabel='F1 (%)',ylim=(0,108))
    axes[1,1].set(title='Direction error: detected single-source frames',ylabel='Mean absolute error (degrees)')
    axes[1,1].set_ylim(0,max(65,axes[1,1].get_ylim()[1]*1.2))
    for ax in axes.flat:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle('PUBGAudio | More scenes + light augmentation + longer training',fontsize=15,fontweight='bold',x=.06,ha='left')
    fig.text(.06,.025,'Combined update, not a single-factor ablation. Validation selects checkpoints; threshold = 0.5.\n'
        'Same previously viewed synthetic test split. Localization uses conditional subsets; real-game performance is untested.',fontsize=9,color='#475569')
    fig.tight_layout(rect=(.025,.085,.99,.95),h_pad=2.6,w_pad=2.3)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=160,facecolor='white');plt.close(fig)
    print(args.output)


if __name__=='__main__':main()
