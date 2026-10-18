"""Canonical presentation paths; frozen experiment paths remain in their protocols."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / 'docs/policy_improvement'
EVIDENCE = DOC / 'evidence'
FIGURES = DOC / 'figures'
EXPORT = ROOT / 'outputs/policy_improvement'
DERIVED = EXPORT / 'evidence'
VECTOR_FIGURES = EXPORT / 'figures'

def ensure_directories():
    for path in [DOC, EVIDENCE, FIGURES, EXPORT, DERIVED, VECTOR_FIGURES]:
        path.mkdir(parents=True, exist_ok=True)

"""Shared display vocabulary and figures for the policy study (no training)."""
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/policy-report-mpl")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

MODEL_NAMES = {
    'historical': 'Original saved model',
    'CONF_U20': 'Baseline model',
    'CONF_U100': 'Baseline with longer training',
    'CONF_C100': 'Revised model',
    'CONF_L100': 'Revised model: original time scale',
    'optimal': 'Best possible in this simulator',
    'planner': 'Planner fitted to the original data',
}
LEGACY_NAMES = dict(zip(['Ordinary 20k', 'Ordinary 100k', 'Frozen candidate', 'Linear-time control'],
                        [MODEL_NAMES[k] for k in ['CONF_U20','CONF_U100','CONF_C100','CONF_L100']]))
EPISODE_NAMES = {'original': 'Original data', 'ordinary': '+400 usual-price periods',
                 'targeted': '+400 late-exploration periods'}
CASE_NAMES = {(200,10): ('A','More time, many rooms'), (200,2): ('B','More time, few rooms'),
              (50,10): ('C','Less time, many rooms'), (1,1): ('D','Last day, one room')}
REVISED_DEFINITION = ('Revised model: additional targeted observations, more attention to late-period records, '
                      'logarithmic time input, and 100,000 training updates.')
PRICES = [55,70,90,110,125]
COLORS = ['#315b80','#39a2ae','#e9c46a','#e58a50','#a13b56']
SEED_COLORS = {10:'#315b80',11:'#39a2ae',12:'#a13b56'}

def case_label(days, rooms):
    marker, name = CASE_NAMES[(days, rooms)]
    return f'{marker} — {name}: {days} '+('day' if days==1 else 'days')+f', {rooms} '+('room' if rooms==1 else 'rooms')

def case_table(rows):
    frame=pd.DataFrame(rows).copy()
    frame.insert(0,'Situation',[case_label(r.days,r.rooms) for r in frame.itertuples()])
    return frame.drop(columns=['days','rooms','case'],errors='ignore').rename(columns={
        'simulator_optimal_price':MODEL_NAMES['optimal'], 'ordinary_20k_all_seeds':'Baseline model (all seeds)',
        **{f'candidate_seed_{s}':f'Revised model (seed {s})' for s in [10,11,12]}})

def regional_table(rows):
    return pd.DataFrame(rows).rename(columns={'region':'Region','ordinary_20k':MODEL_NAMES['CONF_U20'],
        'candidate_mean':'Revised model (mean)','reduction_percent':'Reduction (%)'})

def policy_maps(policies, mask, title):
    """Panels are (role name, subtitle, action-index grid); one embedded case key."""
    fig,axes=plt.subplots(1,len(policies),figsize=(12,6.3),squeeze=False)
    axes=axes[0]
    fig.subplots_adjust(left=.055,right=.90,bottom=.34,top=.78,wspace=.24)
    fig.suptitle(title,y=.98,fontsize=17,weight='bold')
    for ax,(name,subtitle,policy) in zip(axes,policies):
        im=ax.imshow(np.ma.masked_where(~mask,policy)[1:,1:].T,origin='lower',aspect='auto',
            interpolation='nearest',extent=[.5,330.5,.5,10.5],cmap=ListedColormap(COLORS),norm=BoundaryNorm(np.arange(-.5,5.5),5))
        ax.set_xlim(335,-9);ax.set_ylim(.2,10.8)
        ax.set_xticks([330,200,100,0],['330','200','100','0'])
        ax.set_yticks([1,2,5,10]);ax.set_xlabel('Days until arrival');ax.set_ylabel('Rooms remaining')
        ax.set_title(name,fontsize=12,pad=31,weight='bold')
        ax.text(.5,1.035,subtitle,ha='center',transform=ax.transAxes,fontsize=9)
        ax.axvline(30.5,color='#222222',ls='--',lw=.9)
        ax.annotate('Final 30 days',xy=(30.5,6),xytext=(165,7.3),fontsize=8,
                    arrowprops={'arrowstyle':'->','lw':.8})
        for (days,rooms),(marker,_) in CASE_NAMES.items():
            ax.scatter(days,rooms,s=28,facecolors='white',edgecolors='black',lw=.8,zorder=3)
            offset={'A':(5,-14),'B':(5,6),'C':(-14,-14),'D':(-14,6)}[marker]
            ax.annotate(marker,(days,rooms),xytext=offset,textcoords='offset points',weight='bold',fontsize=9)
    cax=fig.add_axes([.925,.34,.016,.44])
    cb=fig.colorbar(im,cax=cax,ticks=range(5));cb.ax.set_yticklabels(PRICES)
    cb.set_label('Price offered',labelpad=8)
    for (days,rooms),(marker,name) in CASE_NAMES.items():
        idx=list(CASE_NAMES).index((days,rooms));x=.055+(idx%2)*.47;y=.225-(idx//2)*.045
        fig.text(x,y,case_label(days,rooms),fontsize=10)
    fig.text(.055,.105,'Compare A with B for inventory; A with C for time. D checks the final day.',fontsize=10)
    fig.text(.055,.06,'Colors show prices. Time counts down left to right; 0 marks arrival (no offer). Blank states are unreachable.',fontsize=9)
    fig.text(.055,.025,'Cases are separate operating situations, not a single selling trajectory.',fontsize=9)
    return fig

def explanation_figure(cases, scarcity=False):
    df=pd.DataFrame(cases);t,r,contrast=(200,2,'125-90') if scarcity else (1,1,'55-90')
    keys=['CONF_U_10_20000','CONF_C_10_100000']+([] if scarcity else ['CONF_C_12_100000'])
    fig,axes=plt.subplots(1,len(keys),figsize=(12,6),sharey=True)
    fig.subplots_adjust(left=.07,right=.98,bottom=.36,top=.71,wspace=.22)
    title=('With 200 days and two rooms, both models choose 90;\nthe simulator’s best choice is 125' if scarcity else
           'Last day, one room: why do the models favor different prices?')
    fig.suptitle(title,fontsize=15,weight='bold',y=.98)
    alt=125 if scarcity else 55
    fig.text(.5,.86 if scarcity else .91,f'Score for {alt} minus score for 90',ha='center',fontsize=12)
    for ax,key in zip(axes,keys):
        row=df[(df.run==key)&(df.days==t)&(df.rooms==r)&(df.contrast==contrast)].iloc[0]
        role=MODEL_NAMES['CONF_U20'] if key.startswith('CONF_U') else MODEL_NAMES['CONF_C100']
        if key.endswith('12_100000'):role+=' · another run'
        ax.set_title(f'{role}\nChooses {row.selected_price:g}',fontsize=11,pad=25,weight='bold')
        ax.text(.5,1.015,f'Seed {row.seed}',ha='center',transform=ax.transAxes,fontsize=9)
        bars=ax.bar([0,1.5,3,4.8],[row.reference_base,row.time_contribution,row.inventory_contribution,row.margin],
                    color=['#999999','#315b80','#39a2ae','#e58a50'])
        bars[-1].set_edgecolor('#333333');bars[-1].set_linewidth(1.2)
        ax.axvline(3.9,color='#aaaaaa',ls=':',lw=1);ax.axhline(0,color='black',lw=.7)
        ax.set_xticks([0,1.5,3,4.8],['Reference\nscore gap','Contribution\nfrom days\nremaining','Contribution\nfrom rooms\nremaining','Final\nscore gap'],fontsize=7.5)
        ax.set_xlim(-.8,5.6)
    axes[0].set_ylabel('Score gap (revenue divided by 90)',fontsize=10)
    fig.text(.07,.23,'Reference gap + time contribution + inventory contribution = final gap',fontsize=11,weight='bold')
    fig.text(.07,.17,f'Positive contributions push toward {alt}; a positive final gap means {alt} outranks 90.',fontsize=10)
    fig.text(.07,.12,'The outlined final bar is the sum, not another input. All five prices determine the actual choice.',fontsize=10)
    fig.text(.07,.06,f'Q({alt}) − Q(90), relative to 64 fixed reference states. Bars explain model scores, not actual revenue.',fontsize=9)
    return fig

def revenue_figure(conf, oracle, planner):
    fig,ax=plt.subplots(figsize=(12,6.5));fig.subplots_adjust(left=.09,right=.98,bottom=.30,top=.82)
    fig.suptitle('The revised model earns more than the baseline\nin all three confirmation runs',fontsize=17,weight='bold',y=.98)
    for x,name in enumerate(LEGACY_NAMES):
        part=conf[conf.configuration==name].sort_values('training_seed')
        for shift,(_,row) in zip([-.15,0,.15],part.iterrows()):
            ax.scatter(x+shift,row.exact_revenue,s=55,color=SEED_COLORS[row.training_seed],label=f'Seed {row.training_seed}' if x==0 else None)
        ax.scatter(x,part.exact_revenue.mean(),marker='_',s=450,color='black',label='Mean' if x==0 else None,zorder=4)
    ax.axhline(oracle,color='black',ls='--',label=f"{MODEL_NAMES['optimal']}: {oracle:.3f}")
    ax.axhline(planner,color='#39a2ae',ls=':',label=f"{MODEL_NAMES['planner']}: {planner:.3f}")
    ax.legend(loc='upper left',fontsize=9,ncol=2)
    ax.set_xticks(range(4),['Baseline model\n20,000 updates','Baseline with\nlonger training\n100,000 updates',
                           'Revised model\n100,000 updates','Revised model:\noriginal time scale\n100,000 updates'],fontsize=10)
    ax.set_ylabel('Expected revenue per selling period');ax.set_ylim(849,892)
    fig.text(.09,.17,"Each dot shows one seed’s result at the stated training endpoint. Short bars show the mean across three seeds.",fontsize=9)
    fig.text(.09,.125,'The longer-training baseline continues the corresponding 20,000-update run.',fontsize=10)
    fig.text(.09,.08,'Seeds 10, 11 and 12 share one new confirmation dataset in both revised conditions.',fontsize=10)
    fig.text(.09,.035,'Original time scale = linear time input; the comparison still uses a neural network, not linear regression.',fontsize=9)
    return fig

def response_figures(grids, mask, refs):
    fig,axes=plt.subplots(3,2,figsize=(12,10),sharey='row')
    fig.subplots_adjust(left=.09,right=.98,bottom=.12,top=.88,hspace=.45,wspace=.14)
    fig.suptitle('The revised model lowers prices as arrival approaches',fontsize=17,weight='bold',y=.98)
    fig.text(.5,.94,'Both models: seed 10. Start with the top row to see the price actually chosen.',ha='center',fontsize=11)
    for col,(name,key) in enumerate([(MODEL_NAMES['CONF_U20'],'CONF_U_10_20000'),(MODEL_NAMES['CONF_C100'],'CONF_C_10_100000')]):
        q=grids[key]
        for rooms in [1,5,10]:
            days=np.flatnonzero(mask[:,rooms]);label=f'{rooms} '+('room' if rooms==1 else 'rooms')
            axes[0,col].step(days,np.array(PRICES)[q[days,rooms].argmax(1)],where='mid',label=label)
            for ax,a in zip(axes[1:,col],[0,4]):ax.plot(days,q[days,rooms,a]-q[days,rooms,2],label=label)
        axes[0,col].set_title(name,fontsize=13,weight='bold')
        axes[0,col].set_ylim(50,130);axes[0,col].set_yticks(PRICES)
        for ax in axes[:,col]:
            ax.set_xlim(335,0);ax.set_xticks([330,200,100,0]);ax.set_xlabel('Days until arrival');ax.legend(fontsize=9)
    for ax,label in zip(axes[:,0],['Price offered','Score for 55 minus\nscore for 90','Score for 125 minus\nscore for 90']):ax.set_ylabel(label)
    axes[0,0].annotate('All three lines overlap at 90',xy=(180,90),xytext=(300,112),fontsize=9,arrowprops={'arrowstyle':'->'})
    fig.text(.09,.06,'Each line holds rooms fixed. Lower rows show pairwise score gaps, not the five-price decision.',fontsize=10)
    fig.text(.09,.025,'Scores use revenue divided by 90. Only reachable states are evaluated; 0 marks arrival.',fontsize=9)
    inventory,axs=plt.subplots(1,3,figsize=(12,5.5));inventory.subplots_adjust(left=.07,right=.98,bottom=.28,top=.73,wspace=.32)
    inventory.suptitle('Inventory affects scores even when the chosen price stays the same',fontsize=16,weight='bold',y=.98)
    inventory.text(.5,.90,MODEL_NAMES['CONF_C100']+' · seed 10',ha='center',fontsize=12)
    q=grids['CONF_C_10_100000']
    for days in sorted({s[0] for s in refs['cases']}):
        rooms=np.flatnonzero(mask[days]);label=f'{days} '+('day' if days==1 else 'days')
        axs[0].step(rooms,np.array(PRICES)[q[days,rooms].argmax(1)],where='mid',label=label)
        for ax,a in zip(axs[1:],[0,4]):ax.plot(rooms,q[days,rooms,a]-q[days,rooms,2],label=label)
    for ax,title in zip(axs,['Price offered','Score for 55 minus\nscore for 90','Score for 125 minus\nscore for 90']):
        ax.set(title=title,xlabel='Rooms remaining');ax.legend(fontsize=8)
    axs[0].set_ylim(50,130);axs[0].set_yticks(PRICES)
    axs[0].annotate('50, 128, 200 and 321 days:\nall choose 90',xy=(3,90),xytext=(1.3,112),fontsize=8,arrowprops={'arrowstyle':'->'})
    inventory.text(.07,.15,'Each line holds days fixed. Overlapping price lines mean the recommendations are identical.',fontsize=10)
    inventory.text(.07,.08,'The score gaps can change without changing the winning price. Scores use revenue divided by 90.',fontsize=9)
    return fig,inventory
