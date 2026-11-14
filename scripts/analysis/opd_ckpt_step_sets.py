"""Incremental-step top-K overlap: base->s35, s35->s40, s40->s45."""
import pandas as pd, numpy as np
from pathlib import Path
base=Path('outputs/opd_context_change')
s=pd.read_csv(base/'state_change.csv')
c=pd.read_csv(base/'gap_predict_controls_states.csv')
t=pd.read_csv(base/'token_change.csv')
m=s.merge(c[['token','cluster','kl_base_s35','kl_s35_s40','kl_s40_s45']],on=['token','cluster'],how='inner',suffixes=('','_c'))
assert len(m)==len(s)
def wmean(g,col): return float((g['n']*g[col]).sum()/max(g['n'].sum(),1))
tok_kl=m.groupby('token_id').apply(lambda g: pd.Series({
    'kl_s35_s40':wmean(g,'kl_s35_s40'),
    'kl_s40_s45':wmean(g,'kl_s40_s45'),
}),include_groups=False).reset_index()
t=t.merge(tok_kl,on='token_id',how='left')
steps_h=['dcos_base_s35','dcos_s35_s40','dcos_s40_s45']
steps_kl=['kl_base_s35','kl_s35_s40','kl_s40_s45']
def sets(df,col,k): return set(df.nlargest(k,col).index)
def jac(a,b): return len(a&b)/max(len(a|b),1)
def report(df,cols,k,label):
    ss=[sets(df,col,k) for col in cols]
    all3=ss[0]&ss[1]&ss[2]; union=ss[0]|ss[1]|ss[2]
    row={'granularity':label,'k':k,'all3':len(all3),'union':len(union),
         'j12':jac(ss[0],ss[1]),'j13':jac(ss[0],ss[2]),'j23':jac(ss[1],ss[2]),
         'only_step1':len(ss[0]-ss[1]-ss[2]),'only_step2':len(ss[1]-ss[0]-ss[2]),'only_step3':len(ss[2]-ss[0]-ss[1]),
         's1s2_not3':len((ss[0]&ss[1])-ss[2]),'s1s3_not2':len((ss[0]&ss[2])-ss[1]),'s2s3_not1':len((ss[1]&ss[2])-ss[0])}
    print(label,k,row); return row,ss
rows=[]
for k in [25,50,100]:
    r,_=report(s,steps_h,k,'state hidden'); rows.append(r)
    r,_=report(m,steps_kl,k,'state KL'); rows.append(r)
for k in [20,30,100]:
    r,_=report(t,steps_h,k,'token hidden'); rows.append(r)
    r,_=report(t,steps_kl,k,'token KL'); rows.append(r)
pd.DataFrame(rows).to_csv(base/'step_set_overlap.csv',index=False)
for df,cols,name,key in [(s,steps_h,'state hidden',['token','cluster']),(m,steps_kl,'state KL',['token','cluster'])]:
    print('\n===',name,'top100 ===')
    ss=[sets(df,col,100) for col in cols]
    all3=ss[0]&ss[1]&ss[2]
    print('all3',len(all3),'union',len(ss[0]|ss[1]|ss[2]),'pairwise',len(ss[0]&ss[1]),len(ss[0]&ss[2]),len(ss[1]&ss[2]))
    for i,ck in enumerate(['base->s35','s35->s40','s40->s45']):
        uniq=ss[i]-ss[(i+1)%3]-ss[(i+2)%3]
        print(f'unique to {ck}:',len(uniq))
        if uniq:
            x=df.loc[sorted(uniq),['token','cluster','n']+cols].copy(); x['token']=x.token.map(repr)
            print(x.to_string(index=False))
for cols,name in [(steps_h,'token hidden'),(steps_kl,'token KL')]:
    print('\n===',name,'top100 ===')
    ss=[sets(t,col,100) for col in cols]
    all3=ss[0]&ss[1]&ss[2]
    print('all3',len(all3),'union',len(ss[0]|ss[1]|ss[2]),'pairwise',len(ss[0]&ss[1]),len(ss[0]&ss[2]),len(ss[1]&ss[2]))
    for i,ck in enumerate(['base->s35','s35->s40','s40->s45']):
        uniq=ss[i]-ss[(i+1)%3]-ss[(i+2)%3]
        if uniq:
            x=t.loc[sorted(uniq),['token','n']+cols].copy(); x['token']=x.token.map(repr)
            print('unique to',ck,len(uniq)); print(x.to_string(index=False))
