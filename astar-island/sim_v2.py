"""Calibrated MC simulator v2 for Astar Island."""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence
import numpy as np

OCEAN = 10; PLAINS = 11; EMPTY = 0; SETTLEMENT = 1; PORT = 2; RUIN = 3; FOREST = 4; MOUNTAIN = 5
NUM_CLASSES = 6
TERRAIN_TO_CLASS = {10: 0, 11: 0, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}

DEFAULT_PARAMS = {
    "food_per_forest": 0.40, "base_food_production": 0.12, "food_production_noise": 0.8,
    "food_consumption_rate": 0.18, "max_food_storage_base": 1.6, "max_food_storage_per_pop": 0.35,
    "pop_growth_threshold": 1.3, "pop_growth_rate": 0.05, "pop_cap_base": 2.0, "pop_cap_per_forest": 1.5,
    "defense_growth_rate": 0.012, "tech_growth_rate": 0.006, "wealth_decay": 0.88,
    "port_dev_prob": 0.12, "port_dev_wealth_threshold": 0.3, "port_degrade_prob": 0.010,
    "longship_prob": 0.10, "longship_wealth_threshold": 0.5,
    "expansion_rate": 0.55, "expansion_pop_threshold": 1.2, "expansion_food_threshold": 0.7,
    "expansion_min_dist": 2, "expansion_max_dist": 5,
    "expansion_parent_pop_retain": 0.70, "expansion_child_pop_fraction": 0.30, "expansion_child_food": 0.40,
    "faction_aggression": 0.30, "raid_range": 5.0, "longship_range_mult": 2.5,
    "raid_desperation_factor": 0.5, "raid_loot_fraction": 0.45,
    "raid_casualty_defender": 0.55, "raid_casualty_attacker": 0.75, "conquer_prob": 0.08,
    "trade_activity": 0.5, "trade_range": 15.0, "fishing_bonus": 0.15, "trade_wealth_bonus": 0.025,
    "winter_severity": 0.28, "winter_pop_decay_min": 0.94, "winter_pop_decay_range": 0.04,
    "collapse_pop_threshold": 0.32, "collapse_food_threshold": 0.30, "collapse_food_prob": 0.50,
    "death_to_ruin_prob": 0.06, "death_to_forest_prob": 0.50, "death_adj_forest_threshold": 0,
    "forest_growth_rate": 0.0008, "forest_decay_base": 0.002, "forest_decay_near_sett": 0.004,
    "ruin_reclaim_rate": 0.12, "ruin_to_forest_rate": 0.025, "ruin_to_forest_base": 0.012,
    "ruin_to_plains_rate": 0.018, "ruin_age_factor_base": 2.0, "ruin_age_factor_decay": 0.3,
    "immutable_floor": 0.0005, "jeffreys_alpha": 0.5,
}

def _count_adjacent(grid, value):
    H, W = grid.shape
    match = (grid == value).astype(np.float32)
    padded = np.pad(match, 1, constant_values=0.0)
    count = np.zeros((H, W), dtype=np.float32)
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            if dy == 0 and dx == 0: continue
            count += padded[1+dy:H+1+dy, 1+dx:W+1+dx]
    return count

def _coastal_mask(grid):
    H, W = grid.shape
    ocean = (grid == OCEAN)
    padded = np.pad(ocean, 1, constant_values=False)
    coastal = np.zeros((H, W), dtype=bool)
    for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
        coastal |= padded[1+dy:H+1+dy, 1+dx:W+1+dx]
    return coastal & ~ocean & (grid != MOUNTAIN)

class SA:
    """Settlement parallel arrays."""
    __slots__ = ['x','y','pop','food','wealth','defense','port','alive','owner','tech','ship','n']
    def __init__(self, ss):
        n = len(ss)
        self.n = n
        self.x = np.array([s.get('x',0) for s in ss], dtype=np.int32)
        self.y = np.array([s.get('y',0) for s in ss], dtype=np.int32)
        self.pop = np.array([s.get('population',1.0) for s in ss], dtype=np.float64)
        self.food = np.array([s.get('food',1.0) for s in ss], dtype=np.float64)
        self.wealth = np.array([s.get('wealth',0.0) for s in ss], dtype=np.float64)
        self.defense = np.array([s.get('defense',0.5) for s in ss], dtype=np.float64)
        self.port = np.array([s.get('has_port',False) for s in ss], dtype=bool)
        self.alive = np.array([s.get('alive',True) for s in ss], dtype=bool)
        self.owner = np.array([s.get('owner_id',i) for i,s in enumerate(ss)], dtype=np.int32)
        self.tech = np.array([s.get('tech_level',0.0) for s in ss], dtype=np.float64)
        self.ship = np.array([s.get('has_longship',False) for s in ss], dtype=bool)
    def add(self, x, y, pop, food, wealth, defense, port, owner, tech):
        self.x=np.append(self.x,x); self.y=np.append(self.y,y)
        self.pop=np.append(self.pop,pop); self.food=np.append(self.food,food)
        self.wealth=np.append(self.wealth,wealth); self.defense=np.append(self.defense,defense)
        self.port=np.append(self.port,port); self.alive=np.append(self.alive,True)
        self.owner=np.append(self.owner,owner); self.tech=np.append(self.tech,tech)
        self.ship=np.append(self.ship,False); self.n+=1
    def compact(self):
        m = self.alive
        if m.all(): return
        for a in self.__slots__:
            if a=='n': continue
            setattr(self,a,getattr(self,a)[m])
        self.n = int(m.sum())

class NorseSimV2:
    def __init__(self, grid, setts, params=None):
        self.g0 = np.asarray(grid, dtype=np.int32)
        self.H, self.W = self.g0.shape
        self.setts0 = setts
        self.p = {**DEFAULT_PARAMS, **(params or {})}
        self._ocean = self.g0 == OCEAN
        self._mount = self.g0 == MOUNTAIN
        md = int(self.p["expansion_max_dist"])
        mn = int(self.p["expansion_min_dist"])
        self._exp_off = np.array([(dy,dx) for dy in range(-md,md+1) for dx in range(-md,md+1)
                                   if mn<=abs(dy)+abs(dx)<=md], dtype=np.int32)

    def _sync(self, g, s):
        for i in range(s.n):
            y,x = s.y[i],s.x[i]
            if not (0<=y<self.H and 0<=x<self.W): continue
            if s.alive[i]:
                g[y,x] = PORT if s.port[i] else SETTLEMENT
            elif g[y,x] in (SETTLEMENT, PORT):
                g[y,x] = RUIN

    def _growth(self, g, s, rng, coast):
        p = self.p; n = s.n; al = s.alive[:n]; ai = np.where(al)[0]
        if len(ai)==0: return
        af = _count_adjacent(g, FOREST)
        yc = np.clip(s.y[:n],0,self.H-1); xc = np.clip(s.x[:n],0,self.W-1)
        ff = af[yc,xc]*p["food_per_forest"]
        bf = p["base_food_production"]+ff
        s.food[:n] += bf*(0.5+p["food_production_noise"]*rng.random(n))*al
        s.food[:n] -= s.pop[:n]*p["food_consumption_rate"]*al
        mf = p["max_food_storage_base"]+s.pop[:n]*p["max_food_storage_per_pop"]
        s.food[:n] = np.minimum(s.food[:n], mf)
        sur = al & (s.food[:n]>p["pop_growth_threshold"])
        if sur.any():
            gr = np.minimum(s.food[:n]*p["pop_growth_rate"],0.2)*(0.5+rng.random(n))*sur
            s.pop[:n]+=gr; s.food[:n]-=gr*0.8
        mp = p["pop_cap_base"]+ff*p["pop_cap_per_forest"]
        s.pop[:n] = np.minimum(s.pop[:n], mp)
        s.defense[:n] = np.minimum(s.defense[:n]+p["defense_growth_rate"]*rng.random(n)*al, 1.5)
        s.tech[:n] = np.minimum(s.tech[:n]+p["tech_growth_rate"]*rng.random(n)*al, 2.5)
        s.wealth[:n] *= np.where(al, p["wealth_decay"], 1.0)
        for i in ai:
            y,x = s.y[i],s.x[i]
            if not s.port[i] and coast[y,x]:
                if s.wealth[i]>p["port_dev_wealth_threshold"] and rng.random()<p["port_dev_prob"]*(1+s.tech[i]):
                    s.port[i]=True; g[y,x]=PORT
            if not s.ship[i] and s.port[i]:
                if s.wealth[i]>p["longship_wealth_threshold"] and rng.random()<p["longship_prob"]*(1+s.tech[i]*0.3):
                    s.ship[i]=True
            if s.port[i] and rng.random()<p["port_degrade_prob"]:
                s.port[i]=False; g[y,x]=SETTLEMENT
        exp = ai[(s.pop[ai]>p["expansion_pop_threshold"])&(s.food[ai]>p["expansion_food_threshold"])]
        if len(exp)>0:
            occ = np.zeros((self.H,self.W),dtype=bool)
            for i in ai:
                y,x=s.y[i],s.x[i]
                occ[max(0,y-1):min(self.H,y+2),max(0,x-1):min(self.W,x+2)]=True
            for i in exp:
                if rng.random()>=p["expansion_rate"]: continue
                y,x=s.y[i],s.x[i]
                cy=y+self._exp_off[:,0]; cx=x+self._exp_off[:,1]
                v=(cy>=0)&(cy<self.H)&(cx>=0)&(cx<self.W)
                cy,cx=cy[v],cx[v]
                if len(cy)==0: continue
                t=g[cy,cx]
                ok=((t==PLAINS)|(t==EMPTY)|(t==FOREST))&~occ[cy,cx]
                cy,cx=cy[ok],cx[ok]
                if len(cy)==0: continue
                j=rng.integers(len(cy)); ny,nx=int(cy[j]),int(cx[j])
                s.add(nx,ny,s.pop[i]*p["expansion_child_pop_fraction"],
                      s.food[i]*p["expansion_child_food"],s.wealth[i]*0.2,
                      0.4,False,s.owner[i],s.tech[i]*0.6)
                s.pop[i]*=p["expansion_parent_pop_retain"]; s.food[i]*=0.6; s.wealth[i]*=0.7
                g[ny,nx]=SETTLEMENT
                occ[max(0,ny-1):min(self.H,ny+2),max(0,nx-1):min(self.W,nx+2)]=True

    def _conflict(self, g, s, rng):
        p=self.p; ai=np.where(s.alive[:s.n])[0]
        if len(ai)<2: return
        cy,cx,ow=s.y[ai],s.x[ai],s.owner[ai]
        for ii in range(len(ai)):
            i=ai[ii]
            if not s.alive[i]: continue
            desp=max(0.0,1.0-s.food[i])*p["raid_desperation_factor"]
            if rng.random()>p["faction_aggression"]*0.3+desp: continue
            rr=p["raid_range"]*(p["longship_range_mult"] if s.ship[i] else 1.0)
            d=np.abs(cy-s.y[i])+np.abs(cx-s.x[i])
            m=(d<=rr)&(d>0)&(ow!=s.owner[i])
            ti=[j for j in np.where(m)[0] if s.alive[ai[j]]]
            if not ti: continue
            tj=ai[ti[rng.integers(len(ti))]]
            a=s.pop[i]*s.defense[i]*(1+0.2*s.tech[i])*(0.6+0.8*rng.random())
            d2=s.pop[tj]*s.defense[tj]*(1+0.2*s.tech[tj])*(0.6+0.8*rng.random())
            if a>d2:
                lf=s.food[tj]*p["raid_loot_fraction"]; lw=s.wealth[tj]*p["raid_loot_fraction"]
                s.food[i]+=lf; s.wealth[i]+=lw; s.food[tj]-=lf; s.wealth[tj]-=lw
                s.pop[tj]*=p["raid_casualty_defender"]; s.defense[tj]*=0.6
                if rng.random()<p["conquer_prob"]: s.owner[tj]=s.owner[i]
            else:
                s.pop[i]*=p["raid_casualty_attacker"]; s.defense[i]*=0.8

    def _trade(self, g, s, rng):
        p=self.p; ai=np.where(s.alive[:s.n])[0]
        pi=ai[s.port[ai]]
        if len(pi)>0:
            s.food[pi]+=p["fishing_bonus"]*(0.5+rng.random(len(pi)))
            s.wealth[pi]+=p["trade_wealth_bonus"]*(0.5+rng.random(len(pi)))
        if len(pi)<2 or p["trade_activity"]<0.05: return
        for ii in range(len(pi)):
            a=pi[ii]
            for jj in range(ii+1,len(pi)):
                b=pi[jj]
                d=abs(int(s.y[a])-int(s.y[b]))+abs(int(s.x[a])-int(s.x[b]))
                if d>p["trade_range"]: continue
                tp=p["trade_activity"]*0.5*(1.5 if s.owner[a]==s.owner[b] else 1.0)
                if rng.random()>tp: continue
                tv=0.15*p["trade_activity"]*(0.5+rng.random())
                s.wealth[a]+=tv; s.wealth[b]+=tv; s.food[a]+=tv*0.5; s.food[b]+=tv*0.5

    def _winter(self, g, s, rng):
        p=self.p; n=s.n; al=s.alive[:n]; ai=np.where(al)[0]
        if len(ai)==0: return
        fl=p["winter_severity"]*(0.4+0.6*rng.random(n))*(1.0+s.pop[:n]*0.15)
        s.food[:n]-=fl*al
        st=al&(s.food[:n]<0)
        if st.any():
            pl=np.minimum(np.abs(s.food[:n])*0.4,s.pop[:n]*0.5)
            s.pop[:n]-=pl*st; s.food[:n]=np.where(st,0.0,s.food[:n])
        dc=p["winter_pop_decay_min"]+p["winter_pop_decay_range"]*rng.random(n)
        s.pop[:n]*=np.where(al,dc,1.0)
        afg=_count_adjacent(g,FOREST)
        for i in ai:
            if not s.alive[i]: continue
            if s.pop[i]<p["collapse_pop_threshold"] or \
               (s.food[i]<p["collapse_food_threshold"] and rng.random()<p["collapse_food_prob"]*p["winter_severity"]):
                s.alive[i]=False; y,x=s.y[i],s.x[i]
                if rng.random()<p["death_to_ruin_prob"]: g[y,x]=RUIN
                elif afg[y,x]>=p["death_adj_forest_threshold"] and rng.random()<p["death_to_forest_prob"]:
                    g[y,x]=FOREST
                else: g[y,x]=PLAINS
                if s.pop[i]>0:
                    nb=s.alive[:s.n]&(s.owner[:s.n]==s.owner[i])&(np.abs(s.y[:s.n]-y)+np.abs(s.x[:s.n]-x)<=5)
                    nb[i]=False; ni=np.where(nb)[0]
                    if len(ni)>0: s.pop[ni]+=s.pop[i]*0.5/len(ni)
                s.pop[i]=0.0

    def _env(self, g, s, rng, coast):
        p=self.p; ry,rx=np.where(g==RUIN); ai=np.where(s.alive[:s.n])[0]
        for r,c in zip(ry,rx):
            k=(int(r),int(c))
            if k not in self._ra: self._ra[k]=0
            self._ra[k]+=1
        for k in list(self._ra.keys()):
            if g[k[0],k[1]]!=RUIN: del self._ra[k]
        fa=_count_adjacent(g,FOREST)
        for r,c in zip(ry,rx):
            ri,ci=int(r),int(c); recl=False
            if len(ai)>0:
                d=np.abs(s.y[ai]-ri)+np.abs(s.x[ai]-ci)
                nb=ai[d<=5]
                for si in nb:
                    sf=min(s.pop[si],2.0)*0.5
                    if rng.random()<p["ruin_reclaim_rate"]*sf:
                        ic=bool(coast[ri,ci])
                        s.add(ci,ri,s.pop[si]*0.2,s.food[si]*0.2,s.wealth[si]*0.1,0.3,ic,s.owner[si],s.tech[si]*0.3)
                        s.pop[si]*=0.8; s.food[si]*=0.8
                        g[ri,ci]=PORT if ic else SETTLEMENT; recl=True; break
            if not recl:
                ra=self._ra.get((ri,ci),10)
                af2=max(0.3,p["ruin_age_factor_base"]/(1.0+ra*p["ruin_age_factor_decay"]))
                afc=fa[ri,ci]
                fp=af2*(p["ruin_to_forest_rate"]*min(afc,3) if afc>0 else p["ruin_to_forest_base"])
                pp=af2*p["ruin_to_plains_rate"]
                rv=rng.random()
                if rv<fp: g[ri,ci]=FOREST
                elif rv<fp+pp: g[ri,ci]=PLAINS
        em=(g==PLAINS)|(g==EMPTY); gm=em&(fa>0); gy,gx=np.where(gm)
        if len(gy)>0:
            pr=p["forest_growth_rate"]*np.minimum(fa[gy,gx],3)
            cv=rng.random(len(gy))<pr; g[gy[cv],gx[cv]]=FOREST
        fy,fx=np.where(g==FOREST)
        if len(fy)>0:
            sa2=_count_adjacent(g,SETTLEMENT)+_count_adjacent(g,PORT)
            pr=p["forest_decay_base"]+p["forest_decay_near_sett"]*sa2[fy,fx]
            cv=rng.random(len(fy))<pr; g[fy[cv],fx[cv]]=PLAINS

    def run(self, seed=None):
        rng=np.random.default_rng(seed); g=self.g0.copy(); s=SA(self.setts0); self._ra={}
        self._sync(g,s)
        for yr in range(50):
            co=_coastal_mask(g)
            self._growth(g,s,rng,co); self._conflict(g,s,rng); self._trade(g,s,rng)
            self._winter(g,s,rng); self._env(g,s,rng,co); self._sync(g,s)
            if yr%10==9: s.compact()
        g[self._ocean]=OCEAN; g[self._mount]=MOUNTAIN
        return g

    def run_classes(self, seed=None):
        g=self.run(seed); cg=np.zeros_like(g)
        for code,cls in TERRAIN_TO_CLASS.items(): cg[g==code]=cls
        return cg

def run_monte_carlo(grid, setts, params=None, n_runs=200, seeds=None):
    m={**DEFAULT_PARAMS,**(params or {})}
    sim=NorseSimV2(grid,setts,m); H,W=sim.H,sim.W
    if seeds is None: seeds=list(range(n_runs))
    counts=np.zeros((H,W,NUM_CLASSES),dtype=np.int32)
    for sd in seeds:
        cg=sim.run_classes(sd)
        np.add.at(counts,(np.arange(H)[:,None],np.arange(W)[None,:],cg),1)
    alpha=m["jeffreys_alpha"]
    probs=(counts.astype(np.float64)+alpha)/(n_runs+NUM_CLASSES*alpha)
    fl=m["immutable_floor"]
    probs[sim._ocean]=fl; probs[sim._ocean,0]=1.0
    probs[sim._mount]=fl; probs[sim._mount,5]=1.0
    probs=np.maximum(probs,1e-6); probs/=probs.sum(axis=2,keepdims=True)
    return probs

def score_against_gt(pred, gt):
    eps=1e-10; p=np.clip(gt,eps,1.0); q=np.clip(pred,eps,1.0)
    ent=-np.sum(p*np.log(p+eps),axis=-1); kl=np.sum(p*np.log(p/q),axis=-1)
    dyn=ent>0.01
    if not dyn.any(): return 100.0
    wkl=np.sum(ent[dyn]*kl[dyn])/np.sum(ent[dyn])
    return max(0.0,min(100.0,100.0*np.exp(-3.0*wkl)))

def quick_calibrate(init_path, gt_paths, param_grid, n_runs=100, seed_indices=None):
    import json
    init_data=json.load(open(init_path)); results=[]
    for pi,ov in enumerate(param_grid):
        total=0.0; count=0
        for si,gp in gt_paths.items():
            if seed_indices is not None and si not in seed_indices: continue
            gtd=json.load(open(gp)); gt=np.array(gtd['ground_truth'])
            st=init_data['initial_states'][si]
            pred=run_monte_carlo(st['grid'],st['settlements'],ov,n_runs=n_runs)
            total+=score_against_gt(pred,gt); count+=1
        avg=total/max(count,1); results.append((avg,ov))
        print(f"  Config {pi}: score={avg:.2f}")
    results.sort(key=lambda x:-x[0]); return results

if __name__=="__main__":
    import json,time
    cache="/Users/andreashatlem/Projects/nmiai/astar-island/cache"
    init_data=json.load(open(f"{cache}/r9_init.json"))
    gt_data=json.load(open(f"{cache}/r9_gt_s0.json"))
    gt=np.array(gt_data['ground_truth'])
    state=init_data['initial_states'][0]
    grid=state['grid']; setts=state['settlements']
    print("="*60)
    print("simulator_v2 DEFAULT (200 MC)...")
    t0=time.time(); pred=run_monte_carlo(grid,setts,None,n_runs=200); t1=time.time()
    score=score_against_gt(pred,gt)
    print(f"Score: {score:.2f}  Time: {t1-t0:.2f}s")
    g0=np.array(grid)
    for code,name in [(1,'Settlement'),(2,'Port'),(4,'Forest'),(11,'Plains')]:
        mask=g0==code
        if mask.any():
            mp=pred[mask].mean(axis=0); mg=gt[mask].mean(axis=0)
            print(f"  {name:10s} pred: {' '.join(f'{v:.3f}' for v in mp)}")
            print(f"  {'':10s} gt:   {' '.join(f'{v:.3f}' for v in mg)}")
    print(); print("="*60); print("Parameter search (10 configs)...")
    configs=[
        {},
        {"expansion_rate":0.60,"winter_severity":0.35},
        {"expansion_rate":0.40,"winter_severity":0.42},
        {"expansion_rate":0.55,"winter_severity":0.40,"food_consumption_rate":0.20},
        {"expansion_rate":0.45,"winter_severity":0.36,"death_to_forest_prob":0.50},
        {"expansion_rate":0.50,"faction_aggression":0.35,"raid_casualty_defender":0.50},
        {"expansion_rate":0.55,"winter_severity":0.35,"forest_growth_rate":0.0005},
        {"expansion_rate":0.50,"winter_severity":0.40,"collapse_pop_threshold":0.45},
        {"expansion_rate":0.60,"winter_severity":0.30,"death_to_ruin_prob":0.04,"death_to_forest_prob":0.55},
        {"expansion_rate":0.55,"winter_severity":0.28,"food_per_forest":0.35,"base_food_production":0.10},
    ]
    gt_paths={0:f"{cache}/r9_gt_s0.json"}
    results=quick_calibrate(f"{cache}/r9_init.json",gt_paths,configs,n_runs=100,seed_indices=[0])
    print(f"\nBest: score={results[0][0]:.2f}")
    for s,p in results: print(f"  {s:.2f}: {str(p)[:80] if p else 'DEFAULT'}")
