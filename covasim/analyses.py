import pandas as pd

def analyze_transtree(sim):
    ''' Analyzes basic statistics from a transmission tree. '''

    sim.people.make_detailed_transtree()
    tt = sim.people.transtree.detailed

    detailed = filter(None, tt)

    df = pd.DataFrame(detailed)
    df = df.loc[df['layer'] != 'seed_infection']

    for lk in 'hswc':
        infs = [d for d in tt if d and d['layer']==lk]
        print(lk, len(infs))

    return df