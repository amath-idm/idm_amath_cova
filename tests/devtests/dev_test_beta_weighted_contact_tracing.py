"""
random, hybrid, and synthpops look ok.  clustered sees very little effect.
"""
import covasim as cv

if __name__ == "__main__":

    for pop_type in ['random', 'hybrid', 'clustered', 'synthpops']:

        sim = cv.Sim(pars={'pop_type': pop_type, 'diag_factor': 1.0, 'pop_infected': 20})
        beta_layer = list(sim['beta_layer'].keys())
        sim.update_pars({'quar_eff': {k: 1.0 for k in beta_layer}})
        closures = [cv.change_beta(len(beta_layer)*[30], len(beta_layer)*[0.75], layers=beta_layer)]
        testing = [cv.test_num(daily_tests=sim['n_days']*[100])]

        tracing1 = [cv.contact_tracing(trace_probs={k:1.0 for k in beta_layer},
                                                  trace_time={k:0 for k in beta_layer})]
        tracing2 = [cv.beta_weighted_contact_tracing(trace_probs={k:1.0 for k in beta_layer},
                                                  trace_time={k:0 for k in beta_layer})]
        scenarios={
            'contact_tracing':{
                'name': 'contact_tracing',
                'pars': {
                    'interventions': closures + testing + tracing1
                },
            },
            'beta_weighted_contact_tracing':{
                'name': 'beta_weighted_contact_tracing',
                'pars':{
                    'interventions': closures + testing + tracing2
                },
            }
        }
        scen = cv.Scenarios(sim=sim, scenarios=scenarios, metapars={'n_runs': 8, 'noise':0.0})
        scen.run(debug=False, verbose=False)
        print(f'Plotting {pop_type}, close figure to continue')
        fig = scen.plot(to_plot=['cum_quarantined', 'new_infections'])
