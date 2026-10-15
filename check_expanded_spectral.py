"""Independent identities and timing checks for the cross-market screen."""
import numpy as np
import pandas as pd
from expanded_spectral import measure_window, transform_panel, clique_homology, h0_filtration, gf2_rank, mathematical_battery
from spectral_core import analyze_window


def main():
    rng=np.random.default_rng(741)
    sample=pd.DataFrame(rng.normal(size=(60,8)),index=pd.date_range('2020-01-01',periods=60),columns=[f'a{i}' for i in range(8)])
    row,spectrum=measure_window(sample)
    old=analyze_window(sample)
    for key in ('lambda2','normalized_lambda2','mean_raw_correlation','mean_positive_weight','fiedler_ipr'):
        np.testing.assert_allclose(row[key],old[key],atol=1e-10,rtol=1e-10)
    assert len(spectrum['L_positive'])==8
    np.testing.assert_allclose(sum(spectrum['C']),8,atol=1e-12)
    assert row['correlation_rank']<=row['rank_upper_bound']
    print('PASS baseline agreement and complete spectra')

    dates=pd.to_datetime(['2020-01-01T00:00Z','2020-01-01T00:05Z','2020-01-01T00:15Z'])
    prices=pd.DataFrame({'x':[1.,2.,3.],'y':[-4.,0.,-2.]},index=dates)
    differences,gaps=transform_panel(prices,{'transform':'first_difference','bar_seconds':300})
    assert gaps==1 and differences.iloc[1].isna().all() and differences.iloc[0]['y']==4
    try: transform_panel(prices,{'transform':'arithmetic_return'})
    except ValueError: pass
    else: raise AssertionError('Signed prices must not become arithmetic returns')
    print('PASS signed prices and gap/availability policy')

    triangle=np.ones((3,3),dtype=int)-np.eye(3,dtype=int)
    tri=clique_homology(triangle)
    assert tri['betti1_graph']==1 and tri['betti1_clique']==0 and tri['triangles']==1
    square=np.array([[0,1,0,1],[1,0,1,0],[0,1,0,1],[1,0,1,0]])
    assert clique_homology(square)['betti1_clique']==1
    assert gf2_rank([0b011,0b110,0b101])==2
    print('PASS GF(2) ranks, graph cycles and filled-triangle topology')

    h0=h0_filtration(np.ones((4,4)))
    assert h0['finite_H0_deaths']==[0.,0.,0.] and len(h0['MST_edges'])==3
    print('PASS coincident-series zero-distance H0 merges')

    battery=mathematical_battery(sample)
    assert battery['permutation_group_check']['max_L_spectrum_difference']<1e-10
    for graph in battery['threshold_graphs']:
        assert graph['closed_walk_trace_A2']==2*graph['edges']
        assert graph['closed_walk_trace_A3']==6*graph['triangles']
    assert min(battery['operator_spectra']['L_signed'])>-1e-10
    print('PASS permutation invariance, matrix-power counts and signed PSD')


if __name__=='__main__': main()
