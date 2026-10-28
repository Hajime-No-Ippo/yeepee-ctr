from lib import *
rng=np.random.RandomState(0)
for c in (25,23):
    F=load_cut(c); P=np.load(f'cache_prop_c{c}.npy'); names=json.load(open('prop_names.json'))
    print(c,P.shape,flush=True)
    noise=np.repeat(rng.randn(len(F['y'])//10,1).astype(np.float32),10,1).reshape(-1,1)
    report('null_event_noise',F,noise)
    report('prop_all',F,P)
