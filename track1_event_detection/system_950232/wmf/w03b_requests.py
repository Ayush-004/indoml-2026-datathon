"""At-lag requests for the audit: every accepted base hit.
  cache/req_valsyn_base.pkl  validation cross-fit decode (5-fold geography thresholds)
  cache/req_testsyn_base.pkl job2only test events that come from a decoded hit (k known)
Then: python w03_atlag.py valsyn cache/req_valsyn_base.pkl cache/atlag_valsyn_base.pkl 3 (same for testsyn)."""
import wlib

B = wlib.load_pickle(wlib.CACHE / 'base.pkl')
rv = {c: [(k, p) for a, b, k, v, p in dec] for c, dec in B['base_cf'].items() if dec}
rt = {c: [(r['k'], r['p']) for r in rows if r['k'] is not None] for c, rows in B['test_job2'].items()}
rt = {c: v for c, v in rt.items() if v}
wlib.dump_pickle(wlib.CACHE / 'req_valsyn_base.pkl', rv)
wlib.dump_pickle(wlib.CACHE / 'req_testsyn_base.pkl', rt)
print('valsyn requests', sum(len(v) for v in rv.values()), 'testsyn requests', sum(len(v) for v in rt.values()))
