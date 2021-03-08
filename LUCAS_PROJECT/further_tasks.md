linesearch.py:
- autosummary must be updated
- scipy.linalg.norm statt norm
- deepcopy ersetzen!
- prepare_parameters must be simplified!
- replace root_scalar calls by a backtracking procedure?

nonlin.py
- autosummary must be updated
- avoid import of LineSearchWarning and warnings->warn
- avoid changing signature of nonlin_solve and _nonlin_line_search if possible
- avoid that parameters is passed as None -> avoid if parameter == None clause
- (Removed the usage of dx_next in nonlin for brevity)
- (Removed analysis things)
- why do we update jac_tol in the parameter (in line 338)
- (Removed if-clause to avoid updating the Jacobian twice if using BSC for brevity around line 347)
- Maybe we should move parts of scalar_search_rmt/bsc to _nonlin_line_search_. I mean phi and derphi are also defined there. Then ascalar_search_rmt/bsc would also look more like the other scalar_search options

_root.py
- Can we avoid chaning the call to root_nonlin_solve and nonlin_solve?