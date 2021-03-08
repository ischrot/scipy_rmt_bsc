## Tasks and information for Software Practical "Adding RMT and BSC to SciPy"

#### Whenever you have a question or feel in doubt, feel free to contact me :)

### Next steps as agreed on the meeting on December 04:
	- Take a look at the test problems in test_nonlin.py and try to formulate those problems as rootfinding problems suitable for root.
	- Test the current code with those problems and store the results (including details like the iterates, step sizes chosen etc.) for later
	- Start cleaning the code such that only the essential parts are left
	- Adjust the code such that the design is similar to the one of the other strategies and is compliant with the good coding practices from Python which can be found here https://www.python.org/dev/peps/pep-0008/ or here https://docs.python-guide.org/writing/style/
		- A particular thing here is to check whether there are more elegant ways of storing quantities for the next call to the step size strategy than storing them in the parameters=line_search_options dictionary. (Unless the other stepsize strategies do similar things?)
		- A related thing is to have meaningful option names, reasonable default values (feel free to ask me again for those) and to simplify/reduce the available options.
	- (Make sure to check every now and then whether the code still returns the same results as before)
	- Do the documentation.
In my opinion this state then possibly is a good final state for this practical. Becnhmarking is probably a rather extensive topic.


### Goal: 
Add the RMT and BSC as originally suggested in the respective publications to SciPy, except for the following small modifications.
#### Modifications:
	- eta_upper of the RMT does not have an upper bound
	- Both the RMT and the BSC should have the option to switch between finding the minimal feasible step size and any feasible step size. This is handled by the options "min_search"
	- The monotone iterations of the BSC are not implemented
	- The minimum step size amin is considered.
	- To be continued possibly :D

### General Information:
	- The file Changelog.txt contains a list of (hopefully) all modifications I did in files of the SciPy package. Note that the line references refer to the modified files and not the original lines.
	- Besides implementing the RMT and the BSC I also implemented the possibility to use exact Hessians instead of BFGS or Krylov approximations. However, it is not planned to submit those changes, too. Moreover, I made some changes to SciPy files which were only relevant for the analysis of the performance of the stepsize strategies. Finally, I added quite a few variations of and options to the RMT and the BSC. Therefore, you should check which changes are relevant for the goals mentioned above.
	- The file test.py provides code to test the strategies + stuff for plotting.

### Tasks:
#### Getting started:
	- Set up a Python environment with Python 3.5.6 and my modified SciPy package SciPy_MT. This environment has the purpose to check that modifications you are going to do to satisfy all contribution criteria in another environment did not accidentally change the outcome of the strategies.
	- Set up a second environment with the most recent version of Python compatible with SciPy and the most recent version of SciPy. This should be the one where you do all modifications.
	- Look for some benchmark problems that you use to verify that your modifications yield the same results as my code. Please use definitely more than a single problem! I used the benchmark library CUTEst interfaced using the package pycutest. However, that was kind of complicated to realise. So, maybe look for alternatives.
	- Transfer my modifications relevant for our goals to the most recent version of SciPy. Probably SciPy has not changed much, so probably this should be rather straightforward.
#### Preparing the code for its contribution:
From here on the tasks are admittedly more vague. The primary guide here are SciPy's rules for contributing code which can be found here: https://docs.scipy.org/doc/scipy/reference/hacking.html
I suggest to start with the following things:
	- Write a proper documentation of the respective functions in linesearch.py similar to the documentations of the already existing linesearch strategies and the one started for the RMT. Not only is this a necessary thing to do, but it surely helps to understand the code and its variables.
	- Adjust the code such that the design is similar to the one of the other strategies and is compliant with the good coding practices from Python which can be found here https://www.python.org/dev/peps/pep-0008/ or here https://docs.python-guide.org/writing/style/
		- A particular thing here is to check whether there are more elegant ways of storing quantities for the next call to the step size strategy than storing them in the parameters=line_search_options dictionary. (Unless the other stepsize strategies do similar things?)
		- A related thing is to have meaningful option names, reasonable default values (feel free to ask me again for those) and to simplify/reduce the available options.
	- Write unit tests
	- Create benchmarks
	- Do the documentation if you have not done so already
	- Check the code style

