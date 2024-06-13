## Scripts: Colony pattern development of a synthetic bistable switch



Pan Chu (储攀)1,2†, Jingwen Zhu (朱静雯)1†, Zhixin Ma (马智鑫)1,2, Xiongfei Fu (傅雄飞)1,2* 

**Affiliations:**

1Key Laboratory for Quantitative Synthetic Biology, Shenzhen Institute of Synthetic Biology, Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences, Shenzhen 518055, China

2University of Chinese Academy of Sciences, Beijing 100049, China

†These authors contributed equally: Pan Chu, Jingwen Zhu

*Corresponding author. Email: [xiongfei.fu@siat.ac.cn](mailto:xiongfei.fu@siat.ac.cn)



-----------------

### Scripts structure

``` 
|-------- Scripts_for_colony_pattern_development
	|-------- An_agent-based_model_for_colony_establishment
		|-------- SSA_Colony                             // Scripts for agent-based colony model
			|-------- SSAColony_R2G_BatchRun.py           // Automation for compile C++ code and collecting data.
			|-------- colony_agent_based_SSA              // This folder contain the scorce codes for agent-based colony
																		 // with the stochastic simulation algorithm for the toggle switch.
		|--------- SSA_Toggle                            // Scripts for the stochastic simulation algorithm for the toggle
			|--------- src                                // C code for the stochastic simulation algorithm
			|--------- environment.yml                    // conda enviroment config file
			|--------- kde_scatter_plot.py                // Plot cofig
			|--------- sciplot.py                         // Plot cofig
			|--------- SSA_population_transition_rate.py  // main file
	|-------- A_continuum_model_for_colony_expansion	 // Scripts for continuum model for colny expansion
		|-------- colonyPhaseFieldUtilities.py           // utility functions for numerical simulation
		|-------- environment.yml								 // conda enviroment config file for continuum model
		|-------- sciplot.py										 // Plot cofig
		|-------- PFM_colony_CrosFed.py                  // main file
```

